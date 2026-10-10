"""Requested P1 checks, including exact no-op/rollback on an actual DeNICE."""
import copy
import json
import io
from pathlib import Path
import numpy as np
import torch

from appliance.config import Rejected
from appliance.direct_head_contract import binding, VERSION
from appliance.direct_head_codec import encode, decode, export_head
from appliance.direct_head_aggregation import aggregate
from appliance.direct_head_transaction import HeadTransaction, assert_protected
from appliance.direct_head_verification import accept, select_eta, validate_receipts
from appliance.state import complete_hash
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.training.checkpoint_state import snapshot_denice_state
from eval_checkpoint import _make_denice_client_model


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = DeNICEModel((1, 39), 34)
    model.unit_ranks['fc2'][7] = 0
    model.freeze_masks['fc2'] = np.zeros(34, bool)
    router = ContextDetector(router_mode='binary_cosine')
    router.episode_classes = {0: [0, 1], 1: [6]}
    donor, dr = copy.deepcopy(model), copy.deepcopy(router)
    donor.unit_ranks['fc2'][7] = 2
    with torch.no_grad():
        donor.fc2.weight[7].fill_(4)
        donor.weight_masks['fc2'][7].fill_(0)
        donor.weight_masks['fc2'][7, :2] = 1
        donor.fc2.weight[7, 1] = 0  # Learned zero must still contribute.
        donor.fc2.bias[7] = 3
        donor.bias_masks['fc2'][7] = 0
        model.weight_masks['fc2'][7].fill_(0)
        model.weight_masks['fc2'][7, 2] = 1
    original = complete_hash(model, router)
    before_donor = complete_hash(donor, dr)
    p = export_head(donor, dr, 54, 1, 7, {'graph': 'locked'})
    b = encode(p)
    decoded = decode(b)
    checks = {}
    def ok(name, condition):
        assert condition, name
        checks[name] = True
    def rejected(name, fn):
        try:
            fn()
        except Rejected:
            checks[name] = True
        else:
            raise AssertionError(name)
    ok('codec_exact_FP32_masks_bias', np.array_equal(p['weight'], decoded['weight']) and p['bias'] == decoded['bias'] and np.array_equal(p['mask'], decoded['mask']))
    rejected('corrupt_packet_rejected', lambda: decode(b[:-1]+bytes([b[-1]^1])))
    p2 = copy.deepcopy(decoded)
    p2['metadata']['donor'] = 51
    p2['mask'][:] = 0
    p2['mask'][0] = 1
    p2['weight'][0] = 8
    p2['bias_mask'] = 1
    p2['bias'] = 5
    agg = aggregate([decoded, p2], {54: .6, 51: .4})
    ok('coordinate_normalization', np.isclose(agg['weight'][0], 5.6))
    ok('learned_zero_contributes', agg['contributed'][1] and agg['weight'][1] == 0)
    ok('no_contributor_marked_not_applicable', not agg['contributed'][2])
    ok('bias_separate_contributors', agg['bias'] == 5)
    rejected('duplicate_donor_rejected', lambda: aggregate([decoded, decoded], {54: 1}))
    rejected('negative_alpha_rejected', lambda: aggregate([decoded], {54: -1}))
    wrong = copy.deepcopy(p2)
    wrong['metadata']['class_id'] = 8
    rejected('target_mismatch_rejected', lambda: aggregate([decoded, wrong], {54: .6, 51: .4}))
    wrong = copy.deepcopy(decoded)
    wrong['weight'][0] = np.nan
    rejected('nonfinite_rejected', lambda: encode(wrong))
    wrong = copy.deepcopy(decoded)
    wrong['mask'][0] = 2
    rejected('nonbinary_mask_rejected', lambda: encode(wrong))
    tx = HeadTransaction(model, router, 7, 1)
    zero = tx.stage(agg, 0)
    ok('eta_zero_exact_noop', complete_hash(*zero) == original)
    candidate = tx.stage(agg, .5)
    mask = HeadTransaction(model, router, 7, 1).stage(agg, .5, mask_only=True)
    ok('no_contributor_preserves_raw_weight_and_mask', torch.equal(candidate[0].fc2.weight[7, 2:], model.fc2.weight[7, 2:]) and torch.equal(candidate[0].weight_masks['fc2'][7, 2:], model.weight_masks['fc2'][7, 2:]))
    ok('effective_old_semantics', np.isclose(float(candidate[0].fc2.weight[7, 0].detach()), 2.8))
    ok('mask_only_keeps_original_raw_weights', torch.equal(mask[0].fc2.weight, model.fc2.weight))
    ok('mask_control_same_masks_ranks_registration', torch.equal(mask[0].weight_masks['fc2'], candidate[0].weight_masks['fc2']) and np.array_equal(mask[0].unit_ranks['fc2'], candidate[0].unit_ranks['fc2']) and mask[1].episode_classes == candidate[1].episode_classes)
    assert_protected((model, router), candidate, 7, 1)
    ok('protected_state_exact', True)
    rejected('commit_before_verification_rejected', tx.commit)
    rejected('wrong_function_receipt_rejected', lambda: tx.verify(dict(status='EMPIRICAL_CURRENT_PASS', candidate_function='wrong')))
    tx.verify(dict(status='EMPIRICAL_CURRENT_PASS', candidate_function=complete_hash(*candidate)))
    committed, _ = tx.commit()
    ok('isolated_commit_exact', complete_hash(*committed) == complete_hash(*candidate))
    capsule = dict(config=dict(input_shape=(1, 39), num_classes=34, denice_router_mode='binary_cosine'),
        client_model_states={3: committed[0].state_dict()},
        client_algorithm_states={3: {'denice': snapshot_denice_state(*committed)}})
    stream = io.BytesIO()
    torch.save(capsule, stream)
    restored = _make_denice_client_model(torch.load(io.BytesIO(stream.getvalue()), weights_only=False), 3, 'cpu')
    ok('candidate_save_restore_complete_hash', complete_hash(*restored) == complete_hash(*committed))
    ok('rollback_complete_state', complete_hash(*tx.rollback()) == original)
    ok('live_receiver_and_donor_unchanged', complete_hash(model, router) == original and complete_hash(donor, dr) == before_donor)
    mature = copy.deepcopy(model)
    mature.unit_ranks['fc2'][7] = 2
    rejected('mature_receiver_rejected', lambda: HeadTransaction(mature, router, 7, 1))
    rejected('owned_target_rejected', lambda: HeadTransaction(model, router, 7, 1, 1))
    frozen = copy.deepcopy(model)
    frozen.freeze_masks['fc2'][7] = True
    rejected('frozen_receiver_rejected', lambda: HeadTransaction(frozen, router, 7, 1))
    frozen.fixed_task_allocation = True
    ok('fixed_allocation_unowned_reserve_permission', HeadTransaction(frozen, router, 7, 1).state == 'IDLE')
    frozen.task_freeze_layers = ['fc2']
    rejected('task_wide_freeze_rejected', lambda: HeadTransaction(frozen, router, 7, 1))
    changed = copy.deepcopy(candidate)
    with torch.no_grad():
        changed[0].fc1.weight[0, 0] += 1
    rejected('backbone_mutation_rejected', lambda: assert_protected((model, router), changed, 7, 1))
    # Verification checks are synthetic COUNTS, not feasibility evidence.
    variants = ('baseline', 'availability', 'mask_only', 'head_agg', 'single_1', 'single_2')
    receipts = {}
    for owner in (3, 54, 51):
        stats = dict(positive=50, negative=1000, target_hits=50, net_rescue=10,
            recall=1., FAR=0., negative_break=0, by_negative_class={'6': dict(rows=1000, FAR=0.)})
        receipts[owner] = dict(binding=dict(owner=owner, task=1, split='holdout', role_sha256='role'),
            labels_used_by_predictor=False, raw_examples_transmitted=0,
            variants={name: dict(function_hash=name, stats=copy.deepcopy(stats)) for name in variants})
        for name in ('baseline', 'availability', 'mask_only', 'single_1', 'single_2'):
            receipts[owner]['variants'][name]['stats'].update(target_hits=40, net_rescue=0, recall=.8)
    expected = {name: name for name in variants}
    validate_receipts(receipts, 3, [54, 51], 1, 'role', 'holdout', expected)
    ok('bound_owner_function_receipts', True)
    ok('current_empirical_acceptance', accept(receipts, 3, [54, 51])['status'] == 'EMPIRICAL_CURRENT_PASS')
    changed = copy.deepcopy(receipts)
    changed[51]['variants']['head_agg']['stats']['positive'] = 31
    ok('missing_positive_is_unknown', accept(changed, 3, [54, 51])['status'] == 'UNKNOWN_EVIDENCE')
    changed = copy.deepcopy(receipts)
    changed[51]['variants']['head_agg']['stats']['by_negative_class']['6']['FAR'] = .1
    ok('peer_per_class_risk_veto', accept(changed, 3, [54, 51])['status'] == 'FAIL')
    changed = copy.deepcopy(receipts)
    changed[3]['variants']['head_agg']['stats']['negative_break'] = 1
    ok('negative_break_veto', accept(changed, 3, [54, 51])['status'] == 'FAIL')
    changed = copy.deepcopy(receipts)
    changed[54]['variants']['head_agg']['function_hash'] = 'stale'
    rejected('stale_receipt_rejected', lambda: validate_receipts(changed, 3, [54, 51], 1, 'role', 'holdout', expected))
    changed = copy.deepcopy(receipts)
    changed.pop(51)
    rejected('missing_owner_rejected', lambda: validate_receipts(changed, 3, [54, 51], 1, 'role', 'holdout', expected))
    rejected('HOLDOUT_cannot_select_eta', lambda: select_eta(receipts, 'agg', 3, [54, 51]))
    result = dict(protocol=VERSION, passed=len(checks), failed=0, checks=checks,
        encoded_head_bytes=len(b), tested_model='actual DeNICEModel', no_training=True)
    path = Path('artifacts/appliance_direct_head_mechanics_20261010.json')
    path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
