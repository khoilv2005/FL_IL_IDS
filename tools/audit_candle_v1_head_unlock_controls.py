"""Mechanism controls on frozen packets; no fitting or threshold selection."""
import argparse
import copy
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from appliance.runner import load_input
from appliance.state import complete_hash
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from tools.audit_candle_v1_two_tier import read, write
from tools.audit_candle_v1_positive_quantile_gate import evaluation_panel
from tools.audit_candle_v1_head_unlock import cache, candidate_prediction, stats, gate_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    inputs = ROOT / 'audit_denice/appliance_legacy_results11/inputs'
    write(args.out / 'protocol_before_data.json', dict(
        quantiles=[.05, .01], no_fitting=True, no_threshold_selection=True,
        purpose='Post-result mechanism controls, same development panel; no independent confirmation',
        controls=['original_masked_row', 'receiver_row_donor_mask', 'receiver_row_full_unmask',
                  'direct_donor_row', 'frozen_packet', 'prototype_gate_only'],
        script_sha256=file_sha256(__file__)))
    from eval_checkpoint import _make_denice_client_model
    checkpoint, source = load_input(inputs / 'checkpoint_task_4_all_rounds.zip', 4, 19)
    roles = CleanRoleData(inputs / 'roles', source_data_dir=
                          ROOT / 'audit_denice/appliance_transfer_results9/inputs/dataset')
    debug = read(inputs / 'training_artifacts/denice_debug_history.json')
    seen = sorted(map(int, checkpoint['seen_classes']))
    donor, donor_router = _make_denice_client_model(checkpoint, 11, args.device)
    donor.eval()
    donor_hash = complete_hash(donor, donor_router)
    donor_mask = donor.weight_masks['fc2'][24].to(donor.fc2.weight)
    donor_bias_mask = donor.bias_masks['fc2'][24].to(donor.fc2.bias)
    direct = dict(w=(donor.fc2.weight[24] * donor_mask).detach().cpu().numpy(),
                  b=float((donor.fc2.bias[24] * donor_bias_mask).detach().cpu()))
    records = []
    for cid in [7, 65]:
        result = read(args.experiment / f'receiver_{cid}/result.json')
        packet = read(args.experiment / f'receiver_{cid}/primary_packet.json')
        model, router = _make_denice_client_model(checkpoint, cid, args.device)
        model.eval()
        before = complete_hash(model, router)
        learned = set()
        for event in debug:
            if event.get('type') == 'task_start' and event['task'] <= 4:
                learned.update(map(int, event['client_data'].get(str(cid), {}).get('labels', [])))
        panel = evaluation_panel(roles, cid, 11, learned, set())
        values = cache(model, router, panel['x'], seen, args.device)
        raw_w = model.fc2.weight[24].detach().cpu().numpy().copy()
        raw_b = float(model.fc2.bias[24].detach().cpu())
        heads = dict(
            original_masked_row=dict(w=values['w'][24], b=float(values['b'][24])),
            receiver_row_donor_mask=dict(w=raw_w * donor_mask.cpu().numpy(),
                                         b=raw_b * float(donor_bias_mask.cpu())),
            receiver_row_full_unmask=dict(w=raw_w, b=raw_b),
            direct_donor_row=direct,
            frozen_packet=dict(w=np.asarray(packet['head']['w'], np.float32), b=packet['head']['b']))
        prototypes = {key: np.asarray(val, np.float32) for key, val in packet['prototypes'].items()}
        current = sorted(router.episode_classes[4])
        row_norms = {str(c): dict(effective_norm=float(np.linalg.norm(values['w'][c])),
                                  effective_bias=float(values['b'][c])) for c in current}
        controls = []
        for q in [.05, .01]:
            locked = next(row for row in result['ablations']
                          if row['head'] == result['primary']['head']
                          and row['spec']['gate'] == result['primary']['spec']['gate']
                          and row['quantile'] == q)
            spec = copy.deepcopy(locked['spec'])
            for name, head in heads.items():
                pred = candidate_prediction(values, head, prototypes, spec)
                controls.append(dict(quantile=q, control=name,
                                     metrics=stats(pred, panel['y'], values['pred'])))
            # Explicit alternative classifier control: the prototype gate alone,
            # omitting FC2 competition. Not an authorized inference replacement.
            gate = gate_values(values, heads['frozen_packet'], prototypes)[spec['gate']]
            use = gate >= spec['tau']
            pred = values['pred'].copy()
            pred[use] = 24
            frozen_pred = candidate_prediction(values, heads['frozen_packet'], prototypes, spec)
            controls.append(dict(quantile=q, control='prototype_gate_only',
                                 predictions_equal_frozen_packet=bool(np.array_equal(pred, frozen_pred)),
                                 metrics=stats(pred, panel['y'], values['pred'])))
        if before != complete_hash(model, router):
            raise ValueError('Receiver reference changed')
        record = dict(receiver=cid, controls=controls, current_episode_row_norms=row_norms,
                      head_norms={key: dict(norm=float(np.linalg.norm(head['w'])), bias=float(head['b']))
                                  for key, head in heads.items()},
                      source_reference_unchanged=True)
        records.append(record)
        write(args.out / f'receiver_{cid}.json', record)
        for item in controls:
            metric = item['metrics']
            print(cid, item['quantile'], item['control'], 'recall', metric['recall'],
                  'FP', metric['false_activation'], 'break', metric['breaks'], flush=True)
    if donor_hash != complete_hash(donor, donor_router):
        raise ValueError('Donor reference changed')
    write(args.out / 'completion.json', dict(completed=True, seconds=time.monotonic()-started,
          source_hashes=source, records=records, final_TEST_read=False, no_fitting=True,
          same_development_panel_as_previous_experiment=True))


if __name__ == '__main__':
    main()
