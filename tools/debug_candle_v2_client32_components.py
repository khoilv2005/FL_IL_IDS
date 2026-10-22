"""Frozen R4/R19 head, encoder and BN counterfactuals for the observed T3 collapse."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from fed_learning.training.denice_delta_checkpoint import apply_client_model_deltas
from fed_learning.strategies.decentralized.candle_v2 import logits
from fed_learning.strategies.decentralized.candle_v2_inference import features_and_context
from tools.debug_candle_v2_checkpoints import metrics, write
from tools.debug_candle_v1_v2_matched import model_hash


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(2)
    write(a.out/'protocol_before_data.json', dict(client_id=32, task=3, reference_round=4, final_round=19,
        script_sha256=file_sha256(__file__), no_optimizer=True, no_new_TEST=True,
        evaluation='Owner validation T3, oracle task to isolate classification; not deployed inference',
        selection='R4 selected after observing the existing trajectory, for mechanism diagnosis only',
        controls=['h4 W4', 'h19 W4', 'h4 W19', 'h19 W19', 'h19 with R4 BN buffers W19',
                  'h19 with R4 BN affine and buffers W19'],
        precision='R4 reconstructed FP16 archive; R19 exact FP32 terminal. Not a clean training ablation.',
        causal_limit='Mixing checkpoint components identifies functional dependencies, not local-training versus aggregation causality'))
    path = ROOT/'audit_denice/candle_v2_checkpoint_debug_20261022_v1/inputs/checkpoint_task_3_all_rounds.zip'
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read('checkpoint_archive_manifest.json'))
        def read(name):
            blob = z.read(name)
            if hashlib.sha256(blob).hexdigest() != manifest['checksums'][name]:
                raise ValueError('Member checksum mismatch')
            return torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
        base = read('checkpoint_task_3_base.pt')
        states = {32: base['client_model_states'][32]}; del base
        for r in range(5):
            early = read(f'checkpoint_task_3_round_{r}.pt')
            states = apply_client_model_deltas(states, {32: early['client_model_deltas'][32]})
        early['client_model_states'] = states; del early['client_model_deltas']
        model4, _ = _make_denice_client_model(early, 32, a.device)
        final = read(manifest['full_terminal_checkpoint'])
        model19, _ = _make_denice_client_model(final, 32, a.device)
        del early, final, states
    for model in (model4, model19):
        model.clear_active_adapters(); model.eval()
    before = [model_hash(m) for m in (model4, model19)]
    active4 = np.asarray(model4.unit_ranks['fc1']) > 0
    active19 = np.asarray(model19.unit_ranks['fc1']) > 0
    if not np.array_equal(active4, active19):
        raise ValueError('R4 and R19 FC1 layouts differ; cannot interpret a head swap')
    roles = CleanRoleData(ROOT/'audit_denice/appliance_legacy_results11/inputs/roles',
                          source_data_dir=ROOT/'audit_denice/appliance_transfer_results9/inputs/dataset')
    x, y, _ = roles.client_role(32, 'validation'); ix = y//6 == 3; x, y = x[ix], y[ix]
    def feature_array(model):
        return torch.cat([features_and_context(model, torch.from_numpy(x[i:i+512]).to(a.device))[0]
                          for i in range(0, len(x), 512)])
    def evaluate(h, head):
        scores = logits(head, h, list(range(18, 24)), 10.).cpu().numpy()
        pred = scores.argmax(1)
        return dict(**metrics(y, pred), class21_recall=float((pred[y == 21] == 21).mean()),
            class21_to23=int((pred[y == 21] == 23).sum()),
            class21_margin_vs23_mean=float((scores[y == 21, 21]-scores[y == 21, 23]).mean())), pred
    h4, h19 = feature_array(model4), feature_array(model19)
    results = {}; predictions = {}
    for name, h, head in [('h4_W4', h4, model4), ('h19_W4', h19, model4),
                          ('h4_W19', h4, model19), ('h19_W19', h19, model19)]:
        results[name], predictions[name] = evaluate(h, head)
    for affine in (False, True):
        shadow = copy.deepcopy(model19)
        for bn_name in ('bn1', 'bn2', 'bn3'):
            src, dst = getattr(model4, bn_name), getattr(shadow, bn_name)
            dst.running_mean.copy_(src.running_mean); dst.running_var.copy_(src.running_var)
            if affine:
                dst.weight.copy_(src.weight); dst.bias.copy_(src.bias)
        # This is an explicit diagnostic that changes even saved normalization state.
        # Mature R4 statistics accompany the R4 buffers, so the restore hook is coherent.
        shadow._bn_frozen_units = copy.deepcopy(model4._bn_frozen_units)
        shadow._bn_running_mean_frozen = copy.deepcopy(model4._bn_running_mean_frozen)
        shadow._bn_running_var_frozen = copy.deepcopy(model4._bn_running_var_frozen)
        name = 'h19_R4_BN_affine_and_buffers_W19' if affine else 'h19_R4_BN_buffers_W19'
        results[name], predictions[name] = evaluate(feature_array(shadow), model19)
        del shadow
    cos_h = torch.nn.functional.cosine_similarity(h4, h19).cpu().numpy()
    w4 = (model4.fc2.weight * model4.weight_masks['fc2'].to(model4.fc2.weight)).cpu().numpy()
    w19 = (model19.fc2.weight * model19.weight_masks['fc2'].to(model19.fc2.weight)).cpu().numpy()
    parameter_drift = {}
    for name, value in model4.state_dict().items():
        other = model19.state_dict()[name]
        if value.is_floating_point():
            parameter_drift[name] = dict(max_abs=float((value.float()-other.float()).abs().max()),
                                         l2=float(torch.linalg.vector_norm(value.float()-other.float())))
    if before != [model_hash(m) for m in (model4, model19)]:
        raise ValueError('Counterfactual modified source models')
    np.savez_compressed(a.out/'predictions.npz', y=y, **predictions)
    result = dict(completed=True, controls=results, source_hash_before_after=before,
        active_layout_matches=True, row_mask_equal=bool(torch.equal(model4.weight_masks['fc2'], model19.weight_masks['fc2'])),
        feature_cosine_mean=float(cos_h.mean()), feature_cosine_class21_mean=float(cos_h[y == 21].mean()),
        head_row_cosine={str(c): float(w4[c]@w19[c]/max(1e-8,np.linalg.norm(w4[c])*np.linalg.norm(w19[c]))) for c in range(18,24)},
        parameter_drift=parameter_drift, no_optimizer=True, no_new_TEST=True)
    write(a.out/'completion.json', result)
    print(json.dumps({name: dict(accuracy=r['accuracy'], class21_recall=r['class21_recall']) for name,r in results.items()}), flush=True)


if __name__ == '__main__':
    main()
