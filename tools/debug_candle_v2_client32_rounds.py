"""Trace client 32/T3 collapse through saved post-aggregation rounds; validation only."""
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.denice_delta_checkpoint import apply_client_model_deltas
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from fed_learning.strategies.decentralized.candle_v2 import logits, state
from fed_learning.strategies.decentralized.candle_v2_inference import features_and_context
from tools.debug_candle_v2_checkpoints import metrics, write
from tools.debug_candle_v1_v2_matched import scoped, model_hash


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--low-diagnostics', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(2)
    archive = ROOT/'audit_denice/candle_v2_checkpoint_debug_20261022_v1/inputs/checkpoint_task_3_all_rounds.zip'
    write(a.out/'protocol_before_data.json', dict(client_id=32, task=3, rounds=list(range(20)),
        script_sha256=file_sha256(__file__), no_optimizer=True, no_new_TEST=True,
        role='Owner validation Task 3 only, offline diagnostic',
        reconstruction='FP16 base and delta chain, all member checksums verified; compare R19 predictions to exact FP32 terminal',
        availability='Saved CANDLE available classes and class_tasks',
        oracle='True task only for a separate diagnostic',
        causal_limit='Round checkpoints are post-aggregation, no saved pre-aggregation local model; cannot isolate aggregation from local updates'))
    roles = CleanRoleData(ROOT/'audit_denice/appliance_legacy_results11/inputs/roles',
                          source_data_dir=ROOT/'audit_denice/appliance_transfer_results9/inputs/dataset')
    x, y, _ = roles.client_role(32, 'validation'); ix = y//6 == 3; x, y = x[ix], y[ix]
    with zipfile.ZipFile('C:/Users/khoak/Downloads/results (14).zip') as outer:
        training = [json.loads(v) for v in outer.read(next(n for n in outer.namelist() if n.endswith('/candle_v2_training.jsonl'))).decode().splitlines()]
        training = {v['round']: v for v in training if v['client_id'] == 32 and v['task'] == 3}
    started = time.monotonic(); rows = []
    with zipfile.ZipFile(archive) as z:
        manifest = json.loads(z.read('checkpoint_archive_manifest.json'))
        def read(name):
            blob = z.read(name)
            if hashlib.sha256(blob).hexdigest() != manifest['checksums'][name]:
                raise ValueError('Round member checksum mismatch')
            return torch.load(io.BytesIO(blob), map_location='cpu', weights_only=False)
        base = read('checkpoint_task_3_base.pt')
        states = {32: base['client_model_states'][32]}; del base
        for round_id in range(20):
            ckpt = read(f'checkpoint_task_3_round_{round_id}.pt')
            states = apply_client_model_deltas(states, {32: ckpt['client_model_deltas'][32]})
            ckpt['client_model_states'] = states
            del ckpt['client_model_deltas']
            model, router = _make_denice_client_model(ckpt, 32, a.device)
            before = model_hash(model); model.clear_active_adapters()
            available = sorted(map(int, state(model)['available']))
            mapping = {int(c): int(t) for c, t in state(model)['class_tasks'].items()}
            scopes = {t: [c for c in available if mapping[c] == t] for t in range(6)}
            ss, ep = [], []
            for start in range(0, len(x), 512):
                h, acts = features_and_context(model, torch.from_numpy(x[start:start+512]).to(a.device))
                ss.append(logits(model, h, range(34), ckpt['config']['candle_v2_logit_scale']).cpu().numpy())
                bits = router.binarize_layer_activations({k: v.cpu().numpy() for k, v in acts.items()})
                ep.append(router.predict_episodes_batch(bits))
            scores = np.concatenate(ss); episodes = np.concatenate(ep)
            pred, _ = scoped(scores, episodes, scopes, available)
            oracle, _ = scoped(scores, y//6, scopes, available)
            alpha = ckpt['cluster']['alpha_debug'].get(32, ckpt['cluster']['alpha_debug'].get('32'))
            self_alpha = sum(w for peer, w in zip(alpha['group_ids'], alpha['alphas']) if int(peer) == 32)
            row = dict(round=round_id, training_real_CE=training[round_id]['real_CE'],
                training_preservation=training[round_id]['losses']['preservation'],
                optimizer_steps=training[round_id]['optimizer_steps'],
                binary=metrics(y, pred), oracle=metrics(y, oracle), route_accuracy=float((episodes == 3).mean()),
                self_alpha=float(self_alpha), positive_peer_count=sum(w > 0 for peer, w in zip(alpha['group_ids'], alpha['alphas']) if int(peer) != 32),
                row21_rank=int(model.unit_ranks['fc2'][21]), row23_rank=int(model.unit_ranks['fc2'][23]),
                class21_margin_vs23_mean=float((scores[y == 21, 21]-scores[y == 21, 23]).mean()),
                class21_recall=float((oracle[y == 21] == 21).mean()),
                class21_to23=int((oracle[y == 21] == 23).sum()), model_hash_before_after=before)
            if before != model_hash(model):
                raise ValueError('Round inference mutated model')
            rows.append(row); write(a.out/f'round_{round_id}.json', row)
            np.savez_compressed(a.out/f'round_{round_id}_predictions.npz', y=y, binary=pred, oracle=oracle,
                episodes=episodes, class21_margin_vs23=scores[:, 21]-scores[:, 23])
            print(f'R{round_id} owner_oracle={row["oracle"]["accuracy"]:.2%} class21={row["class21_recall"]:.2%} '
                  f'margin={row["class21_margin_vs23_mean"]:.3f} rank21={row["row21_rank"]} '
                  f'train_CE={row["training_real_CE"]:.3f}', flush=True)
            del model, router, ckpt; gc.collect()
    exact = np.load(a.low_diagnostics/'T3_client_32_owner_validation.npz')
    if not np.array_equal(exact['y'], y):
        raise ValueError('Different owner validation order')
    comparison = dict(oracle_prediction_disagreements=int((oracle != exact['oracle_task']).sum()),
        binary_prediction_disagreements=int((pred != exact['binary']).sum()), rows=len(y),
        oracle_accuracy_exact=float((exact['oracle_task'] == y).mean()), oracle_accuracy_compact=float((oracle == y).mean()))
    write(a.out/'completion.json', dict(completed=True, rows=rows, fp16_R19_vs_exact_FP32=comparison,
        seconds=time.monotonic()-started, no_optimizer=True, no_new_TEST=True))


if __name__ == '__main__':
    main()
