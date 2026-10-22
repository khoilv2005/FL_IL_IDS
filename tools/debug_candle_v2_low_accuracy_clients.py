"""Explain low full-TEST client accuracy using saved results and offline validation only."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from appliance.runner import load_input
from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from fed_learning.strategies.decentralized.candle_v2 import logits, state
from fed_learning.strategies.decentralized.candle_v2_inference import features_and_context
from tools.debug_candle_v2_checkpoints import metrics, write
from tools.debug_candle_v1_v2_matched import model_hash, scoped, fit_memory


@torch.no_grad()
def evaluate(model, router, config, x, y, batch, device):
    before = model_hash(model)
    value = state(model); available = sorted(map(int, value['available']))
    mapping = {int(c): int(t) for c, t in value['class_tasks'].items()}
    scopes = {t: [c for c in available if mapping[c] == t] for t in range(6)}
    mc, fit = fit_memory(router)
    scores, binary_episodes, mc_episodes = [], [], []
    for start in range(0, len(x), batch):
        h, acts = features_and_context(model, torch.from_numpy(x[start:start+batch]).to(device))
        scores.append(logits(model, h, range(34), config['candle_v2_logit_scale']).cpu().numpy())
        bits = router.binarize_layer_activations({k: v.cpu().numpy() for k, v in acts.items()})
        binary_episodes.append(router.predict_episodes_batch(bits))
        mc_episodes.append(mc.predict_episodes_batch(bits))
    scores = np.concatenate(scores)
    routes = dict(binary=np.concatenate(binary_episodes), multiclass=np.concatenate(mc_episodes))
    predictions = {}; result = dict(rows=len(y), methods={}, fit_saved_memory=fit, scopes=scopes)
    for name, episodes in routes.items():
        pred, fallback = scoped(scores, episodes, scopes, available)
        predictions[name] = pred
        result['methods'][name] = dict(**metrics(y, pred), route_accuracy=float((episodes == y//6).mean()),
            fallback_rows=fallback)
    predictions['oracle_task'], _ = scoped(scores, y//6, scopes, available)
    result['methods']['oracle_task'] = metrics(y, predictions['oracle_task'])
    for name, pred in predictions.items():
        result['methods'][name]['per_task'] = []
        for t in range(6):
            ix = y//6 == t
            if ix.any():
                result['methods'][name]['per_task'].append(dict(task=t, **metrics(y[ix], pred[ix])))
    if before != model_hash(model):
        raise ValueError('Diagnostic changed model parameters/masks')
    result['model_hash_before_after'] = before
    return result, dict(y=y, scores=scores, **predictions, **{'route_'+k: v for k, v in routes.items()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=Path('C:/Users/khoak/Downloads/results (14).zip'))
    p.add_argument('--results', type=Path, default=ROOT/'audit_denice/candle_v2_results15_20261022_v2')
    p.add_argument('--matched', type=Path, default=ROOT/'audit_denice/candle_v1_v2_matched_20261022_v1')
    p.add_argument('--roles', type=Path, default=ROOT/'audit_denice/appliance_legacy_results11/inputs/roles')
    p.add_argument('--data', type=Path, default=ROOT/'audit_denice/appliance_transfer_results9/inputs/dataset')
    p.add_argument('--inputs', type=Path, default=ROOT/'audit_denice/candle_v2_checkpoint_debug_20261022_v1/inputs')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=.1)
    p.add_argument('--device', default='cuda')
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False); torch.set_num_threads(2)
    clients = json.loads((a.results/'per_client.json').read_text())
    selected = sorted((r for r in clients if r['accuracy'] < a.threshold), key=lambda r: r['accuracy'])
    ids = [r['client_id'] for r in selected]
    write(a.out/'protocol_before_data.json', dict(version='low_accuracy_clients_v1',
        script_sha256=file_sha256(__file__), threshold=a.threshold, selected_client_ids=ids,
        selection='Only already-published full TEST per-client accuracy; no threshold/classifier tuning',
        new_inference='Offline owner validation at class-birth and T5 checkpoints only',
        no_new_TEST_inference=True, no_optimizer=True, no_historical_training_replay=True,
        historical_validation='Offline diagnosis only; never passed to training, router fitting or live clients',
        router_refit='Frozen saved binary sketches only, same balanced LR as previous matched audit',
        oracle='True task used only for a separate diagnostic',
        limitation='Previous validation source, selected worst clients after TEST; not independent confirmation'))
    started = time.monotonic()
    with zipfile.ZipFile(a.archive) as z:
        def read(name):
            return z.read(next(n for n in z.namelist() if n.endswith('/'+name))).decode()
        train = [json.loads(line) for line in read('candle_v2_training.jsonl').splitlines()]
        health = [json.loads(line) for line in read('optimizer_health.jsonl').splitlines()]
        debug = json.loads(read('denice_debug_history.json'))
    roles = CleanRoleData(a.roles, source_data_dir=a.data)
    owner_data = {cid: roles.client_role(cid, 'validation') for cid in ids}
    records = {}
    for saved in selected:
        cid = saved['client_id']; samples = saved['samples']
        counts = np.asarray(saved['class_counts']); recall = np.asarray(saved['per_class_recall'])
        class_correct = np.rint(counts*recall).astype(np.int64)
        covered = np.isin(np.arange(34), saved['available_classes'])
        if int(counts[covered].sum()) != saved['covered_rows'] or class_correct[~covered].sum():
            raise ValueError('Full TEST coverage accounting mismatch')
        correct = int(class_correct.sum())
        if not np.isclose(correct/samples, saved['accuracy'], rtol=0, atol=1e-12):
            raise ValueError('Full TEST correct accounting mismatch')
        matched = json.loads((a.matched/f'v2_client_{cid}.json').read_text())
        r = dict(client_id=cid, full_TEST=dict(accuracy=saved['accuracy'], route_accuracy=saved['route_accuracy'],
            macro_f1=saved['macro_f1_34'], samples=samples, correct=correct,
            coverage=saved['covered_rows']/samples, unavailable_errors=samples-saved['covered_rows'],
            covered_errors=saved['covered_rows']-correct, available_classes=saved['available_classes'],
            per_class=[dict(class_id=c, rows=int(counts[c]), correct=int(class_correct[c]),
                recall=float(recall[c]), registered=bool(covered[c])) for c in range(34)]),
            prior_global_development=matched['panels'], training=[], checkpoint_diagnostics={},
            bootstrap=[event for row in debug if row.get('type') == 'task_start'
                       for event in row.get('bootstrap_events', []) if event.get('client_id') == cid])
        for t in range(6):
            tt = [v for v in train if v['client_id'] == cid and v['task'] == t]
            oo = [v for v in health if v['client_id'] == cid and v['task'] == t]
            if not tt:
                r['training'].append(dict(task=t, active=False)); continue
            tt.sort(key=lambda v: v['round'])
            rows = [v for v in debug if v.get('type') == 'round' and v.get('task') == t]
            # Some historical receipts use a different round event name.
            if not rows:
                rows = [v for v in debug if v.get('task') == t and 'cluster' in v and 'round' in v]
            cap = next((row['cluster'].get('capacity_after_aggregation', {}).get(str(cid))
                        for row in reversed(rows) if str(cid) in row['cluster'].get('capacity_after_aggregation', {})), None)
            local = sorted(set(c for v in tt for c in v['local_classes']))
            imports = sorted(set(c for v in tt for c in v['imported_classes']))
            r['training'].append(dict(task=t, active=True, rounds=len(tt), local_classes=local,
                imported_classes=imports, final_real_CE=tt[-1]['real_CE'], final_losses=tt[-1]['losses'],
                successful_optimizer_steps=sum(v['optimizer_steps'] for v in oo),
                skipped_optimizer_steps=sum(v['skipped_optimizer_steps'] for v in oo),
                zero_update_rounds=sum(v['zero_update'] for v in oo),
                import_round_counts={str(c): sum(c in v['imported_classes'] for v in tt) for c in imports},
                current_BASE_counts_before_internal_validation_split=next((row['client_data'][str(cid)]['class_hist'] for row in debug
                    if row.get('type') == 'task_start' and row['task'] == t and str(cid) in row['client_data']), {}),
                final_round_capacity=cap))
        records[cid] = r
    tasks = sorted(set(r['task'] for v in records.values() for r in v['training'] if r['active']))
    sources = {}
    for task in tasks:
        ckpt, sources[str(task)] = load_input(a.inputs/f'checkpoint_task_{task}_all_rounds.zip', task, 19)
        if ckpt['config']['denice_data_roles_sha256'] != file_sha256(a.roles/'role_manifest.json'):
            raise ValueError('Data-role identity differs')
        for cid, record in records.items():
            entry = next(v for v in record['training'] if v['task'] == task)
            if task != 5 and not entry['active']:
                continue
            model, router = _make_denice_client_model(ckpt, cid, a.device)
            model.clear_active_adapters(); model.eval()
            x, y, rows = owner_data[cid]
            ix = np.ones(len(y), dtype=bool) if task == 5 else y//6 == task
            if not ix.any():
                record['checkpoint_diagnostics'][str(task)] = dict(owner_validation_rows=0)
                continue
            result, arrays = evaluate(model, router, ckpt['config'], x[ix], y[ix], 512, a.device)
            result['offline_validation_role'] = 'same owner, all tasks at T5; current task only at birth checkpoint'
            result['source_rows'] = rows[ix].tolist()
            result['available'] = state(model)['available']
            result['owned_history'] = state(model)['owned_history']
            result['memory_task_ids'] = sorted(int(t) for t, mem in router.activation_memory.items() if len(mem))
            result['row_state'] = [dict(class_id=c, rank=int(model.unit_ranks['fc2'][c]),
                fan_in=int(torch.count_nonzero(model.weight_masks['fc2'][c])),
                effective_weight_norm=float(torch.linalg.vector_norm(model.fc2.weight[c].detach()*model.weight_masks['fc2'][c].to(model.fc2.weight))))
                for c in state(model)['available']]
            record['checkpoint_diagnostics'][str(task)] = result
            np.savez_compressed(a.out/f'T{task}_client_{cid}_owner_validation.npz', **arrays)
            write(a.out/f'client_{cid}.json', record)
            print(f'T{task} client={cid} owner_val={len(y[ix])} '
                  f'binary={result["methods"]["binary"]["accuracy"]:.2%} '
                  f'oracle={result["methods"]["oracle_task"]["accuracy"]:.2%}', flush=True)
            del model, router
        del ckpt; gc.collect()
    write(a.out/'completion.json', dict(completed=True, selected_client_ids=ids, sources=sources,
        clients=list(records.values()), seconds=time.monotonic()-started,
        no_optimizer=True, no_new_TEST_inference=True, source_archive=str(a.archive)))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
