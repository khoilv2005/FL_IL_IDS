"""Paired frozen-checkpoint development audit; no TEST access or backbone updates."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import sklearn
import torch
from threadpoolctl import threadpool_limits
from sklearn.exceptions import InconsistentVersionWarning
from appliance.runner import load_input
from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import file_sha256
from fed_learning.training.checkpoint_state import restore_context_detector
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
from fed_learning.strategies.decentralized.candle_v2 import logits, state
from fed_learning.strategies.decentralized.candle_v2_inference import features_and_context
from tools.debug_candle_v2_checkpoints import argmax, write


def model_hash(model):
    h = hashlib.sha256()
    for name, values in [('parameters', model.state_dict()), ('weight_masks', model.weight_masks),
                         ('bias_masks', model.bias_masks), ('ranks', model.unit_ranks)]:
        for key, value in sorted(values.items()):
            a = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            h.update((name+key+str(a.shape)+str(a.dtype)).encode())
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def scoped(scores, episodes, scopes, fallback):
    pred = np.full(len(scores), -1, dtype=np.int64)
    misses = 0
    for ep in np.unique(episodes):
        ix = episodes == ep
        allowed = scopes.get(int(ep), [])
        if not allowed:
            misses += int(ix.sum())
            allowed = fallback
        pred[ix] = argmax(scores[ix], allowed)
    return pred, misses


def summary(y, pred, episodes=None, covered=None, oracle=None):
    cm = np.zeros((34, 34), np.int64)
    valid = pred >= 0
    np.add.at(cm, (y[valid], pred[valid]), 1)
    counts = np.bincount(y, minlength=34)
    denom = counts + cm.sum(0)
    f1 = np.divide(2*np.diag(cm), denom, out=np.zeros(34), where=denom > 0)
    result = dict(rows=len(y), correct=int((pred == y).sum()), accuracy=float((pred == y).mean()),
                  macro_f1=float(f1.mean()), confusion=cm.tolist(),
                  per_task_accuracy=[float((pred[y//6 == t] == y[y//6 == t]).mean()) for t in range(6)])
    if episodes is not None:
        route = np.bincount((y//6)*6+episodes, minlength=36).reshape(6, 6)
        result.update(route_accuracy=float((episodes == y//6).mean()), route_confusion=route.tolist())
        if covered is not None and oracle is not None:
            error = pred != y
            parts = dict(unregistered=int((error & ~covered).sum()),
                         within_task=int((error & covered & (oracle != y)).sum()),
                         routing=int((error & covered & (oracle == y) & (episodes != y//6)).sum()),
                         unexplained=int((error & covered & (oracle == y) & (episodes == y//6)).sum()))
            if sum(parts.values()) != int(error.sum()):
                raise ValueError('Non-exhaustive error partition')
            result['error_partition_order_coverage_then_within_task_then_route'] = parts
            result['coverage'] = float(covered.mean())
            result['correct_route_but_unregistered'] = int(((episodes == y//6) & ~covered).sum())
    return result


def fit_memory(router):
    refit = copy.deepcopy(router)
    refit.router_mode = 'multiclass_balanced'
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        refit.train_models(max(refit.activation_memory))
    xx, yy = [], []
    for t, values in sorted(refit.activation_memory.items()):
        if len(values):
            xx.append(np.asarray(values)); yy.extend([int(t)]*len(values))
    xx, yy = np.concatenate(xx), np.asarray(yy)
    pred = refit.predict_episodes_batch(xx)
    return refit, dict(memory_rows=len(yy), feature_dim=xx.shape[1],
        per_episode_counts=np.bincount(yy, minlength=6).tolist(),
        memory_resubstitution_accuracy=float((pred == yy).mean()),
        memory_resubstitution_balanced_accuracy=float(np.mean([(pred[yy == t] == t).mean() for t in np.unique(yy)])),
        warnings=[str(w.message) for w in caught])


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--clients', nargs='+', type=int)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    reference = ROOT/'audit_denice/candle_v2_checkpoint_debug_20261022_v1/diagnostics/development_panel.npz'
    tail = ROOT/'audit_denice/candle_v2_routing_repair_20261022_v1/repair_verification/tail/development_panel.npz'
    v1root = ROOT/'audit_denice/appliance_legacy_results11/inputs'
    v2root = ROOT/'audit_denice/candle_v2_checkpoint_debug_20261022_v1/inputs'
    frozen_path = v1root/'legacy_self_task_5/frozen_routers.joblib'
    write(a.out/'protocol_before_data.json', dict(version='v1_v2_matched_router_development_v1',
        script_sha256=file_sha256(__file__), panel_sha256=[file_sha256(reference), file_sha256(tail)],
        frozen_v1_routers_sha256=file_sha256(frozen_path),
        methods=['V1 binary', 'V1 frozen multiclass', 'V1 refit multiclass', 'V2 binary', 'V2 refit multiclass',
                 'true-task oracle with native availability', 'true-task oracle with full task class mask',
                 'true-task oracle with common availability', 'V1 true-task oracle without adapters'],
        refit='Only saved binary activation memory; balanced LR C=1 random_state=0 max_iter=1000',
        evaluation='Both locked, previously exposed validation panels, all matched 98 client models',
        no_final_TEST=True, no_backbone_training=True, no_threshold_search=True,
        oracle='Diagnostic only, true task is never given to deployed inference',
        historical_router_sklearn='1.6.1', current_sklearn=sklearn.__version__,
        version_control='Refit BOTH checkpoints under same current sklearn; also retain historical frozen V1 router',
        error_partition='Among errors: unavailable true class, then wrong oracle classification, then wrong route',
        clients=a.clients or 'all matched clients'))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', InconsistentVersionWarning)
        frozen = joblib.load(frozen_path)
    write(a.out/'router_load_warnings.json', sorted(set(str(w.message) for w in caught)))
    panels = [np.load(reference), np.load(tail)]
    x = np.concatenate([z['x'] for z in panels]); y = np.concatenate([z['y'] for z in panels])
    slices = dict(reference=slice(0, len(panels[0]['y'])), tail=slice(len(panels[0]['y']), len(y)), combined=slice(None))
    np.savez_compressed(a.out/'labels.npz', y=y)
    records = []; sources = {}; started = time.monotonic(); ids = None
    for version, path in [('v1', v1root/'checkpoint_task_5_all_rounds.zip'),
                          ('v2', v2root/'checkpoint_task_5_all_rounds.zip')]:
        print('Loading '+version, flush=True)
        ckpt, sources[version] = load_input(path, 5, 19)
        current_ids = sorted(map(int, ckpt['client_model_states']))
        if a.clients:
            current_ids = [cid for cid in current_ids if cid in a.clients]
        if ids is not None and ids != current_ids:
            raise ValueError('Client IDs do not match')
        ids = current_ids
        if version == 'v1':
            role_hash = ckpt['config']['denice_data_roles_sha256']
        elif role_hash != ckpt['config']['denice_data_roles_sha256']:
            raise ValueError('Different data role assignment')
        for cid in ids:
            model, binary = _make_denice_client_model(ckpt, cid, a.device)
            model.clear_active_adapters(); model.eval()
            before = model_hash(model)
            if version == 'v1':
                restore_context_detector(binary, frozen[cid]['BinarySelf'])
                mc_frozen = copy.deepcopy(binary)
                restore_context_detector(mc_frozen, frozen[cid]['MulticlassSelf'])
                scopes = {int(t): sorted(map(int, classes)) for t, classes in binary.episode_classes.items()}
                fallback = list(range(34))
            else:
                value = state(model)
                fallback = sorted(map(int, value['available']))
                scopes = {t: [c for c in fallback if int(value['class_tasks'][str(c)] if str(c) in value['class_tasks']
                                                             else value['class_tasks'][c]) == t] for t in range(6)}
            router_hash = hashlib.sha256(pickle.dumps(binary)).hexdigest()
            refit, fit_receipt = fit_memory(binary)
            joblib.dump(refit, a.out/f'{version}_client_{cid}_refit_router.joblib', compress=3)
            routers = dict(binary=binary, refit_multiclass=refit)
            if version == 'v1':
                routers['frozen_multiclass'] = mc_frozen
            routes = {key: [] for key in routers}
            banks = []; noadapter = []
            for start in range(0, len(x), a.batch_size):
                inputs = torch.from_numpy(x[start:start+a.batch_size]).to(a.device)
                if version == 'v2':
                    h, acts = features_and_context(model, inputs)
                    banks.append(logits(model, h, range(34), ckpt['config']['candle_v2_logit_scale']).cpu().numpy())
                else:
                    acts = model.get_context_activations_per_sample(inputs)
                    contexts = []
                    for t in range(6):
                        model.set_active_context(t)
                        contexts.append(model(inputs).cpu().numpy())
                    model.clear_active_adapters()
                    banks.append(np.stack(contexts, axis=1))
                    noadapter.append(model(inputs).cpu().numpy())
                bits = binary.binarize_layer_activations({key: val.detach().cpu().numpy() for key, val in acts.items()})
                for key, router in routers.items():
                    routes[key].append(router.predict_episodes_batch(bits))
            scores = np.concatenate(banks)
            routes = {key: np.concatenate(values).astype(np.int64) for key, values in routes.items()}
            predictions = {}
            true_scores = scores[np.arange(len(y)), y//6] if version == 'v1' else scores
            oracle, _ = scoped(true_scores, y//6, scopes, fallback)
            global_scopes = {t: list(range(t*6, min(t*6+6, 34))) for t in range(6)}
            oracle_global, _ = scoped(true_scores, y//6, global_scopes, [])
            predictions.update(oracle_available=oracle, oracle_full_task=oracle_global)
            if version == 'v1':
                predictions['oracle_no_adapter'], _ = scoped(np.concatenate(noadapter), y//6, scopes, fallback)
            fallbacks = {}
            for key, episodes in routes.items():
                selected_scores = scores[np.arange(len(y)), episodes] if version == 'v1' else scores
                predictions[key], fallbacks[key] = scoped(selected_scores, episodes, scopes, fallback)
                if version == 'v1':
                    canonical, canonical_route = _denice_routed_logits_with_episodes(model,
                        torch.from_numpy(x[:32]).to(a.device), routers[key], list(range(34)), a.device,
                        inference_policy='pred_hard')
                    if not np.array_equal(predictions[key][:32], canonical.argmax(1).cpu().numpy()) or not np.array_equal(episodes[:32], canonical_route):
                        raise ValueError('Score bank differs from canonical V1 inference')
            covered = np.asarray([int(c) in scopes.get(int(c//6), []) for c in y])
            record = dict(version=version, client_id=cid, scopes=scopes, fallback_rows=fallbacks,
                          fit=fit_receipt, panels={})
            for panel_name, ix in slices.items():
                record['panels'][panel_name] = {key: summary(y[ix], pred[ix],
                    routes[key][ix] if key in routes else None, covered[ix], oracle[ix])
                    for key, pred in predictions.items()}
            if before != model_hash(model) or router_hash != hashlib.sha256(pickle.dumps(binary)).hexdigest():
                raise ValueError('Inference/refit changed original model or router')
            record['model_hash_before_after'] = before
            write(a.out/f'{version}_client_{cid}.json', record)
            np.savez_compressed(a.out/f'{version}_client_{cid}_predictions.npz',
                **predictions, **{'route_'+key: val for key, val in routes.items()},
                oracle_scores=true_scores, covered=covered)
            records.append(record)
            print(f'{version} client={cid} binary={record["panels"]["tail"]["binary"]["accuracy"]:.2%} '
                  f'mc={record["panels"]["tail"]["refit_multiclass"]["accuracy"]:.2%} '
                  f'oracle={record["panels"]["tail"]["oracle_available"]["accuracy"]:.2%}', flush=True)
            del model, binary, refit
        del ckpt; gc.collect()
    # Same class competition for the two different checkpoints; diagnostic only.
    common_records = []
    for cid in ids:
        rr = {v: next(r for r in records if r['version'] == v and r['client_id'] == cid) for v in ('v1', 'v2')}
        common = {t: sorted(set(rr['v1']['scopes'].get(t, [])) & set(rr['v2']['scopes'].get(t, []))) for t in range(6)}
        for v in ('v1', 'v2'):
            z = np.load(a.out/f'{v}_client_{cid}_predictions.npz')
            pred, _ = scoped(z['oracle_scores'], y//6, common, [])
            common_records.append(dict(version=v, client_id=cid, panels={k: summary(y[ix], pred[ix]) for k, ix in slices.items()}))
    aggregated = {}
    for v in ('v1', 'v2'):
        selected = [r for r in records if r['version'] == v]
        aggregated[v] = {}
        for panel_name in slices:
            aggregated[v][panel_name] = {}
            for key in selected[0]['panels'][panel_name]:
                cells = [r['panels'][panel_name][key] for r in selected]
                combined = {k: float(np.mean([c[k] for c in cells])) for k in ('accuracy', 'macro_f1')}
                combined.update(correct=sum(c['correct'] for c in cells), prediction_rows=sum(c['rows'] for c in cells),
                                confusion=np.sum([c['confusion'] for c in cells], axis=0).tolist(),
                                per_task_accuracy=np.mean([c['per_task_accuracy'] for c in cells], axis=0).tolist())
                if 'route_accuracy' in cells[0]:
                    combined['route_accuracy'] = float(np.mean([c['route_accuracy'] for c in cells]))
                    combined['route_confusion'] = np.sum([c['route_confusion'] for c in cells], axis=0).tolist()
                    combined['error_partition'] = {k: sum(c['error_partition_order_coverage_then_within_task_then_route'][k] for c in cells)
                                                   for k in cells[0]['error_partition_order_coverage_then_within_task_then_route']}
                    combined['coverage'] = float(np.mean([c['coverage'] for c in cells]))
                    combined['correct_route_but_unregistered'] = sum(c['correct_route_but_unregistered'] for c in cells)
                aggregated[v][panel_name][key] = combined
            matched = [r['panels'][panel_name] for r in common_records if r['version'] == v]
            aggregated[v][panel_name]['oracle_common_availability'] = dict(accuracy=float(np.mean([r['accuracy'] for r in matched])),
                per_task_accuracy=np.mean([r['per_task_accuracy'] for r in matched], axis=0).tolist())
    result = dict(completed=True, sources=sources, clients=ids, summaries=aggregated,
                  sklearn=sklearn.__version__, seconds=time.monotonic()-started,
                  no_final_TEST=True, no_backbone_training=True,
                  metric='Mean client metrics; each of 98 models evaluated on entire same development panel')
    write(a.out/'completion.json', result)
    print(json.dumps({v: {k: r['accuracy'] for k, r in aggregated[v]['tail'].items()} for v in aggregated}), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
