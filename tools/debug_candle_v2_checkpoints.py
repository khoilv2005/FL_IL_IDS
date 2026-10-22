"""Frozen-checkpoint development diagnostics. No fitting, replay training, or TEST access."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from appliance.runner import load_input
from eval_checkpoint import _make_denice_client_model
from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
from fed_learning.strategies.decentralized.candle_v2 import features, logits, state


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def content_hash(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    return hashlib.sha256(str(x.shape).encode()+x.tobytes()).hexdigest()


def fingerprint(model):
    sha = hashlib.sha256()
    groups = {'state': model.state_dict(), 'weight': model.weight_masks,
              'bias': model.bias_masks, 'ranks': model.unit_ranks}
    for group, values in groups.items():
        for name, value in sorted(values.items()):
            array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            sha.update((group+name+str(array.shape)+str(array.dtype)).encode())
            sha.update(np.ascontiguousarray(array).tobytes())
    sha.update(json.dumps(state(model), sort_keys=True).encode())
    return sha.hexdigest()


def panel(roles, out, cap):
    # Stable owner order by available validation support, then unique content.
    # These are previously exposed development sources, not confirmation data.
    chunks, labels, receipt, used, loaded = [], [], [], set(), {}
    for c in range(34):
        owners = sorted(map(int, roles.manifest['clients']), key=lambda cid: (
            -roles.manifest['clients'][str(cid)]['role_class_counts']['validation'].get(str(c), 0), cid))
        count = 0
        for owner in owners:
            entry = roles.manifest['clients'][str(owner)]
            if not entry['role_class_counts']['validation'].get(str(c), 0):
                continue
            if owner not in loaded:
                loaded[owner] = roles.client_role(owner, 'validation')
            x, y, original = loaded[owner]
            for ix in np.flatnonzero(y == c):
                key = content_hash(x[ix])
                if key in used:
                    continue
                used.add(key)
                chunks.append(x[ix]); labels.append(c)
                receipt.append(dict(class_id=c, owner=owner, source_row=int(original[ix]), sha256=key))
                count += 1
                if count == cap:
                    break
            if count == cap:
                break
        if not count:
            raise ValueError(f'No validation examples for class {c}')
        print(f'Panel class={c} unique_rows={count}', flush=True)
    x, y = np.stack(chunks), np.asarray(labels, np.int64)
    np.savez_compressed(out/'development_panel.npz', x=x, y=y)
    write(out/'panel_receipt.json', dict(role='validation', final_TEST_read=False,
        historical_development_read_by_offline_evaluator_only=True, rows=receipt,
        cap_per_class=cap, exposure='Previously exposed development source; no untouched confirmation claim'))
    return x, y


def argmax(scores, allowed):
    allowed = np.asarray(sorted(set(map(int, allowed))), np.int64)
    return allowed[scores[:, allowed].argmax(1)] if len(allowed) else np.full(len(scores), -1, np.int64)


def metrics(y, pred):
    correct = pred == y
    cm = np.zeros((34, 34), np.int64)
    valid = pred >= 0
    np.add.at(cm, (y[valid], pred[valid]), 1)
    support = np.bincount(y, minlength=34)
    f1 = np.divide(2*np.diag(cm), support+cm.sum(0), out=np.zeros(34), where=support+cm.sum(0)>0)
    return dict(rows=len(y), correct=int(correct.sum()), accuracy=float(correct.mean()),
        macro_f1_present=float(f1[support>0].mean()),
        per_class_correct=np.bincount(y, weights=correct, minlength=34).astype(int).tolist(),
        class_counts=support.tolist())


def distribution(values):
    values = np.asarray(values)
    if not len(values):
        return None
    return dict(rows=len(values), mean=float(values.mean()),
        quantiles={str(q): float(np.quantile(values, q)) for q in (0,.05,.25,.5,.75,.95,1)},
        positive_fraction=float((values>0).mean()))


@torch.no_grad()
def infer(model, x, scale, batch):
    hh, ss, rr = [], [], []
    for start in range(0, len(x), batch):
        h = features(model, torch.from_numpy(x[start:start+batch]).to(next(model.parameters()).device))
        s = logits(model, h, range(34), scale)
        raw = model._apply_masked_linear(h, model.fc2, 'fc2')
        if not torch.isfinite(h).all() or not torch.isfinite(s).all():
            raise FloatingPointError('Nonfinite features or cosine scores')
        hh.append(h.cpu().numpy()); ss.append(s.cpu().numpy()); rr.append(raw.cpu().numpy())
    return np.concatenate(hh), np.concatenate(ss), np.concatenate(rr)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--roles', type=Path, default=ROOT/'audit_denice/appliance_legacy_results11/inputs/roles')
    p.add_argument('--data', type=Path, default=ROOT/'audit_denice/appliance_transfer_results9/inputs/dataset')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tasks', type=int, nargs='+', default=list(range(6)))
    p.add_argument('--clients', type=int, nargs='+')
    p.add_argument('--cap', type=int, default=64)
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=512)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    write(a.out/'protocol_before_data.json', dict(version='candle_v2_checkpoint_debug_v1',
        tasks=a.tasks, clients=a.clients or 'all checkpoint clients', cap_per_class=a.cap,
        script_sha256=file_sha256(__file__), no_training=True, no_fitting=True, no_final_TEST=True,
        oracle='True-task mask and old-only argmax are diagnostics, never deployable inference',
        raw_affine='Untrained evaluation ablation, not a fitted replacement',
        development='Global unique validation panel, reused on every model and checkpoint; balanced panel accuracy is not full TEST accuracy'))
    roles = CleanRoleData(a.roles, source_data_dir=a.data)
    role_sha = file_sha256(roles.root/'role_manifest.json')
    x, y = panel(roles, a.out, a.cap)
    records, birth, base = [], {}, {}
    baseline_models = {}
    for task in a.tasks:
        print(f'Loading terminal checkpoint T{task}', flush=True)
        ckpt, source = load_input(a.inputs/f'checkpoint_task_{task}_all_rounds.zip', task, 19)
        config = ckpt['config']
        if not config.get('candle_v2_enabled') or config['denice_data_roles_sha256'] != role_sha:
            raise ValueError('Checkpoint method or data-role identity differs')
        use = y < min((task+1)*6, 34)
        xx, yy = x[use], y[use]
        ids = sorted(map(int, ckpt['client_model_states']))
        if a.clients:
            ids = [cid for cid in ids if cid in a.clients]
        for pos, cid in enumerate(ids, 1):
            model, _ = _make_denice_client_model(ckpt, cid, a.device)
            model.eval()
            if model.adapter_registry or model.continual_head is not None:
                raise ValueError('Unexpected inference branch')
            before = fingerprint(model)
            available = list(map(int, state(model)['available']))
            h, s, raw = infer(model, xx, config['candle_v2_logit_scale'], a.batch_size)
            pred = argmax(s, available)
            raw_pred = argmax(raw, available)
            oracle = np.full(len(yy), -1, np.int64)
            oracle_all = oracle.copy()
            per_task = []
            for t in range(task+1):
                ix = yy//6 == t
                classes = list(range(t*6, min((t+1)*6,34)))
                allowed = sorted(set(classes)&set(available))
                oracle[ix] = argmax(s[ix], allowed)
                oracle_all[ix] = argmax(s[ix], classes)
                if t == task:
                    birth[(cid,t)] = allowed
                reference = birth.get((cid,t))
                entry = dict(true_task=t, native=metrics(yy[ix], pred[ix]),
                    oracle_available=metrics(yy[ix], oracle[ix]),
                    oracle_all_task_rows=metrics(yy[ix], oracle_all[ix]), available=allowed)
                if reference:
                    entry['birth_mask_oracle'] = metrics(yy[ix], argmax(s[ix], reference))
                per_task.append(entry)
            old = [c for c in available if c < task*6]
            new = [c for c in available if c >= task*6]
            margin = s[:,new].max(1)-s[:,old].max(1) if old and new else None
            old_only = metrics(yy[yy<task*6], argmax(s[yy<task*6], old)) if old else None
            row = dict(task=task, client_id=cid, source=source, available=available,
                native=metrics(yy,pred), raw_affine=metrics(yy,raw_pred),
                oracle_available=metrics(yy,oracle), oracle_all_task_rows=metrics(yy,oracle_all),
                covered_rows=int(np.isin(yy,available).sum()), by_true_task=per_task,
                predicted_task_counts=np.bincount(pred//6,minlength=6).tolist(),
                old_only=old_only,
                old_input_new_minus_old=distribution(margin[yy<task*6]) if margin is not None else None,
                new_input_new_minus_old=distribution(margin[yy>=task*6]) if margin is not None else None,
                zero_feature_rows=int((np.linalg.norm(h,axis=1)<=1e-6).sum()),
                active_fc1=int((np.asarray(model.unit_ranks['fc1'])>0).sum()),
                mature_fc2=int((np.asarray(model.unit_ranks['fc2'])>1).sum()))
            effective = (model.fc2.weight*model.weight_masks['fc2'].to(model.fc2.weight)).detach().cpu().numpy()
            row['zero_effective_available_rows'] = [c for c in available if np.linalg.norm(effective[c])<=1e-6]
            if task == 0:
                base[cid] = dict(h=h.copy(), effective=effective.copy(), mask=model.weight_masks['fc2'].cpu().numpy().copy(), available=available)
                # Full T0 parameters are retained only in RAM for drift diagnosis.
                baseline_models[cid] = {k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
            elif cid in base:
                ref = base[cid]; old_h = h[yy<6]
                hn = old_h/np.maximum(np.linalg.norm(old_h,axis=1,keepdims=True),1e-8)
                rn = ref['h']/np.maximum(np.linalg.norm(ref['h'],axis=1,keepdims=True),1e-8)
                row['T0_reference'] = dict(feature_cosine=distribution((hn*rn).sum(1)),
                    old_effective_row_max_abs_change=float(np.abs(effective[ref['available']]-ref['effective'][ref['available']]).max()),
                    old_connection_mask_changed=int(np.count_nonzero(model.weight_masks['fc2'].cpu().numpy()[ref['available']] != ref['mask'][ref['available']])),
                    parameter_max_abs_change={k:float(np.abs(v.detach().cpu().numpy()-baseline_models[cid][k]).max()) for k,v in model.state_dict().items() if k in baseline_models[cid] and v.is_floating_point()})
            if fingerprint(model) != before:
                raise ValueError('Evaluation mutated the checkpoint model')
            row['model_fingerprint_before_after'] = before
            write(a.out/f'T{task}_client_{cid}.json', row)
            np.savez_compressed(a.out/f'T{task}_client_{cid}_scores.npz', y=yy, cosine=s, raw=raw, pred=pred, oracle=oracle)
            records.append(row)
            print(f'T{task} receiver={cid} ({pos}/{len(ids)}) native={row["native"]["accuracy"]:.2%} task-oracle={row["oracle_available"]["accuracy"]:.2%}', flush=True)
            del model, h, s, raw
        del ckpt
        gc.collect()
        torch.cuda.empty_cache()
    summary = dict(completed=True, role_sha256=role_sha, no_final_TEST=True, no_training=True,
        panel_rows=len(y), cap_per_class=a.cap, seconds=time.monotonic()-started,
        records=records, limitation='Descriptive frozen-checkpoint development diagnostics, balanced unique panel; no fitting or independent confirmation')
    write(a.out/'completion.json', summary)
    print(f'Completed {len(records)} model evaluations in {summary["seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
