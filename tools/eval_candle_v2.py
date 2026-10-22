"""Terminal task-free CANDLE V2 evaluation; one local predictor per test row."""
import gc
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
from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.strategies.decentralized.candle_v2 import predict_logits, state, validate
from fed_learning.training.decentralized_denice_il import _partition_test_data_by_client


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def statistics(cm):
    correct = np.diag(cm)
    support = cm.sum(1)
    predicted = cm.sum(0)
    f1 = np.divide(2 * correct, support + predicted, out=np.zeros(34), where=support+predicted > 0)
    recall = np.divide(correct, support, out=np.zeros(34), where=support > 0)
    precision = np.divide(correct, predicted, out=np.zeros(34), where=predicted > 0)
    return dict(samples=int(cm.sum()), accuracy=float(correct.sum()/cm.sum()),
                macro_f1_34=float(f1.mean()), macro_recall_34=float(recall.mean()),
                macro_precision_34=float(precision.mean()), per_class_f1=f1.tolist(),
                per_class_recall=recall.tolist(), class_counts=support.tolist())


@torch.no_grad()
def run(checkpoint, roles_dir, data_dir, out, device='cuda', batch_size=4096):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    ckpt, source = load_input(Path(checkpoint), 5, 19)
    config = ckpt['config']
    if not config.get('candle_v2_enabled'):
        raise ValueError('Expected a CANDLE V2 checkpoint')
    validate(config)
    roles = CleanRoleData(roles_dir, source_data_dir=data_dir)
    if config['denice_data_roles_sha256'] != file_sha256(roles.root/'role_manifest.json'):
        raise ValueError('Evaluation role identity differs from training')
    ids = sorted(map(int, ckpt['client_model_states']))
    write(out/'pipeline_lock.json', dict(method='CANDLE V2', source=source,
        primary='TaskFreeCosineSelf', scale=config['candle_v2_logit_scale'],
        task=5, round=19, oracle=False, donor_inference=False, task_router=False,
        test_partition='V1-matched disjoint receiver shards, each global row exactly once',
        receivers=ids, role_manifest_sha256=config['denice_data_roles_sha256']))
    # The lock is written before the only access to final TEST.
    loader = IncrementalDataLoader(str(roles.source))
    x, y = loader.get_full_test_data()
    if not np.array_equal(np.unique(y.numpy()), np.arange(34)):
        raise ValueError('Expected the original 34-class full TEST')
    expected = len(y)
    shards, partition = _partition_test_data_by_client(
        x, y, ids, seed=int(config.get('seed', 42)) + 104729*5, lazy_features=True)
    del x, y, loader
    gc.collect()
    cm = np.zeros((34, 34), dtype=np.int64)
    client_rows = []
    evaluated = 0
    covered = 0
    started = time.monotonic()
    for pos, cid in enumerate(ids, 1):
        model, _ = _make_denice_client_model(ckpt, cid, device)
        model.eval()
        if model.adapter_registry:
            raise ValueError('Task-free CANDLE V2 cannot silently select context-specific adapters')
        shard = shards[cid]
        labels = shard['y_test']
        inputs = shard['X_test']
        local_cm = np.zeros_like(cm)
        available = state(model)['available']
        local_covered = 0
        for start in range(0, len(labels), batch_size):
            truth = labels[start:start+batch_size].cpu().numpy().astype(np.int64)
            batch = inputs[start:start+batch_size].to(device)
            pred = predict_logits(model, batch, config['candle_v2_logit_scale']).argmax(1).cpu().numpy()
            local_cm += np.bincount(truth*34 + pred, minlength=34*34).reshape(34, 34)
            local_covered += int(np.isin(truth, available).sum())
            evaluated += len(truth)
            if start % (100*batch_size) == 0:
                print(f'CANDLE V2 full TEST {evaluated}/{expected}; receiver={cid}', flush=True)
        covered += local_covered
        cm += local_cm
        client_rows.append(dict(client_id=cid, available_classes=available,
                                covered_rows=local_covered, **statistics(local_cm)))
        print(f'Evaluated receiver {pos}/{len(ids)}: accuracy={client_rows[-1]["accuracy"]:.4%}', flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if evaluated != expected or cm.sum() != expected:
        raise ValueError('Full TEST row accounting mismatch')
    write(out/'confusion_matrix.json', cm.tolist())
    write(out/'per_client.json', client_rows)
    result = dict(completed=True, experiment='CANDLE V2', primary='TaskFreeCosineSelf',
        metrics=statistics(cm), test_rows=expected, evaluated_rows=evaluated,
        label_coverage=covered/expected, covered_rows=covered,
        mean_client_accuracy=float(np.mean([r['accuracy'] for r in client_rows])),
        mean_client_macro_f1=float(np.mean([r['macro_f1_34'] for r in client_rows])),
        route_accuracy=None, route_accuracy_reason='Task-free inference; no task prediction',
        partition=partition, runtime_seconds=time.monotonic()-started, source=source,
        limitation='Engineering run, not a population safety certificate or untouched confirmation')
    write(out/'completion.json', result)
    return result
