"""Owner-local CAL evaluation; FIT selection precedes any HOLDOUT call."""
import numpy as np
from .config import Rejected
from .direct_head_contract import RULES
from .functional_gap_discovery import current_split, comparison
from .evaluate import Predictor
from .state import complete_hash, digest


def scoped_pool(view, split):
    pool = current_split(view, split)
    rng = np.random.default_rng(RULES['seed']+view.client_id)
    selected = []
    for c in sorted(np.unique(pool['y'])):
        ix = np.flatnonzero(pool['y'] == c)
        selected.extend(rng.permutation(ix)[:RULES['cap_per_class']].tolist())
    ix = np.asarray(sorted(selected), np.int64)
    return dict(X=pool['X'][ix], y=pool['y'][ix], rows=pool['rows'][ix],
        binding=dict(pool['binding'], selected_rows_digest=digest(pool['rows'][ix]),
            selected_unique_rows=len(ix), cap_per_class=RULES['cap_per_class']))


def evaluate_functions(functions, pool, target, seen, device, batch_size):
    """Only x reaches the predictor. Labels stay at the owning endpoint."""
    predictor = Predictor(seen, device, batch_size)
    records, hashes = {}, {}
    for name, fn in functions.items():
        hashes[name] = complete_hash(*fn)
        records[name] = predictor.records(*fn, pool['X'])
        if complete_hash(*fn) != hashes[name]:
            raise Rejected('DIRECT_HEAD_INFERENCE_MUTATED_FUNCTION')
    baseline = records['baseline']['pred']
    result = {}
    for name, record in records.items():
        stats = comparison(baseline, record['pred'], pool['y'], target)
        result[name] = dict(function_hash=hashes[name], stats=stats,
            target_routed_to_task=int(((pool['y'] == target) & (record['task'] == pool['binding']['task'])).sum()),
            comparisons={ref: comparison(records[ref]['pred'], record['pred'], pool['y'], target)
                         for ref in ('availability', 'mask_only') if ref in records})
    return dict(binding=pool['binding'], variants=result,
        labels_used_by_predictor=False, raw_examples_transmitted=0)


def validate_receipts(receipts, receiver, donors, task, role_sha, split, expected):
    if set(receipts) != {receiver, *donors}:
        raise Rejected('DIRECT_HEAD_REQUIRED_OWNERS_MISSING')
    for owner, e in receipts.items():
        b = e['binding']
        if (b['owner'] != owner or b['task'] != task or b['split'] != split or b['role_sha256'] != role_sha
                or e['labels_used_by_predictor'] or e['raw_examples_transmitted']):
            raise Rejected('DIRECT_HEAD_RECEIPT_AUTHORITY')
        if set(e['variants']) != set(expected):
            raise Rejected('DIRECT_HEAD_RECEIPT_VARIANTS')
        if any(e['variants'][name]['function_hash'] != h for name, h in expected.items()):
            raise Rejected('DIRECT_HEAD_RECEIPT_FUNCTION')


def risk(receipts, variant):
    stats = [r['variants'][variant]['stats'] for r in receipts.values()]
    return all((s['FAR'] is None or s['FAR'] <= RULES['FAR']) and
        all(v['FAR'] <= RULES['FAR'] for v in s['by_negative_class'].values()) and
        s['negative_break'] == 0 for s in stats)


def select_eta(receipts, prefix, receiver, donors):
    if any(r['binding']['split'] != 'fit' for r in receipts.values()):
        raise Rejected('DIRECT_HEAD_SELECTION_REQUIRES_FIT')
    options = []
    for eta in RULES['eta_grid']:
        name = f'{prefix}_{eta:g}'
        ds = [receipts[j]['variants'][name]['stats'] for j in donors]
        rs = receipts[receiver]['variants'][name]['stats']
        sufficient = all(s['positive'] >= RULES['min_positive'] for s in ds) and rs['negative'] >= RULES['min_negative']
        recalls = [s['recall'] or 0. for s in ds]
        safe = risk(receipts, name)
        net = sum(r['variants'][name]['stats']['net_rescue'] for r in receipts.values())
        options.append((sufficient and safe and min(recalls) >= RULES['recall'],
                        sufficient and safe, min(recalls), net, -eta, eta))
    return max(options)[-1]


def accept(receipts, receiver, donors):
    ds = [receipts[j]['variants']['head_agg']['stats'] for j in donors]
    rs = receipts[receiver]['variants']['head_agg']['stats']
    sufficient = all(s['positive'] >= RULES['min_positive'] for s in ds) and rs['negative'] >= RULES['min_negative']
    gains = {}
    for control in ('availability', 'mask_only', 'single_1', 'single_2'):
        gains[control] = dict(
            target_hits=sum(receipts[j]['variants']['head_agg']['stats']['target_hits'] -
                receipts[j]['variants'][control]['stats']['target_hits'] for j in donors),
            net_correct=sum(receipts[o]['variants']['head_agg']['stats']['net_rescue'] -
                receipts[o]['variants'][control]['stats']['net_rescue'] for o in receipts))
    passed = bool(sufficient and all(s['recall'] >= RULES['recall'] for s in ds) and
        risk(receipts, 'head_agg') and all(gains[c]['target_hits'] > 0 and gains[c]['net_correct'] > 0
                                       for c in ('availability', 'mask_only')))
    h = receipts[receiver]['variants']['head_agg']['function_hash']
    return dict(status='EMPIRICAL_CURRENT_PASS' if passed else 'FAIL' if sufficient else 'UNKNOWN_EVIDENCE',
        candidate_function=h, required_donor_recalls=[s['recall'] for s in ds],
        current_risk_pass=risk(receipts, 'head_agg'), sufficient_evidence=sufficient,
        gains=gains, multi_donor_advantage=all(gains[c]['net_correct'] > 0 and gains[c]['target_hits'] > 0
                                            for c in ('single_1', 'single_2')),
        old_risk='UNKNOWN', population_FAR_certified=False,
        interpretation='retrospective current CAL development; not independent confirmation or full test',
        native_install_authorized=False, cumulative_safety_established=False)
