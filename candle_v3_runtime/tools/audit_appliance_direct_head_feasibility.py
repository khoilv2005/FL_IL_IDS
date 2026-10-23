"""Read-only consistency audit of a completed direct-head pilot; no CAL opens."""
import argparse
import json
from pathlib import Path

from appliance.direct_head_contract import RULES
from appliance.direct_head_verification import validate_receipts, accept
from fed_learning.data.denice_clean_roles import file_sha256


def audit(root):
    root = Path(root)
    protocol = json.loads((root/'protocol_before_CAL.json').read_text(encoding='utf-8'))
    lock = json.loads((root/'protocol_lock.json').read_text(encoding='utf-8'))
    assert file_sha256(root/'protocol_before_CAL.json') == lock['sha256']
    summary = json.loads((root/'execution/completion.json').read_text(encoding='utf-8'))
    assert summary['completed_execution'] and summary['error'] is None
    checks, table = 1, []
    for position, result in enumerate(summary['results']):
        if not result.get('HOLDOUT_opened'):
            continue
        frozen = json.loads((root/f'execution/case_{position}_lock_before_HOLDOUT.json').read_text(encoding='utf-8'))
        evidence = json.loads((root/f'execution/case_{position}_HOLDOUT.json').read_text(encoding='utf-8'))
        receipts = {int(k): v for k, v in evidence.items()}
        case = result['case']
        validate_receipts(receipts, case['receiver'], case['donors'], case['task'],
            protocol['role_sha256'], 'holdout', frozen['function_hashes'])
        assert accept(receipts, case['receiver'], case['donors']) == result['qualification']
        checks += 2
        for receipt in receipts.values():
            for variant in receipt['variants'].values():
                for stats in [variant['stats'], *variant['comparisons'].values()]:
                    assert stats['net_rescue'] == stats['rescue']-stats['break_count']
                    assert stats['positive']+stats['negative'] == stats['rows']
                    assert sum(v['rows'] for v in stats['by_negative_class'].values()) == stats['negative']
                    assert sum(v['false_positive'] for v in stats['by_negative_class'].values()) == stats['false_positive']
                    checks += 4
        table.append(dict(task=case['task'], receiver=case['receiver'], class_id=case['class_id'], eta=result['eta'],
            owners=[dict(owner=o, positive=receipts[o]['variants']['head_agg']['stats']['positive'],
                recalls={n: v['stats']['recall'] for n, v in receipts[o]['variants'].items()},
                false_positive=receipts[o]['variants']['head_agg']['stats']['false_positive'],
                negative=receipts[o]['variants']['head_agg']['stats']['negative'],
                negative_break=receipts[o]['variants']['head_agg']['stats']['negative_break']) for o in receipts],
            qualification=result['qualification']))
    messages = [json.loads(line) for line in (root/'execution/communication.jsonl').read_text().splitlines()]
    total = sum(m['bytes'] for m in messages)
    assert total == summary['communication']['application_egress_bytes'] <= RULES['application_bytes']
    assert sum(m['kind'] == 'cold_receiver_function_reference' for m in messages) <= 6
    checks += 2
    accesses = json.loads((root/'execution/CAL_access.json').read_text(encoding='utf-8'))
    for endpoint in accesses:
        task = protocol['cases'][endpoint['case']]['task']
        for entry in endpoint['access_log']:
            assert entry['task'] == task
            assert entry['client_id'] == endpoint['owner'] and entry['role'] == 'calibration'
            checks += 2
    return dict(checks=checks, passed=True, source=root.as_posix(), table=table,
        application_bytes=total, read_only_counts_audit=True,
        predictions_independently_recomputed=False, raw_CAL_opened=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    result = audit(a.root)
    if a.out:
        a.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
