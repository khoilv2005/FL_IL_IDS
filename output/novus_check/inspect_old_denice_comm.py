import json
import zipfile
from pathlib import Path

def decode(x):
    if x[0] == 'scalar': return x[1]
    if x[0] in ('list', 'tuple'): return [decode(v) for v in x[1]]
    if x[0] == 'dict': return {decode(k):decode(v) for k,v in x[1]}
    raise ValueError(x[0])

base = Path(r'E:\Study\DACN_KLTN\Results\100-clients\DENICE\Train')
all_rows = []
for folder, archive, member in [('results (2)', 'checkpoint_task_2_round_000.zip', 'rounds/000.json'), ('results (3)', 'checkpoint_task_4_round_020.zip', 'rounds/020.json')]:
    path = base / folder / 'results_denice_cluster_server_seed_42' / archive
    with zipfile.ZipFile(path) as z:
        d = json.loads(z.read(member))
    fields = {decode(k):v for k,v in d['payload'][1]}
    rows = decode(fields['communication'])
    all_rows.append(rows)
    total = sum(v for r in rows for k,v in r.items() if k.endswith('_bytes'))
    print(json.dumps(dict(folder=folder, archive=archive, member=member, snapshot_kind=d.get('snapshot_kind'), task=d['task'], round=d['round'], records=len(rows), total_bytes=total, GiB=total/2**30, first_round=next(r for r in rows if r['phase']=='cluster_round'), last=rows[-1]),indent=2))
print('earlier_is_exact_prefix', all_rows[1][:len(all_rows[0])] == all_rows[0])
