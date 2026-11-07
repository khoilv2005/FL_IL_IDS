"""Paired client bootstrap: descriptive development uncertainty, not seed replication."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'audit_denice/candle_v4_five_router_methods_20261107/audit.json'
d=json.loads(p.read_text());rng=np.random.default_rng(42)
ix=rng.integers(0,100,size=(10000,100));base=np.array([v['baseline'] for v in d['clients'].values()])
d['client_bootstrap']={}
for method in d['counts']:
    if method=='baseline':continue
    delta=np.array([v[method] for v in d['clients'].values()])-base
    assert np.isclose(delta.mean(),d['metrics'][method]['accuracy']-d['metrics']['baseline']['accuracy'])
    d['client_bootstrap'][method]=dict(improved=int(np.sum(delta>0)),worse=int(np.sum(delta<0)),
        mean_gain_pp=float(delta.mean()*100),interval95_gain_pp=np.quantile(delta[ix].mean(1)*100,[.025,.975]).tolist())
p.write_text(json.dumps(d,indent=2),encoding='utf-8')
(ROOT/'artifacts/candle_v4_five_router_methods_20261107.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(d['client_bootstrap'],indent=2))
