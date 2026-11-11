"""Summarize the fixed six-component smoke without selecting parameters."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
source=ROOT/'audit_denice/candle_v4_six_component_smoke_20261111/audit.json'
if __name__=='__main__':
    d=json.loads(source.read_text());rng=np.random.default_rng(42)
    idx=rng.integers(0,100,size=(10000,100))
    base=np.array([v['baseline'] for v in d['clients'].values()]);d['client_bootstrap']={}
    for method in d['counts']:
        if method=='baseline':continue
        delta=np.array([v[method] for v in d['clients'].values()])-base
        assert np.isclose(delta.mean(),d['metrics'][method]['accuracy']-d['metrics']['baseline']['accuracy'])
        d['client_bootstrap'][method]=dict(improved=int(np.sum(delta>0)),worse=int(np.sum(delta<0)),
            gain_pp=float(delta.mean()*100),interval95_gain_pp=np.quantile(delta[idx].mean(1)*100,[.025,.975]).tolist())
    d['interpretation']='Six adapted frozen-feature routing component probes, not complete paper reproductions or independent confirmation'
    source.write_text(json.dumps(d,indent=2),encoding='utf-8')
    (ROOT/'artifacts/candle_v4_six_component_smoke_20261111.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(d['client_bootstrap'],indent=2))
