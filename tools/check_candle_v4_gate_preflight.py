"""Controlled metadata guard probes only; no native/expert training or CAL access."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from run_candle_v4_transition import spec,ROOT
from pilot_candle_v4_standalone import write,file_sha256
from fed_learning.training.candle_v4_bank import ExpertBank


def main(out):
    graph={'alpha_debug':{18:{'group_ids':[8,18,30],'alphas':[.26,.42,.32]}}}
    checks={}; receipts={}
    for name in ('untrained','unreachable_owner','no_active'):
        bank=ExpertBank(spec(out))
        if name=='unreachable_owner':
            # Synthetic counters exercise guard ordering, not a trained-expert claim.
            for e in bank.engines.values():e.steps=1
        if name=='no_active':
            bank.status={c:'REJECTED' for c in bank.engines}
        with patch.object(bank,'role',side_effect=AssertionError('FIT/CAL must not open')) as access:
            receipt=bank.finish_task(3 if name=='no_active' else 2,{}, {}, {}, [8,16,18,30],graph)
        expected={'untrained':'DEFERRED_NO_AUTHORIZED_TRAINING','unreachable_owner':'DEFERRED_GATE_GRAPH_SCOPE',
                  'no_active':'SKIPPED_NO_ACTIVE_BANK'}[name]
        checks[name]=receipt['outcome']==expected and access.call_count==0 and bank.bytes==0
        checks[name+'_no_ACTIVE']=all(s!='ACTIVE' for s in bank.status.values())
        receipts[name]=receipt
    report=dict(completed=all(checks.values()),checks=checks,receipts=receipts,
        scope='Pure preflight guard checks after the single native smoke; zero native/expert optimizer steps, no real ACTIVE bank or CAL outcomes',
        synthetic_counter_for_guard_ordering=True,source_sha256={p:file_sha256(ROOT/p) for p in (
            'candle_v4_runtime/fed_learning/training/candle_v4_bank.py','tools/check_candle_v4_gate_preflight.py')})
    write(ROOT/'artifacts/candle_v4_gate_preflight_fix_20261023.json',report)
    if not report['completed']:raise AssertionError(checks)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out.resolve())
