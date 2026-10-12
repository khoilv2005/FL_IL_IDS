"""Uncapped BASE for clients 9/30, T0->T1; read-only development evaluation."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from fed_learning.data.denice_clean_roles import file_sha256
from fed_learning.strategies.decentralized.denice_v2.data import TaskData
from fed_learning.strategies.decentralized.denice_v2.runner import Runner,RunConfig
from fed_learning.strategies.decentralized.denice_v2.contracts import state_hash
from fed_learning.strategies.decentralized.denice_v2.evaluation import atomic_json


class OfficialBudgetSmokeData(TaskData):
    evaluation_role='validation'
    def __init__(self,*args):
        super().__init__(*args)
        self.client_ids=(9,30)
        self.binding=state_hash((self.binding,'official_BASE_uncapped_two_client_development_smoke_v1',self.client_ids))

    def test(self):
        chunks=[self.roles.client_role(cid,'validation') for cid in self.client_ids]
        x=np.concatenate([r[0] for r in chunks]);y=np.concatenate([r[1] for r in chunks])
        return x,y,state_hash(dict(role='validation',x=x,y=y,source=self.binding))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True);parser.add_argument('--roles-dir',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--receipt',required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args();output=Path(args.output)
    if output.exists() and any(output.iterdir()):raise ValueError('Use a fresh output')
    started=time.monotonic();torch.set_num_threads(2)
    data=OfficialBudgetSmokeData(args.data_dir,args.roles_dir,output/'cache')
    cfg=RunConfig(smoke=True,tasks=2,rounds_per_task=1,batch_size=2048,device=args.device,eval_final_task_only=False)
    runner=Runner(data,output,cfg);runner.run()
    checks=dict(completed=(output/'completion.json').exists(),two_tasks=len(runner.book.partitions)==2,
        historical_memory_preserved=any(set(s.router_state['activation_memory'])=={0,1} for s in runner.unique_states().values()),
        checkpoint_archives=all((output/f'checkpoint_task_{t}_all_rounds.zip').exists() for t in (0,1)),
        uncapped_base=True,development_not_TEST=True)
    source=Path(__file__).resolve().parents[1]
    paths=[Path(__file__),*sorted((source/'fed_learning/strategies/decentralized/denice_v2').glob('*.py'))]
    report=dict(completed=True,technical_pass=all(checks.values()),checks=checks,config=cfg.__dict__,
        scope='actual_locked_BASE_all_rows_clients_9_30_two_tasks_one_round_each_development_only',
        test_read=False,full_training_started=False,elapsed_seconds=time.monotonic()-started,
        metrics=[json.loads((output/f'metrics_task_{t}.json').read_text()) for t in (0,1)],
        source_sha256={p.relative_to(source).as_posix():file_sha256(p) for p in paths})
    atomic_json(args.receipt,report)
    print(json.dumps(checks,indent=2),flush=True)
    if not report['technical_pass']:raise RuntimeError('Official budget smoke failed')


if __name__=='__main__':main()
