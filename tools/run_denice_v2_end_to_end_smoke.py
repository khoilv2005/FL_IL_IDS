"""Controlled P3 integration fixture, NOT a scientific accuracy benchmark."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from fed_learning.data.denice_clean_roles import file_sha256
from fed_learning.strategies.decentralized.denice_v2.contracts import state_hash
from fed_learning.strategies.decentralized.denice_v2.native import CurrentBase
from fed_learning.strategies.decentralized.denice_v2.runner import Runner, RunConfig
from fed_learning.strategies.decentralized.denice_v2.archive import TaskArchive
from fed_learning.strategies.decentralized.denice_v2.evaluation import atomic_json


class FixtureData:
    evaluation_role='controlled_fixture'
    client_ids=(0,1,2,3)
    input_shape=(39,1)
    task_classes={t:tuple(range(t*6,min(t*6+6,34))) for t in range(6)}
    binding='controlled_current_only_fixture_v1'

    def __init__(self):
        self.task=None;self.counts={};self.raw={};self.reads=[]
        for task in range(6):
            for cid in self.client_ids:
                # Two genuine clusters with different positive class support.
                labels=[task*6,task*6+1] if cid<2 else [min(task*6+2,33),min(task*6+3,33)]
                rng=np.random.default_rng(5000+task*100+cid)
                y=np.tile(labels,8).astype(np.int64)
                x=rng.normal(size=(len(y),39,1)).astype(np.float32)
                self.raw[task,cid]=(x,y,np.arange(len(y),dtype=np.int64))
        rng=np.random.default_rng(901)
        self.test_y=np.repeat(np.arange(34),3).astype(np.int64)
        self.test_x=rng.normal(size=(len(self.test_y),39,1)).astype(np.float32)

    def prepare(self,task):
        self.task=task
        self.counts={cid:{int(c):int(n) for c,n in zip(*np.unique(self.raw[task,cid][1],return_counts=True))} for cid in self.client_ids}

    def base(self,cid):
        self.reads.append((self.task,cid,'current_BASE'))
        x,y,rows=self.raw[self.task,cid]
        return CurrentBase(cid,self.task,torch.from_numpy(x),torch.from_numpy(y),rows,state_hash((self.binding,self.task,cid)))

    def test(self):return self.test_x,self.test_y,state_hash((self.test_x,self.test_y))
    def release(self,task):pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--receipt',required=True)
    parser.add_argument('--tasks',type=int,default=2)
    parser.add_argument('--rounds',type=int,default=3)
    args=parser.parse_args()
    root=Path(args.output)
    if root.exists() and any(root.iterdir()):raise ValueError('Use a fresh smoke directory')
    started=time.monotonic();torch.set_num_threads(2)
    config=RunConfig(tasks=args.tasks,rounds_per_task=args.rounds,batch_size=8,eval_batch_size=24,context_rows=16,
        eval_final_task_only=False,smoke=True)
    uninterrupted=Runner(FixtureData(),root/'uninterrupted',config)
    uninterrupted.run()
    expected={cid:state.fingerprint() for cid,state in uninterrupted.states.items()}
    expected_payload=state_hash(uninterrupted.checkpoint_payload())
    interrupted=Runner(FixtureData(),root/'resumed',config)
    interrupted.run(stop_after_rounds=1)
    restored=Runner(FixtureData(),root/'resumed',config)
    restored.resume(0,0)
    restored.run()
    actual={cid:state.fingerprint() for cid,state in restored.states.items()}
    checks=dict(state_resume_exact=expected==actual,full_payload_resume_exact=expected_payload==state_hash(restored.checkpoint_payload()),
        partition_resume_exact=uninterrupted.book.snapshot()==restored.book.snapshot(),
        knowledge_resume_exact=uninterrupted.ledger.snapshot(task=5,round=999)==restored.ledger.snapshot(task=5,round=999),
        communication_resume_exact=uninterrupted.communication==restored.communication,
        evaluation_resume_exact=uninterrupted.evaluations==restored.evaluations,
        multicluster=all(p.K>=2 for p in restored.book.partitions.values()),
        two_task_memories=any(len(s.router_state['activation_memory'])>=2 for s in restored.unique_states().values()),
        full_cumulative_rows=all(r['metrics']['rows']==min(6*(r['task']+1),34)*3 for r in restored.evaluations),
        low_accuracy_did_not_block=True)
    for task in range(config.tasks):
        archive=TaskArchive(root/'resumed',task)
        checks[f'task_{task}_self_contained']=archive.sealed.exists() and len(archive.sources())==1
        checks[f'task_{task}_every_round_restore']=all(archive.load(r)['next_round']==r+1 for r in range(config.rounds_per_task))
    root_repo=Path(__file__).resolve().parents[1]
    sources=[Path(__file__),*sorted((root_repo/'fed_learning/strategies/decentralized/denice_v2').glob('*.py'))]
    report=dict(completed=True,technical_pass=all(checks.values()),checks=checks,config=config.__dict__,
        elapsed_seconds=time.monotonic()-started,scope=f'synthetic_four_client_{config.tasks}_task_{config.rounds_per_task}_round_controlled_integration_fixture',
        official_dataset_used=False,official_full_training_started=False,
        source_sha256={p.relative_to(root_repo).as_posix():file_sha256(p) for p in sources})
    atomic_json(args.receipt,report)
    print(json.dumps(report['checks'],indent=2),flush=True)
    if not report['technical_pass']:raise RuntimeError('P3 integration smoke failed')


if __name__=='__main__':main()
