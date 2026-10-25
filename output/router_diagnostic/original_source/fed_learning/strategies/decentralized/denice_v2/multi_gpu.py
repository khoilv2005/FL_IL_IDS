"""One persistent spawned process per GPU; independent client optimizers.

Schedule across all clusters so singleton clusters can use both GPUs too.
TRAIN stays in current-task mmap files. References are typed, checksummed CPU
state files; no CUDA tensor or whole TRAIN tensor is sent through IPC.
"""
from collections import deque
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import fields
import gc
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .aggregation import RoundPlan
from .contracts import encode,decode
from .native import CurrentBase,train_native_worker
from .state import snapshot,restore


_DEVICE=None
_PLAN_CACHE={}
_GENERATION=None


def initialize_gpu_worker(device):
    global _DEVICE
    _DEVICE=device
    torch.set_num_threads(2)
    torch.cuda.set_device(torch.device(device))
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    print(f'DeNICE V2 GPU worker ready: {device} ({torch.cuda.get_device_name(torch.device(device))})',flush=True)


def export_plan(plan,path):
    plan.validate()
    value={f.name:getattr(plan,f.name) for f in fields(plan) if f.name!='reference'}
    value['reference']=snapshot(plan.reference)
    raw=json.dumps(encode(value),separators=(',',':'),allow_nan=False).encode()
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def load_plan(path,expected):
    global _GENERATION
    generation=str(Path(path).parent)
    if generation!=_GENERATION:
        _PLAN_CACHE.clear();_GENERATION=generation
    key=(path,expected)
    if key not in _PLAN_CACHE:
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('GPU worker reference checksum mismatch')
        value=decode(json.loads(raw))
        value['reference']=restore(value['reference'])
        plan=RoundPlan(**value);plan.validate()
        _PLAN_CACHE[key]=plan
    return _PLAN_CACHE[key]


def train_gpu_job(plan_path,plan_sha,descriptor,base_fingerprint,training):
    if _DEVICE is None:raise RuntimeError('GPU worker not initialized')
    started=time.monotonic()
    plan=load_plan(plan_path,plan_sha)
    values={k:np.load(p,mmap_mode='c',allow_pickle=False) for k,p in descriptor['paths'].items()}
    base=CurrentBase(descriptor['client_id'],descriptor['task'],torch.from_numpy(values['x']),
        torch.from_numpy(values['y']),values['rows'],descriptor['source_hash'],
        role=descriptor['role'],allow_singleton=descriptor['allow_singleton'])
    if base.fingerprint()!=base_fingerprint:raise ValueError('GPU worker current TRAIN content mismatch')
    print(f'Native local T{base.task} R{plan.target_round} cluster={plan.reference.cluster_uid} '
        f'client={base.client_id} GPU={_DEVICE} TRAIN={len(base.y):,}',flush=True)
    packet,receipt=train_native_worker(plan,base,device=_DEVICE,**training)
    receipt.update(device=_DEVICE,execution_policy='spawn_process_per_gpu_lpt_v1',
        elapsed_seconds=time.monotonic()-started,batch_size=training['batch_size'])
    print(f'  GPU={_DEVICE} client={base.client_id} optimizer steps={receipt["steps"]} '
        f'loss={receipt["loss_mean"]} skip={receipt.get("skipped")}',flush=True)
    # Return CPU numpy deltas/masks only; release CUDA allocations before the
    # parent starts canonical BN/router work on its GPU.
    del base,values
    gc.collect();torch.cuda.empty_cache()
    return packet,receipt


class GpuWorkerPool:
    def __init__(self,devices,staging_root):
        self.devices=tuple(devices)
        self.staging_root=Path(staging_root).resolve()
        if len(self.devices)!=2 or len(set(self.devices))!=2:
            raise ValueError('This execution policy requires exactly two distinct GPU devices')
        indices=[]
        for name in self.devices:
            device=torch.device(name)
            if device.type!='cuda' or device.index is None:raise ValueError('Use explicit cuda:N GPU identifiers')
            indices.append(device.index)
        if not torch.cuda.is_available() or max(indices)>=torch.cuda.device_count():
            raise RuntimeError('Enable Kaggle T4 x2: both requested GPUs must be visible')
        context=multiprocessing.get_context('spawn')
        self.executors=[ProcessPoolExecutor(max_workers=1,mp_context=context,
            initializer=initialize_gpu_worker,initargs=(device,)) for device in self.devices]

    def close(self):
        for executor in self.executors:executor.shutdown(wait=True,cancel_futures=True)
        self.executors=[]

    def train_round(self,prepared,data,directory,training,*,budget_guard=None):
        started=time.monotonic()
        directory=Path(directory).resolve()
        if directory.parent!=self.staging_root:raise ValueError('GPU staging escaped configured cache root')
        directory.mkdir(parents=True,exist_ok=True)
        jobs=[];plan_files={}
        for uid,(plan,allocation,bases) in prepared.items():
            path=directory/f'{uid}.json'
            plan_files[uid]=(str(path),export_plan(plan,path))
            if budget_guard is not None:budget_guard()
            for cid in plan.reference.members:
                rows=len(bases[cid].y)
                jobs.append((uid,cid,rows,data.base_descriptor(cid),bases[cid].fingerprint()))
        queues=[deque() for _ in self.devices];loads=[0 for _ in self.devices];assignments={}
        # Stable longest-job-first greedy balancing; no test/validation evidence.
        for uid,cid,rows,descriptor,fingerprint in sorted(jobs,key=lambda j:(-j[2],j[0],j[1])):
            slot=min(range(len(loads)),key=lambda i:(loads[i],i))
            queues[slot].append((uid,cid,descriptor,fingerprint))
            loads[slot]+=max(1,(rows+training['batch_size']-1)//training['batch_size'])*training['epochs']
            assignments[cid]=self.devices[slot]
        print('Parallel local scheduling: '+str({d:sum(v==d for v in assignments.values()) for d in self.devices})+
            f'; batch/client={training["batch_size"]}',flush=True)
        outputs={uid:{} for uid in prepared};inflight={}

        def submit(slot):
            if queues[slot]:
                uid,cid,descriptor,fingerprint=queues[slot].popleft()
                path,sha=plan_files[uid]
                future=self.executors[slot].submit(train_gpu_job,path,sha,descriptor,fingerprint,training)
                inflight[future]=(slot,uid,cid)

        for slot in range(len(queues)):submit(slot)
        while inflight:
            done,_=wait(tuple(inflight),return_when=FIRST_COMPLETED)
            for future in sorted(done,key=lambda f:inflight[f]):
                slot,uid,cid=inflight.pop(future)
                try:packet,receipt=future.result()
                except Exception as exc:
                    raise RuntimeError(f'GPU worker failed: {self.devices[slot]}, cluster={uid}, client={cid}') from exc
                if packet.event.client_id!=cid or receipt['client']!=cid or receipt['device']!=self.devices[slot]:
                    raise ValueError('GPU result assignment mismatch')
                outputs[uid][cid]=(packet,receipt)
                submit(slot)
        # All futures completed and workers no longer read these files.
        if directory.parent!=self.staging_root:raise ValueError('Unsafe GPU staging cleanup path')
        shutil.rmtree(directory)
        return outputs,dict(policy='spawn_process_per_gpu_lpt_v1',devices=self.devices,
            client_assignments=assignments,estimated_steps_by_device=dict(zip(self.devices,loads)),
            parallel_local_seconds=time.monotonic()-started,jobs=len(jobs),shared_reference_per_cluster=True,
            whole_train_arrays_sent_over_ipc=False)
