"""P3 round-boundary orchestrator. Quality is reported, correctness is enforced."""
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from fed_learning.models.denice_model import DeNICEModel
from .archive import TaskArchive
from .contracts import encode, state_hash
from .coordinate_router import fresh_coordinate_state
from .election import elect_leader
from .evaluation import atomic_json, evaluate_cluster, aggregate_evaluations, forgetting
from .knowledge import ClassKnowledgeLedger
from .native import consolidate_native_task,load_native,run_native_round,prepare_native_round,finalize_native_round
from .partition import bootstrap_similarity, SimilaritySnapshot, partition_clients, PartitionBook
from .router import stage_cluster_router
from .state import snapshot, restore, handoff, merge_gate


@dataclass(frozen=True)
class RunConfig:
    seed: int = 42
    rounds_per_task: int = 20
    tasks: int = 6
    xi: float = .8
    batch_size: int = 4086
    local_epochs: int = 1
    learning_rate: float = .001
    eval_batch_size: int = 8192
    context_rows: int = 1024
    output_budget_gib: float = 18.
    device: str = 'cpu'
    worker_devices: tuple = ()
    engineering_baseline: bool = True
    policy: str = 'cnn_coordinate_router_v1'
    feature_layer: str = 'conv2'
    eval_final_task_only: bool = True
    smoke: bool = False

    def validate(self):
        if (self.policy!='cnn_coordinate_router_v1' or self.feature_layer!='conv2' or self.xi!=.8
            or not self.engineering_baseline or not np.isfinite([self.learning_rate,self.output_budget_gib]).all()
            or self.learning_rate<=0 or self.output_budget_gib<=0 or self.batch_size<2 or self.local_epochs<1
            or self.eval_batch_size<1 or self.context_rows<1 or self.rounds_per_task<1 or not 1<=self.tasks<=6
            or type(self.eval_final_task_only) is not bool):
            raise ValueError('Invalid locked V2 configuration')
        if not self.smoke and (self.tasks!=6 or self.rounds_per_task!=20 or self.seed!=42 or not self.eval_final_task_only):
            raise ValueError('Full main run is seed42, six tasks, twenty rounds/task, final-task-only evaluation')
        if self.worker_devices:
            if len(self.worker_devices)!=2 or len(set(self.worker_devices))!=2:
                raise ValueError('Parallel training requires exactly two distinct GPU workers')
            if any(torch.device(d).type!='cuda' or torch.device(d).index is None for d in self.worker_devices):
                raise ValueError('Worker devices must use explicit cuda:N identifiers')


def wire_bytes(value):
    return len(json.dumps(encode(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode())


class Runner:
    def __init__(self, data, output, config=RunConfig()):
        config.validate()
        self.data, self.output, self.config = data, Path(output), config
        self.output.mkdir(parents=True,exist_ok=True)
        self.states, self.active = {}, ()
        self.book = PartitionBook()
        self.ledger = ClassKnowledgeLedger(data.task_classes)
        self.communication, self.evaluations = [], []
        self.next_task, self.next_round, self.task_end_pending = 0, 0, False
        self.lock = state_hash(dict(config=asdict(config),data_binding=data.binding,
            input_shape=data.input_shape,clients=data.client_ids,task_classes=data.task_classes,
            optimizer_policy='round_phase_reset_fp32_v1',evaluation='full_cumulative_each_cluster_uniform_mean_v1'))
        self._test = None
        self._gpu_pool=None
        self._parallel_timing={}

    def unique_states(self):
        return {s.cluster_uid:s for s in self.states.values()}

    def checkpoint_payload(self):
        return dict(schema=1, lock=self.lock,config=asdict(self.config),data_binding=self.data.binding,
            states={uid:snapshot(s) for uid,s in self.unique_states().items()},
            client_state={cid:s.cluster_uid for cid,s in self.states.items()}, active=self.active,
            partition_book=self.book.snapshot(), ledger=self.ledger.snapshot(task=5,round=10**9),
            communication=self.communication,evaluations=self.evaluations,
            next_task=self.next_task,next_round=self.next_round,task_end_pending=self.task_end_pending,
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))

    def resume(self, task, round_id=None, source_root=None):
        source=TaskArchive(source_root or self.output,task,self.config.output_budget_gib)
        p = source.load(round_id)
        if p['schema']!=1 or p['lock']!=self.lock or p['data_binding']!=self.data.binding:
            raise ValueError('Resume protocol/dataset lock mismatch')
        states = {uid:restore(s) for uid,s in p['states'].items()}
        self.states = {cid:states[uid] for cid,uid in p['client_state'].items()}
        if set(self.states)!=set(self.data.client_ids) or any(cid not in s.members for cid,s in self.states.items()):
            raise ValueError('Resume membership mismatch')
        self.active=p['active']; self.book=PartitionBook.restore(p['partition_book'])
        self.ledger=ClassKnowledgeLedger.restore(p['ledger'])
        self.communication=p['communication']; self.evaluations=p['evaluations']
        self.next_task=p['next_task'];self.next_round=p['next_round'];self.task_end_pending=p['task_end_pending']
        random.setstate(p['rng']['python']);np.random.set_state(p['rng']['numpy']);torch.set_rng_state(p['rng']['torch'])
        if p['rng']['cuda']:
            if not torch.cuda.is_available() or len(p['rng']['cuda'])!=torch.cuda.device_count():
                raise ValueError('Resume CUDA RNG topology mismatch')
            torch.cuda.set_rng_state_all(p['rng']['cuda'])
        if source.root!=self.output.resolve() and p['next_task']==task:
            TaskArchive(self.output,task,self.config.output_budget_gib).import_prefix(source,
                round_id if round_id is not None else p['next_round']-1)
        # Completed task metrics travel in the checkpoint, not raw TEST rows.
        # Recreate reports when moving continuation to another Kaggle session.
        for t in sorted({r['task'] for r in self.evaluations}):
            rows=[r for r in self.evaluations if r['task']==t]
            summary=aggregate_evaluations(rows)
            summary['forgetting']=(dict(mean_forgetting=None,reason='NOT_MEASURED_FINAL_TASK_ONLY',lineage_task_pairs=[])
                if self.config.eval_final_task_only else forgetting(rows,[r for r in self.evaluations if r['task']<t]))
            summary.update(task=t,round=self.config.rounds_per_task-1,quality_is_report_only=True)
            atomic_json(self.output/f'metrics_task_{t}.json',summary)
        print(f'Restored checkpoint T{task} / round={round_id}; next T{self.next_task} R{self.next_round}',flush=True)

    def initialize(self):
        torch.manual_seed(self.config.seed);np.random.seed(self.config.seed);random.seed(self.config.seed)
        template=fresh_coordinate_state(DeNICEModel(self.data.input_shape,34),feature_layer='conv2',
            members=self.data.client_ids,coordinator_id=min(self.data.client_ids),cluster_uid='fresh-population')
        self.states={cid:template for cid in self.data.client_ids}

    def start_task(self, task):
        active=tuple(cid for cid in self.data.client_ids if sum(self.data.counts[cid].values()))
        if not active:
            raise ValueError('No active BASE clients for this task')
        reference=state_hash({uid:s.fingerprint() for uid,s in self.unique_states().items()})
        if task==0:
            sim=bootstrap_similarity({cid:self.data.counts[cid] for cid in active},data_task=0,
                task_classes=self.data.task_classes[0],reference_hash=reference)
            partition=partition_clients(sim,threshold=self.config.xi)
        else:
            prototypes=[]
            for cid in active:
                base=self.data.base(cid)
                # Native current BASE contexts; no historical input, future
                # support or TEST used. Compare common-parent features; merge
                # gate rejects distinct, incompatible model/router histories.
                rows=np.linspace(0,len(base.y)-1,min(len(base.y),self.config.context_rows),dtype=int)
                model=load_native(self.states[cid],self.config.device).eval()
                with torch.no_grad():
                    z=model.get_context_activations_per_sample(base.x[rows].to(self.config.device))['conv2'].mean(0).cpu().numpy()
                if not np.isfinite(z).all():
                    raise ValueError('Nonfinite current context capsule')
                prototypes.append(z);del model,base
            z=np.asarray(prototypes,np.float64)
            norms=np.linalg.norm(z,axis=1)
            scores=np.divide(z@z.T,norms[:,None]*norms[None,:],out=np.zeros((len(active),len(active))),
                where=norms[:,None]*norms[None,:]>0)
            scores=np.clip(scores,-1,1);np.fill_diagonal(scores,1)
            sim=SimilaritySnapshot(active,tuple(map(tuple,scores)),task,task,'current_base_context_capsules',reference)
            partition=partition_clients(sim,threshold=self.config.xi,
                merge_check=merge_gate({cid:self.states[cid] for cid in active}))
        self.book.freeze(partition,active)
        previous=self.states
        new={}; groups=list(partition.clusters)
        # Persist dormant states with explicit membership and causal task time.
        for state in self.unique_states().values():
            dormant=tuple(cid for cid in state.members if cid not in active)
            if dormant:groups.append(dormant)
        active_uids=[]
        for index,members in enumerate(groups):
            parents={previous[cid].cluster_uid:previous[cid] for cid in members}
            uid=f't{task}-'+state_hash((tuple(sorted(parents)),members))[:16]
            if task==0:
                p=snapshot(next(iter(parents.values())))
                p.update(cluster_uid=uid,members=members,coordinator_id=elect_leader(members,self.ledger,task=0).leader)
                state=restore(p)
            else:
                state=handoff(tuple(parents.values()),members=members,cluster_uid=uid,task=task,ledger=self.ledger)
            for cid in members:new[cid]=state
            if index<len(partition.clusters):active_uids.append(uid)
        if set(new)!=set(self.data.client_ids):raise ValueError('Task handoff lost clients')
        self.states,self.active=new,tuple(active_uids)
        # Explicit metadata/capsule/handoff traffic model, includes broadcasts.
        metadata=sum(wire_bytes(dict(client=cid,counts=self.data.counts[cid],knowledge=self.ledger.learned_classes(cid,task=task))) for cid in active)
        self.communication.append(dict(task=task,round=-1,phase='task_partition_handoff',metadata_bytes=metadata,
            current_context_bytes=0 if task==0 else sum(wire_bytes(z) for z in prototypes),
            state_handoff_bytes=sum(wire_bytes(snapshot(new[min(g)]))*(len(g)-1) for g in partition.clusters)))
        atomic_json(self.output/f'partition_task_{task}.json',dict(task=task,K=partition.K,
            clusters=partition.clusters,active_clients=active,dormant_clients=sorted(set(self.data.client_ids)-set(active)),
            source=partition.source,rejected_merges=partition.rejected_merges,similarity_hash=partition.similarity_hash,
            coordinators={uid:self.unique_states()[uid].coordinator_id for uid in self.active}))
        print(f'Task {task}: auto K={partition.K}, active={len(active)}, dormant={len(new)-len(active)}, singleton={sum(len(g)==1 for g in partition.clusters)}',flush=True)

    def cluster_round_results(self,task,round_id):
        training=dict(seed=self.config.seed+task*100000+round_id*1000,batch_size=self.config.batch_size,
            epochs=self.config.local_epochs,lr=self.config.learning_rate,engineering_baseline=True)
        if not self.config.worker_devices:
            for uid in self.active:
                state=self.unique_states()[uid]
                bases={cid:self.data.base(cid) for cid in state.members}
                result,plan,packets,receipt=run_native_round(state,bases,self.data.task_classes[task],
                    device=self.config.device,log_progress=True,**training)
                yield uid,bases,result,plan,packets,receipt
            return
        if self._gpu_pool is None:
            from .multi_gpu import GpuWorkerPool
            self._gpu_pool=GpuWorkerPool(self.config.worker_devices,self.output/'current_data_cache')
        started=time.monotonic();prepared={}
        for uid in self.active:
            state=self.unique_states()[uid]
            bases={cid:self.data.base(cid) for cid in state.members}
            plan,allocation=prepare_native_round(state,bases,self.data.task_classes[task])
            prepared[uid]=(plan,allocation,bases)
            print(f'Common reference ready T{task} R{round_id} cluster={uid} members={len(state.members)}',flush=True)
        prepare_seconds=time.monotonic()-started
        outputs,execution=self._gpu_pool.train_round(prepared,self.data,
            self.output/'current_data_cache'/f'gpu_jobs_task_{task}_round_{round_id}',training,
            budget_guard=TaskArchive(self.output,task,self.config.output_budget_gib).guard)
        self._parallel_timing=dict(prepare_seconds=prepare_seconds,**execution)
        for uid in self.active:
            plan,allocation,bases=prepared[uid]
            rows=[outputs[uid][cid] for cid in plan.reference.members]
            packets=[r[0] for r in rows];workers=[r[1] for r in rows]
            result,plan,packets,receipt=finalize_native_round(plan,bases,packets,workers,allocation,
                batch_size=self.config.batch_size,device=self.config.device)
            receipt['parallel_execution']=dict(policy=execution['policy'],devices=self.config.worker_devices,
                client_assignments={cid:execution['client_assignments'][cid] for cid in plan.reference.members})
            yield uid,bases,result,plan,packets,receipt

    def round(self, task, round_id):
        started=time.monotonic();self._parallel_timing={}
        for uid,bases,result,plan,packets,receipt in self.cluster_round_results(task,round_id):
            state=plan.reference
            if any(p.event.successful_steps for p in packets):
                proposal=stage_cluster_router(result.state,plan,packets,bases,feature_policy=self.config.policy,
                    batch_size=self.config.batch_size,device=self.config.device,engineering_baseline=True)
                final=proposal.commit(engineering_baseline=True)
                routing=proposal.receipt
            else:
                final=result.state
                routing=dict(skipped='NO_SUCCESSFUL_CURRENT_TRAINING',capacity_report_only=True,
                    threshold_wire_bytes_by_client={},sketch_wire_bytes_by_client={})
            # Protected coordinates and prior routing memories are correctness,
            # not quality metrics; validate on every completed cluster round.
            final.validate()
            for name,mask in state.protected_parameters.items():
                a=state.model_state[name].numpy();b=final.model_state[name].numpy()
                m=np.asarray(mask,bool)
                if not np.array_equal(a[m],b[m]):raise ValueError(f'Protected mutation: {name}')
            if state.router_state['activation_memory'] and final.routing_space_hash!=state.routing_space_hash:
                raise ValueError('Historical routing function drift')
            for t,bank in state.router_state['activation_memory'].items():
                if t<task and state_hash(bank)!=state_hash(final.router_state['activation_memory'][t]):
                    raise ValueError('Historical routing memory changed')
            for packet in packets:self.ledger.record(packet.event,at_task=task,at_round=round_id)
            for cid in final.members:self.states[cid]=final
            leader=state.coordinator_id
            bn_bytes=sum(n for call in receipt['bn_collector_calls'] for cid,n in call['wire_bytes_by_client'].items() if cid!=leader)
            self.communication.append(dict(task=task,round=round_id,cluster_uid=uid,phase='cluster_round',
                reference_broadcast_bytes=wire_bytes(snapshot(plan.reference))*(len(state.members)-1),
                model_delta_upload_bytes=sum(wire_bytes(asdict(p)) for p in packets if p.event.client_id!=leader),
                bn_moment_upload_bytes=bn_bytes,
                router_upload_bytes=sum(n for k in ('threshold_wire_bytes_by_client','sketch_wire_bytes_by_client')
                    for cid,n in routing[k].items() if cid!=leader),
                final_state_broadcast_bytes=wire_bytes(snapshot(final))*(len(state.members)-1)))
            atomic_json(self.output/f'round_task_{task}_{round_id:02d}_{uid}.json',
                dict(task=task,round=round_id,cluster_uid=uid,native=receipt,router=routing,final_state_hash=final.fingerprint()))
            del bases,result,plan,packets,receipt,final;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        self.next_round=round_id+1;self.task_end_pending=self.next_round==self.config.rounds_per_task
        compute_seconds=time.monotonic()-started
        checkpoint_started=time.monotonic()
        saved=TaskArchive(self.output,task,self.config.output_budget_gib).save(round_id,self.checkpoint_payload())
        checkpoint_seconds=time.monotonic()-checkpoint_started
        atomic_json(self.output/f'timing_task_{task}_round_{round_id:03d}.json',dict(task=task,round=round_id,
            compute_seconds=compute_seconds,checkpoint_seconds=checkpoint_seconds,parallel=self._parallel_timing))
        print(f'Round complete T{task} R{round_id}: checkpoint {self.output/saved["path"]}; '
            f'{saved["bytes"]/1024**2:.1f} MiB; output {saved["total_output_bytes"]/1024**3:.2f} GiB',flush=True)
        print(f'Round timing: compute={compute_seconds:.1f}s, parallel_local={self._parallel_timing.get("parallel_local_seconds",0):.1f}s, '
            f'checkpoint={checkpoint_seconds:.1f}s',flush=True)

    def finish_task(self, task):
        summary=None
        if not self.config.eval_final_task_only or task==self.config.tasks-1:
            if self.next_round!=self.config.rounds_per_task:
                raise ValueError('Evaluation requires the final completed training round')
            if self._test is None:self._test=self.data.test()
            x,y,binding=self._test
            TaskArchive(self.output,task,self.config.output_budget_gib).guard()
            results=[evaluate_cluster(self.unique_states()[uid],x,y,binding,
                self.output/f'evaluation_task_{task}_{uid}.json',batch_size=self.config.eval_batch_size,device=self.config.device,
                evaluation_role=getattr(self.data,'evaluation_role','test'))
                for uid in self.active]
            # JSON-normalize keys so restored and uninterrupted metrics coincide.
            results=json.loads(json.dumps(results))
            summary=aggregate_evaluations(results)
            summary['forgetting']=(dict(mean_forgetting=None,reason='NOT_MEASURED_FINAL_TASK_ONLY',lineage_task_pairs=[])
                if self.config.eval_final_task_only else forgetting(results,self.evaluations))
            summary.update(task=task,round=self.config.rounds_per_task-1,quality_is_report_only=True)
            atomic_json(self.output/f'metrics_task_{task}.json',summary)
            self.evaluations.extend(results)
        for uid in self.active:
            state=self.unique_states()[uid]
            if task in state.router_state['activation_memory']:
                final=consolidate_native_task(state)
                for cid in final.members:self.states[cid]=final
        # Dormant parent tasks advance in start_task without reading old BASE.
        self.next_task=task+1;self.next_round=0;self.task_end_pending=False
        archive=TaskArchive(self.output,task,self.config.output_budget_gib)
        archive.save(self.config.rounds_per_task,self.checkpoint_payload()) # explicit task-boundary continuation
        archive.seal()
        self.data.release(task)
        if summary is None:
            print(f'Task {task} DONE: evaluation skipped (final task only); checkpoint archive sealed.',flush=True)
        else:
            print(f'Task {task} DONE: Mean Cluster Accuracy={summary["mean_cluster_accuracy"]:.4%}, Macro-F1={summary["mean_cluster_macro_f1"]:.4%}',flush=True)

    def run(self, *, stop_after_rounds=None):
        try:
            return self._run(stop_after_rounds=stop_after_rounds)
        finally:
            if self._gpu_pool is not None:
                self._gpu_pool.close();self._gpu_pool=None

    def _run(self, *, stop_after_rounds=None):
        if not self.states:self.initialize()
        completed=0
        while self.next_task<self.config.tasks:
            task=self.next_task
            self.data.prepare(task)
            if task not in self.book.partitions:self.start_task(task)
            if self.task_end_pending:self.finish_task(task);continue
            for round_id in range(self.next_round,self.config.rounds_per_task):
                self.round(task,round_id);completed+=1
                if stop_after_rounds is not None and completed>=stop_after_rounds:return
            self.finish_task(task)
        self._test=None;gc.collect()
        if hasattr(self.data,'release_test'):self.data.release_test()
        total=sum(int(v) for item in self.communication for k,v in item.items() if k.endswith('_bytes'))
        atomic_json(self.output/'communication.json',dict(total_application_wire_bytes=total,entries=self.communication,
            encoding='typed_json_utf8',scope='modeled_P2P_application_messages_in_single_process_includes_native_model_BN_router_handoff',
            tcp_tls_network_overhead_included=False))
        atomic_json(self.output/'completion.json',dict(completed=True,protocol_lock=self.lock,
            config=asdict(self.config),tasks_completed=self.config.tasks,quality_is_report_only=True,
            full_training=not self.config.smoke,total_application_wire_bytes=total,
            evaluation_cadence='last_round_final_task' if self.config.eval_final_task_only else 'last_round_each_task',
            metrics_files=[f'metrics_task_{t}.json' for t in sorted({r['task'] for r in self.evaluations})],
            checkpoints=[p.name for p in sorted(self.output.glob('checkpoint_task_*_all_rounds.zip'))],
            earlier_archives_may_remain_in_resume_dataset=True))
