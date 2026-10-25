"""Common-reference hierarchical learning; no raw rows cross cluster boundaries.

Simulation: client TRAIN remains in local providers/mmap, as in the native runner.
Temporary servers reduce masked deltas; an elected temporary server reduces the
partial sums and distributes one common reference to cluster servers only.
"""
from dataclasses import asdict

from .runner import Runner, wire_bytes
from .state import snapshot, restore, compatibility
from .election import elect_leader
from .native import prepare_native_round, train_native_worker, finalize_native_round
from .aggregation import parameter_names
from .evaluation import atomic_json
from .archive import TaskArchive


class HierarchicalRunner(Runner):
    def __init__(self, data, output, config):
        if not config.server_exchange:
            raise ValueError('Explicit versioned server_exchange configuration required')
        super().__init__(data, output, config)

    def cluster_round_results(self, task, round_id):
        originals = self.unique_states()
        active_uids = self.active
        groups = tuple(originals[uid].members for uid in active_uids)
        members = tuple(sorted(cid for group in groups for cid in group))
        leaders = tuple(originals[uid].coordinator_id for uid in active_uids)
        leader = elect_leader(leaders, self.ledger, task=task, round=round_id).leader
        first = originals[active_uids[0]]
        for other in originals.values():
            ok, reason = compatibility(first, other)
            if not ok:
                raise ValueError('Cannot average independent old histories: '+reason+'; start fresh')
        common = snapshot(first)
        common.update(cluster_uid=f'hierarchy-t{task}', members=members, coordinator_id=leader)
        reference = restore(common)
        bases = {cid: self.data.base(cid) for cid in members}
        plan, allocation = prepare_native_round(reference, bases, self.data.task_classes[task])
        training = dict(seed=self.config.seed+task*100000+round_id*1000,
            batch_size=self.config.batch_size, epochs=self.config.local_epochs,
            lr=self.config.learning_rate, engineering_baseline=True)
        if self.config.worker_devices:
            from .multi_gpu import GpuWorkerPool
            if self._gpu_pool is None:
                self._gpu_pool=GpuWorkerPool(self.config.worker_devices,self.output/'current_data_cache')
            uid=reference.cluster_uid
            outputs,timing=self._gpu_pool.train_round({uid:(plan,allocation,bases)}, self.data,
                self.output/'current_data_cache'/f'gpu_jobs_task_{task}_round_{round_id}',training,
                budget_guard=TaskArchive(self.output,task,self.config.output_budget_gib).guard)
            self._parallel_timing=timing
            pairs=[outputs[uid][cid] for cid in members]
        else:
            pairs=[]
            for cid in members:
                print(f'Hierarchy T{task} R{round_id} local client={cid}',flush=True)
                pairs.append(train_native_worker(plan,bases[cid],device=self.config.device,**training))
        packets=[p for p,_ in pairs]; workers=[r for _,r in pairs]
        result,plan,packets,receipt=finalize_native_round(plan,bases,packets,workers,allocation,
            batch_size=self.config.batch_size,device=self.config.device,verify_broadcast=False,groups=groups)
        # Runner commits real training receipts and learned labels once, then
        # this generator restores the original cluster identities before save.
        yield reference.cluster_uid,bases,result,plan,packets,receipt
        final=self.states[members[0]]
        common=snapshot(final)
        for uid,old in originals.items():
            payload=dict(common)
            payload.update(cluster_uid=uid,members=old.members,coordinator_id=old.coordinator_id,
                parent_lineage=old.parent_lineage)
            server=restore(payload)
            for cid in old.members:
                self.states[cid]=server
        self.active=active_uids
        local_leader={cid:originals[uid].coordinator_id for uid in active_uids for cid in originals[uid].members}
        # Binary sufficient-statistic payload: float64 numerator + denominator.
        stat_bytes=sum(final.model_state[n].numel()*16 for n in parameter_names(final))
        previous=self.communication[-1]
        receipt_bytes=wire_bytes([asdict(p.event) for p in packets])
        self.communication[-1]=dict(task=task,round=round_id,phase='hierarchical_round',
            exchange_coordinator=leader,temporary_servers=leaders,
            model_delta_upload_bytes=sum(wire_bytes(asdict(p)) for p in packets
                if p.event.client_id!=local_leader[p.event.client_id]),
            reference_broadcast_bytes=wire_bytes(snapshot(plan.reference))*sum(len(g)-1 for g in groups),
            server_statistic_upload_bytes=stat_bytes*(len(groups)-1),
            server_receipt_upper_bound_bytes=receipt_bytes*(len(groups)-1),
            server_reference_download_bytes=wire_bytes(common)*(len(originals)-1),
            bn_moment_upload_bytes=previous['bn_moment_upload_bytes'],
            bn_reference_download_bytes=previous['bn_reference_download_bytes'],
            final_state_broadcast_bytes=0,raw_training_rows_shared=False,
            accounting='modeled binary sums; typed JSON models/receipts; BN client-to-reducer upper bound')
        registered=tuple(final.algorithm_state['seen_classes'])
        expected=tuple(sorted({c for t in range(task+1) for c in self.data.task_classes[t]}))
        missing=sorted(set(expected)-set(registered))
        atomic_json(self.output/f'coverage_task_{task}_round_{round_id:02d}.json',dict(
            task=task,round=round_id,registered_classes=registered,expected_classes=expected,
            missing_classes=missing,all_servers_share_class_knowledge=True,
            predictor='routerless_v1',raw_training_rows_shared=False))
        if missing:
            raise ValueError(f'TRAIN coverage incomplete; classes absent from successful updates: {missing}')
