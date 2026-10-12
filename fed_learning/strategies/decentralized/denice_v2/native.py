"""P1 native workers and sequential BASE BN collection.

Opt-in integration with legacy DeNICEClient/Trainer. This is not a P2 router
builder. P3 orchestrates these workers; one phase per round, FP32.
"""
from dataclasses import dataclass

import numpy as np
import torch

from fed_learning.models.denice_model import DeNICEModel
from fed_learning.clients.denice_client import DeNICEClient
from fed_learning.strategies.incremental.denice import DeNICETrainer
from .aggregation import aggregate_round, finish_worker, prepare_round, start_worker
from .contracts import ClusterState, array, decode, encode, state_hash
from .knowledge import LocalLearningEvent
from .normalization import BNMoment, model_hash, protected_bn_channels
from .state import restore, snapshot


NATIVE_FIELDS = ('structural_protection', 'fixed_task_allocation', 'allocation_policy',
    'capacity_per_class', 'task_freeze_layers', 'gru_connection_masks')


def capture_native(model, *, members, coordinator_id, cluster_uid, task=0, round=-1,
                   router_state, routing_dependencies, protected_parameters=None):
    """Capture a plain legacy network; unsupported branches fail explicitly."""
    if (model.adapters or model.continual_head is not None or model.adapter_mode != 'legacy_output'
        or model.adapter_registry or model.active_adapters or model.recycling_registry
        or model.candle_state or model.elastic_state or model.adapter_input_masks or model.adapter_output_masks):
        raise ValueError('P1 native integration supports plain legacy networks only')
    tensors = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    protected = decode(encode(protected_parameters or {}))
    for name, value in tensors.items():
        layer = name.split('.')[0]
        if layer not in model.unit_ranks:
            continue
        mature = model.unit_ranks[layer] >= 2
        if layer == 'gru':
            mature = np.tile(mature, 3)
        mask = np.broadcast_to(mature.reshape((-1,) + (1,) * (value.ndim - 1)), value.shape)
        protected[name] = torch.as_tensor(mask.copy()) | torch.as_tensor(protected.get(name, np.zeros(value.shape, bool)))
    algorithms = {key: decode(encode(getattr(model, key)))
                  for key in ('unit_ranks', 'weight_masks', 'bias_masks', 'freeze_masks', *NATIVE_FIELDS)}
    for layer, ranks in algorithms['unit_ranks'].items():
        algorithms['freeze_masks'][layer] = np.asarray(algorithms['freeze_masks'].get(layer, ranks >= 2)) | (ranks >= 2)
    algorithms['native_input_shape'] = tuple(model.input_shape)
    algorithms['native_optimizer_policy'] = 'native_phase_reset_fp32_masked_projection_v1'
    routerless = router_state.get('router_mode') == 'routerless_v1'
    if routerless:
        algorithms.update(seen_classes=(),learned_classes_by_task={},classification_objective='cumulative_registered_ce_v1')
    return ClusterState.create(cluster_uid=cluster_uid, task=task, round=round,
        members=tuple(sorted(members)), coordinator_id=coordinator_id,
        versions=dict(architecture='denice_legacy_plain_v1', feature='routerless_v1' if routerless else 'p1_declared_dependencies_v1',
                      class_mapping='global_contiguous_v1', class_ids=tuple(range(model.num_classes))),
        model_state=tensors, algorithm_state=algorithms, router_state=router_state,
        bn_state={k: v for k, v in tensors.items() if k.endswith(('running_mean', 'running_var', 'num_batches_tracked'))},
        protected_parameters=protected, routing_dependencies=tuple(routing_dependencies))


def load_native(state, device='cpu'):
    state.validate()
    if state.versions['architecture'] != 'denice_legacy_plain_v1':
        raise ValueError('Unsupported native architecture')
    # Construction must not consume the worker's dropout/shuffle RNG stream.
    with torch.random.fork_rng(devices=[]):
        model = DeNICEModel(tuple(state.algorithm_state['native_input_shape']), len(state.versions['class_ids']))
    model.load_state_dict(state.model_state, strict=True)
    for key in ('unit_ranks', 'weight_masks', 'bias_masks', 'freeze_masks', *NATIVE_FIELDS):
        setattr(model, key, decode(encode(state.algorithm_state[key])))
    model.to(device)
    model._move_masks_to_device(device)
    model.gru_connection_masks = {k: torch.as_tensor(v, device=device) for k, v in model.gru_connection_masks.items()}
    for layer, conv in (('bn1', 'conv1'), ('bn2', 'conv2'), ('bn3', 'conv3')):
        bn = getattr(model, layer)
        model._bn_frozen_units[conv] = torch.as_tensor(protected_bn_channels(state, layer), device=device)
        model._bn_running_mean_frozen[conv] = bn.running_mean.detach().clone()
        model._bn_running_var_frozen[conv] = bn.running_var.detach().clone()
    return model


def allocate_cluster(state, current_classes):
    """Allocate once on the canonical model, never independently at workers."""
    classes = tuple(sorted(set(current_classes)))
    if not classes or any(type(c) is not int or c not in state.versions['class_ids'] for c in classes):
        raise ValueError('Invalid allocation classes')
    model = load_native(state)
    allocated = model.allocate_task_neurons(list(classes))
    rows = model.unit_ranks['fc2']
    allocated['fc2'] = sum(rows[c] == 0 for c in classes)
    rows[np.asarray([c for c in classes if rows[c] == 0], dtype=np.int64)] = 1
    payload = snapshot(state)
    payload['algorithm_state']['unit_ranks'] = model.unit_ranks
    # Weights, masks, prior ages and router metadata remain byte exact.
    candidate = ClusterState.create(**{k: v for k, v in payload.items() if k not in ('model_spec', 'routing_space_hash')})
    return candidate, {k: int(v) for k, v in allocated.items()}


def consolidate_native_task(state):
    """Age the canonical allocation once, without replay, rewiring or weights changes.

    This explicit bounded-smoke step is not a full production task scheduler.
    """
    from fed_learning.strategies.incremental.nice import increase_unit_ranks, update_freeze_masks
    routerless = state.router_state['router_mode'] == 'routerless_v1'
    completed_tasks = (state.algorithm_state.get('learned_classes_by_task', {}) if routerless
                       else state.router_state['activation_memory'])
    if state.round < 0 or state.task not in completed_tasks:
        raise ValueError('Consolidation requires a trained task and committed router memory')
    if state.algorithm_state.get('last_consolidated_task', -1) >= state.task:
        raise ValueError('Task already consolidated')
    model = load_native(state)
    increase_unit_ranks(model)
    update_freeze_masks(model)
    for layer, old_mask in state.algorithm_state['freeze_masks'].items():
        model.freeze_masks[layer] |= np.asarray(old_mask, dtype=bool)
    captured = capture_native(model, members=state.members, coordinator_id=state.coordinator_id,
        cluster_uid=state.cluster_uid, task=state.task, round=state.round, router_state=state.router_state,
        routing_dependencies=state.routing_dependencies, protected_parameters=state.protected_parameters)
    payload = snapshot(state)
    for key in ('unit_ranks', 'freeze_masks'):
        payload['algorithm_state'][key] = captured.algorithm_state[key]
    payload['algorithm_state']['last_consolidated_task'] = state.task
    payload['protected_parameters'] = captured.protected_parameters
    candidate = ClusterState.create(**{k: v for k, v in payload.items() if k not in ('model_spec', 'routing_space_hash')})
    if candidate.routing_space_hash != state.routing_space_hash:
        raise ValueError('Consolidation changed routing function')
    return candidate


@dataclass(frozen=True)
class CurrentBase:
    client_id: int
    task: int
    x: torch.Tensor
    y: torch.Tensor
    row_ids: np.ndarray
    source_hash: str
    role: str = 'base'
    allow_singleton: bool = False

    def validate(self, current_classes):
        if self.role != 'base' or not self.source_hash or type(self.task) is not int or type(self.client_id) is not int:
            raise ValueError('Invalid BASE provenance')
        if (self.x.device.type != 'cpu' or self.y.device.type != 'cpu' or self.x.dtype != torch.float32
            or self.y.dtype != torch.int64 or self.y.ndim != 1 or len(self.x) != len(self.y)
            or (len(self.y)==1 and not self.allow_singleton)
            or len(self.row_ids) != len(self.y) or np.asarray(self.row_ids).dtype.kind not in 'iu'
            or len(np.unique(self.row_ids)) != len(self.y) or not torch.isfinite(self.x).all()
            or not set(self.y.tolist()) <= set(current_classes)):
            raise ValueError('Invalid current-task BASE rows/tensors')
        return {int(c): int(n) for c, n in zip(*torch.unique(self.y, return_counts=True))}

    def fingerprint(self):
        return state_hash(dict(client=self.client_id, task=self.task, role=self.role, source=self.source_hash,
                               rows=self.row_ids, x=self.x, y=self.y,allow_singleton=self.allow_singleton))


def prepare_native_round(state, bases, current_classes):
    if set(bases) != set(state.members):
        raise ValueError('BASE providers must cover exactly the cluster')
    counts = {}
    for cid, base in bases.items():
        if cid != base.client_id or base.task != state.task:
            raise ValueError('Historical/wrong-client BASE forbidden')
        counts[cid] = base.validate(current_classes)
        shape = tuple(state.algorithm_state['native_input_shape'])
        if tuple(base.x.shape[1:]) != shape:
            raise ValueError('BASE input shape differs from canonical network')
    # Allocate the class UNION, including rows absent at some members.
    if not any(counts.values()):
        raise ValueError('No current BASE in cluster')
    allocated, receipt = allocate_cluster(state, tuple(sorted({c for row in counts.values() for c in row})))
    authority = state_hash({cid: b.fingerprint() for cid, b in bases.items()})
    return prepare_round(allocated, current_classes=current_classes, base_counts=counts, base_role_hash=authority), receipt


def _validate_base(plan, base):
    if base.task != plan.reference.task or base.client_id not in plan.reference.members:
        raise ValueError('Historical/wrong-client BASE forbidden')
    if base.validate(plan.current_classes) != plan.base_counts[base.client_id]:
        raise ValueError('BASE class counts changed')


class ClusterWorkerClient(DeNICEClient):
    """Natural native sampling, merging a singleton tail without dropping rows."""
    def _create_batches(self, batch_size):
        batches = list(torch.randperm(self.num_samples).split(batch_size))
        if self.num_samples == 1:
            # BN needs two observations: reuse this current row, never another role.
            batches = [batches[0].repeat(2)]
        if len(batches[-1]) == 1:
            tail = batches.pop()
            batches[-1] = torch.cat((batches[-1], tail))
        for indices in batches:
            yield self.X_train[indices].to(self.device), self.y_train[indices].to(self.device)


def train_native_worker(plan, base, *, seed=42, batch_size=16, epochs=1, lr=0.001,
                        optimizer_family='adam', weight_decay=0.0, momentum=0.0, device='cpu',
                        engineering_baseline=False):
    _validate_base(plan, base)
    if (type(batch_size) is not int or batch_size < 2 or type(epochs) is not int or epochs < 1
        or not np.isfinite([lr, weight_decay, momentum]).all() or lr <= 0 or weight_decay < 0 or not 0 <= momentum < 1
        or optimizer_family not in ('adam', 'sgd') or (optimizer_family == 'adam' and momentum != 0)):
        raise ValueError('Invalid native optimizer configuration')
    worker = start_worker(plan, base.client_id)
    if not len(base.y):
        event = LocalLearningEvent(f'{plan.hash}/{base.client_id}', base.client_id, plan.reference.task,
            plan.target_round, 0, (), base.task)
        packet = finish_worker(plan, worker, event)
        return packet, dict(client=base.client_id, steps=0, loss_mean=None, skipped='NO_CURRENT_BASE',
            changed_coordinates={k: 0 for k in packet.deltas}, initial_reference_hash=worker.initial_reference_hash,
            initial_topology_hash=worker.initial_topology_hash)
    cuda_devices = [torch.device(device).index or 0] if str(device).startswith('cuda') else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed + base.client_id)
        model = load_native(worker.state, device).train()
        permitted = {k: torch.as_tensor(v, device=device) for k, v in plan.parameter_permissions.items()}
        if not any(bool(mask.any()) for mask in permitted.values()):
            if engineering_baseline:
                event = LocalLearningEvent(f'{plan.hash}/{base.client_id}', base.client_id, plan.reference.task,
                    plan.target_round, 0, (), base.task)
                packet = finish_worker(plan, worker, event)
                return packet, dict(client=base.client_id, steps=0, loss_mean=None,
                    skipped='NO_TRAINABLE_CAPACITY', changed_coordinates={k: 0 for k in packet.deltas})
            raise ValueError('NO_TRAINABLE_CAPACITY')
        before = {k: p.detach().clone() for k, p in model.named_parameters()}
        seen = {}

        def gradient_filter():
            for name, p in model.named_parameters():
                if p.grad is not None:
                    p.grad.masked_fill_(~permitted[name], 0)

        def project(current_model):
            # Gradient masks alone do not stop weight decay/momentum drift.
            with torch.no_grad():
                for name, p in current_model.named_parameters():
                    p[~permitted[name]] = before[name][~permitted[name]]

        def observed(x, y, output):
            for c, n in zip(*torch.unique(y.cpu(), return_counts=True)):
                seen[int(c)] = seen.get(int(c), 0) + int(n)

        client = ClusterWorkerClient(base.client_id, base.x, base.y)
        client.setup_for_gpu(model, str(device))
        client.set_task_data(base.x, base.y, base.task, list(plan.current_classes))
        trainer = DeNICETrainer(max_phases=1, phase_epochs=epochs)
        trainer.set_task(base.task, list(plan.current_classes))
        objective = {}
        if plan.reference.algorithm_state.get('classification_objective') == 'cumulative_registered_ce_v1':
            from .routerless import CumulativeClassTrainer, training_classes, masked_logits
            active_classes = training_classes(plan)
            trainer = CumulativeClassTrainer(active_classes,max_phases=1,phase_epochs=epochs)
            trainer.set_task(base.task,list(plan.current_classes))
            objective['supervised_forward'] = lambda x: masked_logits(model(x),active_classes)
        factory = (lambda parameters, rate: torch.optim.Adam(parameters, lr=rate, weight_decay=weight_decay)) if optimizer_family == 'adam' else (
            lambda parameters, rate: torch.optim.SGD(parameters, lr=rate, momentum=momentum, weight_decay=weight_decay))
        trained = client.train(trainer, epochs=epochs, batch_size=batch_size, lr=lr,
            cluster_layout_locked=True, amp_enabled=False, gradient_filter=gradient_filter,
            after_optimizer_step_update=project, after_optimizer_step=observed, optimizer_factory=factory, **objective)
        successful = trained['optimizer_steps']
        native_topology = {key: getattr(model, key) for key in ('unit_ranks', 'weight_masks', 'bias_masks', 'freeze_masks', *NATIVE_FIELDS)}
        expected_topology = {key: worker.state.algorithm_state[key] for key in native_topology}
        if state_hash(native_topology) != state_hash(expected_topology):
            raise ValueError('Native worker changed canonical allocation/masks')
        worker.state.model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        worker.state.bn_state = {k: worker.state.model_state[k].clone() for k in worker.state.bn_state}
    event = LocalLearningEvent(f'{plan.hash}/{base.client_id}', base.client_id, plan.reference.task,
        plan.target_round, successful, tuple(sorted(seen.items())), base.task)
    packet = finish_worker(plan, worker, event)
    changed = {name: int(np.count_nonzero(packet.deltas[name])) for name in packet.deltas}
    if not any(changed.values()) and not engineering_baseline:
        raise ValueError('Optimizer stepped but no contributed parameter changed')
    return packet, dict(client=base.client_id, steps=successful, loss_mean=float(trained['loss']),
                       classification_objective=('cumulative_registered_ce_v1' if objective else 'native_let_learner_ce'),
                       softmax_classes=active_classes if objective else None,
                       native_training_loop='DeNICEClient/DeNICETrainer', optimizer_family=optimizer_family,
                       changed_coordinates=changed, initial_reference_hash=worker.initial_reference_hash,
                       zero_parameter_update=not any(changed.values()),
                       singleton_current_row_reused_for_bn=len(base.y)==1,
                       initial_topology_hash=worker.initial_topology_hash)


class NativeBNCollector:
    """Pre-BN hooks on each sequential canonical post-aggregation model."""
    def __init__(self, plan, bases, *, batch_size=16, device='cpu'):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('Invalid BN batch size')
        if state_hash({cid: b.fingerprint() for cid, b in bases.items()}) != plan.base_role_hash:
            raise ValueError('BASE content differs from round authority')
        for base in bases.values():
            _validate_base(plan, base)
        self.plan, self.bases, self.batch_size, self.device = plan, bases, batch_size, device
        self.calls = []

    def __call__(self, stage, layer):
        stage.validate()
        if state_hash({cid: b.fingerprint() for cid, b in self.bases.items()}) != self.plan.base_role_hash:
            raise ValueError('BASE mutated during round')
        model = load_native(stage, self.device).eval()
        packets = []
        for cid, base in sorted(self.bases.items()):
            if not len(base.y):
                continue
            width = getattr(model, layer).num_features
            total, squared = np.zeros(width, np.float64), np.zeros(width, np.float64)
            observations = 0

            def hook(module, inputs):
                nonlocal observations
                z = inputs[0].detach().to('cpu', dtype=torch.float64).movedim(1, 0).reshape(width, -1)
                total[:] += z.sum(1).numpy()
                squared[:] += (z * z).sum(1).numpy()
                observations += z.shape[1]

            handle = getattr(model, layer).register_forward_pre_hook(hook)
            try:
                with torch.no_grad():
                    for x in base.x.split(self.batch_size):
                        model(x.to(self.device))
            finally:
                handle.remove()
            packets.append(BNMoment(cid, stage.task, self.plan.target_round, layer, model_hash(stage),
                self.plan.hash, len(base.y), np.full(width, observations, np.int64), total, squared, base.task))
        import json
        self.calls.append(dict(layer=layer, input_model_hash=model_hash(stage), members=tuple(sorted(self.bases)),
            wire_bytes_by_client={p.client_id:len(json.dumps(encode(p.__dict__),separators=(',',':')).encode()) for p in packets}))
        return packets


def run_native_round(state, bases, current_classes, *, log_progress=False, **training):
    """Canonical allocation -> real optimizers -> BN -> identical broadcast."""
    plan, allocation = prepare_native_round(state, bases, current_classes)
    packets, workers = [], []
    for cid in plan.reference.members:
        if log_progress:
            print(f'Native local T{state.task} R{plan.target_round} cluster={state.cluster_uid} client={cid} BASE={len(bases[cid].y):,}',flush=True)
        packet, receipt = train_native_worker(plan, bases[cid], **training)
        packets.append(packet)
        workers.append(receipt)
        if log_progress:
            print(f'  optimizer steps={receipt["steps"]} loss={receipt["loss_mean"]} skip={receipt.get("skipped")}',flush=True)
    collector = NativeBNCollector(plan, bases, batch_size=training.get('batch_size', 16), device=training.get('device', 'cpu'))
    result = aggregate_round(plan, packets, bn_provider=collector)
    copies = result.broadcast()
    witness = torch.cat([base.x[:4] for base in bases.values()])
    logits = []
    for cid in sorted(copies):
        model = load_native(copies[cid], training.get('device', 'cpu')).eval()
        with torch.no_grad():
            logits.append(model(witness.to(next(model.parameters()).device)).cpu())
    same_state = len({s.fingerprint() for s in copies.values()}) == 1
    same_logits = all(torch.equal(logits[0], value) for value in logits[1:])
    if not same_state or not same_logits:
        raise ValueError('Broadcast behavior mismatch')
    receipt = dict(allocation=allocation, workers=workers, aggregation=result.receipt,
        bn_collector_calls=collector.calls, broadcast_state_identical=same_state,
        broadcast_logits_identical=same_logits, prediction_witness_role='current_BASE',
        prediction_witness_rows=len(witness), router_fitted=False, test_read=False)
    return result, plan, packets, receipt
