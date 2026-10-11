"""P0 canonical-state operations. No weight averaging or router fitting here."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import uuid

from .contracts import ClusterState, array, decode, encode, state_hash
from .election import elect_leader
from .knowledge import ClassKnowledgeLedger
from .partition import PartitionBook


COMPATIBILITY_FIELDS = [('versions', 'VERSION_MISMATCH'), ('model_spec', 'MODEL_SCHEMA_MISMATCH'),
    ('routing_space_hash', 'ROUTING_SPACE_MISMATCH'), ('model_state', 'MODEL_STATE_MISMATCH'),
    ('algorithm_state', 'ALGORITHM_STATE_MISMATCH'), ('bn_state', 'BN_STATE_MISMATCH'),
    ('protected_parameters', 'PROTECTION_MISMATCH'), ('router_state', 'ROUTER_HISTORY_MISMATCH')]


def _signature(state):
    return ({field: state_hash(getattr(state, field)) for field, _ in COMPATIBILITY_FIELDS},
            {field: {k: state_hash(v) for k, v in getattr(state, field).items()}
             for field in ('ownership_metadata', 'training_rng_state')})


def _compare_signatures(left, right):
    for field, reason in COMPATIBILITY_FIELDS:
        if left[0][field] != right[0][field]:
            return False, reason
    for field in ('ownership_metadata', 'training_rng_state'):
        a, b = left[1][field], right[1][field]
        if any(a[k] != b[k] for k in a.keys() & b.keys()):
            return False, field.upper() + '_CONFLICT'
    return True, 'EXACT_COMPATIBLE_STATE'


def snapshot(state):
    state.validate()
    return decode(encode(state.payload()))


def restore(payload):
    state = ClusterState(**decode(encode(payload)))
    state.validate()
    return state


def broadcast(state):
    """Independent working copies; receiving weights never adds local ownership."""
    payload = snapshot(state)
    return {cid: restore(payload) for cid in state.members}


def compatibility(left, right):
    """Conservative P0 gate: equality of function AND algorithm/history state.

    Equal dimensions alone cannot authorize merge. P1/P2 may introduce a
    stronger verified handoff later; P0 never averages independent encoders.
    """
    left.validate()
    right.validate()
    return _compare_signatures(_signature(left), _signature(right))


def merge_gate(client_states):
    """Partition callback checks all distinct parent states of a proposed group."""
    # Hash each canonical parent once, never hash a full model for each group pair.
    parents, client_parent = {}, {}
    for cid, state in client_states.items():
        if cid not in state.members:
            raise ValueError('Client is not a member of its declared parent state')
        previous = parents.get(state.cluster_uid)
        if previous is not None and previous is not state and previous.fingerprint() != state.fingerprint():
            raise ValueError('Same parent UID has divergent state')
        parents[state.cluster_uid] = state
        client_parent[cid] = state.cluster_uid
    signatures = {uid: _signature(restore(snapshot(state))) for uid, state in parents.items()}
    cached_pairs = {}

    def check(left, right):
        uids = sorted({client_parent[cid] for cid in left + right})
        for i, uid in enumerate(uids):
            for other in uids[i + 1:]:
                key = (uid, other)
                if key not in cached_pairs:
                    cached_pairs[key] = _compare_signatures(signatures[uid], signatures[other])
                allowed, reason = cached_pairs[key]
                if not allowed:
                    return allowed, reason
        return True, 'EXACT_COMPATIBLE_STATE'
    return check


def _union_metadata(parents, field):
    result = {}
    for parent in parents:
        for key, value in getattr(parent, field).items():
            if key in result and state_hash(result[key]) != state_hash(value):
                raise ValueError(f'Conflicting {field}: {key}')
            result[key] = decode(encode(value))
    return result


def handoff(parents, *, members, cluster_uid, task, ledger):
    """Clone/split/exact-merge into a next-task canonical state, atomically.

    Members must come from the supplied parent states. Caller supplies an
    exhaustive partition to Registry.install_task; this function changes no
    parent. Historical router memory is copied once, never concatenated.
    """
    parents = tuple(parents)
    members = tuple(sorted(members))
    if not parents or len({p.cluster_uid for p in parents}) != len(parents):
        raise ValueError('Handoff requires unique parent states')
    if task != max(p.task for p in parents) + 1:
        raise ValueError('Handoff must advance to the next task')
    if not members or len(set(members)) != len(members):
        raise ValueError('Invalid destination membership')
    if not cluster_uid:
        raise ValueError('Missing destination UID')
    parent_members = [cid for parent in parents for cid in parent.members]
    if len(parent_members) != len(set(parent_members)) or not set(members) <= set(parent_members):
        raise ValueError('Overlapping parents or foreign destination members')
    if any(not set(parent.members) & set(members) for parent in parents):
        raise ValueError('Unrelated parent supplied to handoff')
    for parent in parents[1:]:
        allowed, reason = compatibility(parents[0], parent)
        if not allowed:
            raise ValueError(f'MERGE_REJECTED: {reason}')
    payload = snapshot(parents[0])
    payload.update(cluster_uid=cluster_uid, task=task, round=-1, members=members,
        coordinator_id=elect_leader(members, ledger, task=task).leader,
        parent_lineage=tuple(sorted({uid for p in parents for uid in (p.cluster_uid,) + p.parent_lineage
                                     if uid != cluster_uid})),
        ownership_metadata=_union_metadata(parents, 'ownership_metadata'),
        training_rng_state=_union_metadata(parents, 'training_rng_state'))
    return restore(payload)


def split(parent, groups, *, task, ledger):
    """groups: mapping child UID -> members, exhaustive over the parent."""
    flat = [cid for group in groups.values() for cid in group]
    if not groups or len(flat) != len(set(flat)) or sorted(flat) != list(parent.members):
        raise ValueError('Split must be disjoint and exhaustive over parent members')
    if parent.cluster_uid in groups and len(groups) != 1:
        raise ValueError('A true split requires fresh child UIDs')
    return tuple(handoff([parent], members=groups[uid], cluster_uid=uid, task=task, ledger=ledger)
                 for uid in sorted(groups))


def failover(state, ledger, *, unavailable):
    result = elect_leader(state.members, ledger, task=state.task, round=state.round,
                          incumbent=state.coordinator_id, unavailable=unavailable)
    payload = snapshot(state)
    payload['coordinator_id'] = result.leader
    return restore(payload), result


def initialize_clusters(template, partition, ledger):
    """Task0 broadcast from a common, history-free initialization."""
    template.validate()
    partition.validate(template.members)
    if template.task != 0 or template.round != -1 or partition.task != 0:
        raise ValueError('Initialization requires Task0 start')
    if template.router_state['activation_memory'] or template.router_state['episode_classes'] or ledger.events:
        raise ValueError('Fresh initialization cannot inherit learned history')
    if template.ownership_metadata or any(array(rank).any() for rank in template.algorithm_state['unit_ranks'].values()):
        raise ValueError('Fresh initialization requires untrained ranks/ownership')
    if partition.reference_hash != template.fingerprint():
        raise ValueError('Task0 partition reference does not match initialization')
    result = []
    for members in partition.clusters:
        payload = snapshot(template)
        payload.update(cluster_uid='task0-' + '-'.join(map(str, members)), members=members,
            coordinator_id=elect_leader(members, ledger, task=0).leader, parent_lineage=())
        result.append(restore(payload))
    return tuple(result)


def transition_clusters(partition, client_states, ledger):
    """Explicit parent map -> next-task states; no hidden catch-up/reset policy.

    Dormant/rejoining clients must be assigned an existing authorized parent by
    the caller. Rejected compatibility leaves all supplied parents untouched.
    Registry installation is a separate atomic operation over the whole task.
    """
    partition.validate(client_states)
    for cid, state in client_states.items():
        state.validate()
        if cid not in state.members:
            raise ValueError('Invalid client-to-parent mapping')
    result = []
    for members in partition.clusters:
        parents = {}
        for cid in members:
            state = client_states[cid]
            previous = parents.get(state.cluster_uid)
            if previous is not None and previous.fingerprint() != state.fingerprint():
                raise ValueError('Same parent UID has divergent state')
            parents[state.cluster_uid] = state
        ordered = tuple(parents[uid] for uid in sorted(parents))
        unchanged = len(ordered) == 1 and members == ordered[0].members
        uid = ordered[0].cluster_uid if unchanged else (
            f'task{partition.task}-' + state_hash((tuple(parents), members))[:20])
        result.append(handoff(ordered, members=members, cluster_uid=uid, task=partition.task, ledger=ledger))
    return tuple(result)


class ClusterRegistry:
    """Canonical current partition plus preserved dormant states, not a server."""
    def __init__(self):
        self.states = {}
        self.active_uids = ()
        self.active_clients = ()
        self.task = -1

    def install_task(self, states, *, active_clients):
        states = tuple(states)
        active_clients = tuple(sorted(active_clients))
        clones = tuple(restore(snapshot(s)) for s in states)
        flat = [cid for s in clones for cid in s.members]
        if not clones or len({s.cluster_uid for s in clones}) != len(clones):
            raise ValueError('Missing/duplicate canonical cluster state')
        if len(flat) != len(set(flat)) or sorted(flat) != list(active_clients):
            raise ValueError('Active cluster membership must be disjoint and exhaustive')
        task = clones[0].task
        if any(s.task != task or s.round != clones[0].round for s in clones):
            raise ValueError('Current cluster states have different clocks')
        if task < self.task or (self.task >= 0 and task > self.task + 1):
            raise ValueError('Invalid registry task transition')
        if task == self.task:
            previous = {uid: self.states[uid].members for uid in self.active_uids}
            if previous != {s.cluster_uid: s.members for s in clones}:
                raise ValueError('Membership/UID is locked within task')
            if any(s.round < self.states[s.cluster_uid].round for s in clones):
                raise ValueError('Cannot rewind canonical round')
        # No mutation until every validation has passed; dormant parents are kept.
        self.states.update({s.cluster_uid: s for s in clones})
        self.active_uids = tuple(sorted(s.cluster_uid for s in clones))
        self.active_clients, self.task = active_clients, task

    def snapshot(self):
        return dict(task=self.task, active_clients=self.active_clients, active_uids=self.active_uids,
                    states={uid: snapshot(self.states[uid]) for uid in sorted(self.states)})

    @classmethod
    def restore(cls, payload):
        registry = cls()
        states = {uid: restore(s) for uid, s in payload['states'].items()}
        if any(uid != s.cluster_uid for uid, s in states.items()):
            raise ValueError('Registry UID mismatch')
        active = tuple(payload['active_uids'])
        if len(set(active)) != len(active) or any(uid not in states for uid in active):
            raise ValueError('Invalid active registry UIDs')
        registry.install_task([states[uid] for uid in active], active_clients=payload['active_clients'])
        if registry.task != payload['task'] or any(s.task > registry.task for s in states.values()):
            raise ValueError('Invalid registry/dormant clock')
        registry.states = states
        return registry


@dataclass
class P0Checkpoint:
    registry: ClusterRegistry
    ledger: ClassKnowledgeLedger
    partitions: PartitionBook

    def payload(self):
        # Revalidate even if a caller has mutated a public state buffer since install.
        ClusterRegistry.restore(self.registry.snapshot())
        active_states = [self.registry.states[uid] for uid in self.registry.active_uids]
        if not active_states:
            raise ValueError('Cannot checkpoint empty active registry')
        task, round = self.registry.task, active_states[0].round
        partition = self.partitions.partitions.get(task)
        if partition is None or partition.clusters != tuple(sorted(s.members for s in active_states)):
            raise ValueError('Canonical states differ from locked partition')
        if set(self.partitions.partitions) - set(range(task + 1)):
            raise ValueError('Future partition in checkpoint')
        if any((e.task, e.round) > (task, round) for e in self.ledger.events.values()):
            raise ValueError('Future local-learning evidence in checkpoint')
        if any(tuple(s.versions['class_ids']) != tuple(range(self.ledger.num_classes)) for s in self.registry.states.values()):
            raise ValueError('Ledger/model class mapping mismatch')
        return dict(schema_version=1, registry=self.registry.snapshot(),
                    ledger=self.ledger.snapshot(task=task, round=round), partitions=self.partitions.snapshot())


def save_checkpoint(path, checkpoint):
    """Lossless typed JSON, atomic replacement, integrity hash; no pickle."""
    payload = checkpoint.payload()
    envelope = dict(payload=encode(payload), sha256=state_hash(payload))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(envelope, stream, separators=(',', ':'), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return envelope['sha256']


def load_checkpoint(path):
    with Path(path).open(encoding='utf-8') as stream:
        envelope = json.load(stream)
    payload = decode(envelope['payload'])
    if envelope['sha256'] != state_hash(payload) or payload.get('schema_version') != 1:
        raise ValueError('Checkpoint integrity/schema mismatch')
    result = P0Checkpoint(ClusterRegistry.restore(payload['registry']),
        ClassKnowledgeLedger.restore(payload['ledger']), PartitionBook.restore(payload['partitions']))
    result.payload()  # Cross-validate partition, population, clocks and ledger after load.
    return result
