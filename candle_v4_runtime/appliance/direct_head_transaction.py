"""Isolated transaction; never mutates the runner's live endpoint.

Commit is an isolated pilot clone, not a production install or survival claim.
"""
import copy
import numpy as np
import torch
from .config import Rejected
from .state import boundary_hash, complete_hash, digest
from .direct_head_contract import architecture, clean_endpoint, VERSION
from .functional_gap_discovery import availability_shadow


def target_state(model, router, target, task):
    return dict(weight=model.fc2.weight[target].detach().cpu().clone(),
        bias=model.fc2.bias[target].detach().cpu().clone(),
        mask=model.weight_masks['fc2'][target].detach().cpu().clone(),
        bias_mask=model.bias_masks['fc2'][target].detach().cpu().clone(),
        rank=int(model.unit_ranks['fc2'][target]),
        freeze=copy.deepcopy(model.freeze_masks.get('fc2')),
        episode_classes=copy.deepcopy(router.episode_classes))


def assert_protected(reference, candidate, target, task):
    old, r0 = reference
    new, r1 = candidate
    if boundary_hash(old) != boundary_hash(new):
        raise Rejected('DIRECT_HEAD_BACKBONE_CHANGED')
    ix = [c for c in range(old.fc2.out_features) if c != target]
    for before, after in ((old.fc2.weight, new.fc2.weight), (old.fc2.bias, new.fc2.bias),
                          (old.weight_masks['fc2'], new.weight_masks['fc2']),
                          (old.bias_masks['fc2'], new.bias_masks['fc2'])):
        if not torch.equal(before[ix], after[ix]):
            raise Rejected('DIRECT_HEAD_OTHER_ROWS_CHANGED')
    from fed_learning.training.checkpoint_state import snapshot_denice_state
    s0, s1 = snapshot_denice_state(old, r0), snapshot_denice_state(new, r1)
    # Normalize only the explicitly authorized target state; all other Python
    # algorithm state (BN, CANC, adapters, learned router memory) stays exact.
    for state in (s0, s1):
        state['connection_masks']['weight_fc2'][target] = 0
        state['connection_masks']['bias_fc2'][target] = 0
        state['neuron_ages']['fc2'][target] = 0
        if len(state.get('freeze_masks', {}).get('fc2', [])):
            state['freeze_masks']['fc2'][target] = False
        state['context_detector']['episode_classes'] = copy.deepcopy(r0.episode_classes)
    if digest(s0) != digest(s1):
        raise Rejected('DIRECT_HEAD_PROTECTED_ALGORITHM_CHANGED')
    expected = copy.deepcopy(r0.episode_classes)
    expected[task] = sorted(set(expected.get(task, [])) | {target})
    if r1.episode_classes not in (r0.episode_classes, expected):
        raise Rejected('DIRECT_HEAD_UNAUTHORIZED_AVAILABILITY')


class HeadTransaction:
    def __init__(self, model, router, target, task, current_owned_target=0):
        clean_endpoint(model)
        if current_owned_target or int(model.unit_ranks['fc2'][target]) != 0:
            raise Rejected('DIRECT_HEAD_RECEIVER_SLOT_NOT_FREE')
        freeze = model.freeze_masks.get('fc2', [])
        # Fixed allocation freezes rank-0 reserve slots for ordinary SGD
        # (ranks != 1), not because those slots contain mature knowledge.
        # The explicit direct-head permission is limited to unowned rank 0;
        # a task-wide freeze or a genuine protected slot cannot be bypassed.
        reserved_gradient_freeze = bool(getattr(model, 'fixed_task_allocation', False)
            and 'fc2' not in getattr(model, 'task_freeze_layers', []))
        if len(freeze) and bool(freeze[target]) and not reserved_gradient_freeze:
            raise Rejected('DIRECT_HEAD_RECEIVER_SLOT_FROZEN')
        self.reference = (model, router)
        self.before = complete_hash(model, router)
        self.backup = copy.deepcopy(self.reference)
        self.target, self.task = target, task
        self.state, self.candidate, self.receipt = 'IDLE', None, None

    def stage(self, agg, eta, mask_only=False):
        if not np.isfinite(eta) or not 0 <= eta <= 1:
            raise Rejected('DIRECT_HEAD_ETA_INVALID')
        model, router = self.reference
        if agg['metadata']['architecture'] != architecture(model) or agg['metadata']['class_id'] != self.target or agg['metadata']['task'] != self.task:
            raise Rejected('DIRECT_HEAD_RECEIVER_ARCHITECTURE_OR_TARGET')
        if complete_hash(*self.reference) != self.before:
            raise Rejected('DIRECT_HEAD_STALE_REFERENCE')
        if eta == 0:
            self.candidate = copy.deepcopy(self.reference)
            self.state = 'STAGED'
            return self.candidate
        candidate, detector = availability_shadow(model, router, self.target, self.task)
        c = self.target
        allowed = torch.as_tensor(agg['contributed'], device=model.fc2.weight.device, dtype=torch.bool)
        if not allowed.any() and not agg['bias_contributed']:
            raise Rejected('DIRECT_HEAD_NO_APPROVED_CONTRIBUTION')
        with torch.no_grad():
            if not mask_only:
                old = model.fc2.weight[c]*model.weight_masks['fc2'][c]
                incoming = torch.as_tensor(agg['weight'], device=old.device, dtype=old.dtype)
                candidate.fc2.weight[c, allowed] = ((1-eta)*old + eta*incoming)[allowed]
            candidate.weight_masks['fc2'][c, allowed] = 1
            if agg['bias_contributed']:
                if not mask_only:
                    old_b = model.fc2.bias[c]*model.bias_masks['fc2'][c]
                    candidate.fc2.bias[c] = (1-eta)*old_b + eta*agg['bias']
                candidate.bias_masks['fc2'][c] = 1
        candidate.unit_ranks['fc2'][c] = 2
        if len(candidate.freeze_masks.get('fc2', [])):
            candidate.freeze_masks['fc2'][c] = True
        assert_protected(self.reference, (candidate, detector), c, self.task)
        if complete_hash(*self.reference) != self.before:
            raise Rejected('DIRECT_HEAD_STAGING_MUTATED_LIVE')
        self.candidate, self.state = (candidate, detector), 'STAGED'
        return self.candidate

    def verify(self, receipt):
        if self.state != 'STAGED' or receipt.get('candidate_function') != complete_hash(*self.candidate):
            raise Rejected('DIRECT_HEAD_RECEIPT_FUNCTION_MISMATCH')
        if receipt.get('status') != 'EMPIRICAL_CURRENT_PASS':
            raise Rejected('DIRECT_HEAD_NOT_ACCEPTED')
        self.receipt, self.state = copy.deepcopy(receipt), 'VERIFIED'

    def commit(self):
        if self.state != 'VERIFIED' or complete_hash(*self.reference) != self.before:
            raise Rejected('DIRECT_HEAD_COMMIT_NOT_VERIFIED_OR_STALE')
        self.state = 'COMMITTED'
        return copy.deepcopy(self.candidate), dict(protocol=VERSION,
            receipt=self.receipt, reference_function=self.before,
            scope='isolated pilot; native protection/checkpoint integration pending')

    def rollback(self):
        self.candidate = copy.deepcopy(self.backup)
        self.receipt, self.state = None, 'ROLLED_BACK'
        if complete_hash(*self.candidate) != self.before or complete_hash(*self.reference) != self.before:
            raise Rejected('DIRECT_HEAD_ROLLBACK_MISMATCH')
        return self.candidate
