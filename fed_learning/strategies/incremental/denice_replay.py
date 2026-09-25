"""DENICE replay: strictly local, bounded rehearsal and dark-logit regularization.

Memory belongs to a client, never to a model/capsule. Store training examples
once at task completion, after consolidation, and preserve their original
logits. No extra teacher network or cross-client raw-data exchange is needed.
"""
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReplayConfig:
    capacity: int = 0
    batch_size: int = 32
    ce_weight: float = 1.0
    logit_weight: float = 0.2
    calibration_weight: float = 0.2
    selection: str = 'priority'
    candidate_limit: int = 512

    @classmethod
    def from_dict(cls, config):
        values = {k: config.get('denice_replay_' + k, v)
                  for k, v in asdict(cls()).items()}
        for key in ('capacity', 'batch_size', 'candidate_limit'):
            value = values[key]
            if isinstance(value, bool) or int(value) != value:
                raise ValueError(f'denice_replay_{key} must be an integer.')
            values[key] = int(value)
        if values['capacity'] < 0 or values['batch_size'] <= 0 or values['candidate_limit'] <= 0:
            raise ValueError('Replay capacity must be nonnegative and batch size positive.')
        if values['selection'] not in ('priority', 'herding'):
            raise ValueError('denice_replay_selection must be priority or herding.')
        for key in ('ce_weight', 'logit_weight', 'calibration_weight'):
            values[key] = float(values[key])
            if not math.isfinite(values[key]) or values[key] < 0:
                raise ValueError(f'denice_replay_{key} must be finite and nonnegative.')
        return cls(**values)


@contextmanager
def inference_statistics(model):
    """Stable statistics with a backward-capable cuDNN recurrent forward.

    cuDNN inference RNN forwards do not save the reserve space required by
    backward. For differentiable calls only, keep recurrent modules in training
    mode but disable their internal dropout. BN and ordinary dropout remain in
    eval mode. Restore every flag even if the forward raises; restoring a flag
    after forward cannot repair a graph already created in cuDNN inference mode.
    """
    modes = [(module, module.training) for module in model.modules()]
    recurrent_dropout = []
    model.eval()
    try:
        if torch.is_grad_enabled():
            for module, _ in modes:
                if isinstance(module, (torch.nn.RNN, torch.nn.GRU, torch.nn.LSTM)):
                    recurrent_dropout.append((module, module.dropout))
                    module.training = True
                    module.dropout = 0.0
        yield
    finally:
        for module, dropout in recurrent_dropout:
            module.dropout = dropout
        for module, training in modes:
            module.training = training


class LocalReplay:
    def __init__(self, config):
        self.config = config
        self.entries = {}  # class -> CPU tensors, independent of model state
        self.completed_tasks = set()

    def __len__(self):
        return sum(len(e['y']) for e in self.entries.values())

    def stats(self):
        return {'buffer_type': 'local_balanced_dark_replay', 'has_replay': bool(len(self)),
                'capacity': self.config.capacity, 'size': len(self),
                'class_counts': {k: len(v['y']) for k, v in self.entries.items()},
                'bytes': sum(t.numel() * t.element_size() for e in self.entries.values()
                             for t in e.values())}

    @torch.no_grad()
    def commit(self, model, x, y, task_id, batch_size=128):
        """Balanced random-priority reservoir; repeated rounds never reinsert data.

        Per-class quotas shrink as classes arrive. Selection happens before
        inference, bounding temporary device memory and teacher computation.
        """
        if task_id in self.completed_tasks or not self.config.capacity:
            return
        y = y.detach().cpu().long()
        labels = sorted(set(self.entries) | set(y.tolist()))
        if not labels:
            return
        if self.config.capacity < len(labels):
            raise ValueError('Replay capacity must cover every locally observed class.')
        valid = torch.zeros(model.num_classes, dtype=torch.bool)
        valid[labels] = True
        device = next(model.parameters()).device
        base, remainder = divmod(self.config.capacity, len(labels))
        updated = {}
        with inference_statistics(model):
            for position, label in enumerate(labels):
                quota = base + (position < remainder)
                old = self.entries.get(label)
                idx = torch.nonzero(y == label, as_tuple=False).flatten()
                if self.config.selection == 'herding':
                    # Bound candidate feature extraction without adding raw
                    # memory. Historical exemplars remain eligible.
                    limit = max(quota, self.config.candidate_limit)
                    idx = idx[torch.randperm(len(idx))[:limit]]
                priority = torch.rand(len(idx))
                old_n = 0 if old is None else len(old['y'])
                all_priority = priority if old is None else torch.cat([old['priority'], priority])
                if self.config.selection == 'herding':
                    from .denice_classifier import encode_features, herding_indices
                    candidates = x[idx].detach().cpu()
                    if old is not None:
                        candidates = torch.cat([old['x'], candidates])
                    features = encode_features(model, candidates, batch_size)
                    keep = herding_indices(features, quota)
                else:
                    keep = all_priority.argsort(descending=True)[:quota]
                old_keep, new_keep = keep[keep < old_n], keep[keep >= old_n] - old_n
                parts = []
                if len(old_keep):
                    parts.append({k: v[old_keep].clone() for k, v in old.items()})
                if len(new_keep):
                    selected = idx[new_keep]
                    inputs = x[selected].detach().cpu().clone()
                    logits = torch.cat([model(inputs[i:i+batch_size].to(device)).float().cpu()
                                        for i in range(0, len(inputs), batch_size)])
                    parts.append({'x': inputs, 'y': y[selected].clone(), 'logits': logits,
                                  'valid': valid.expand(len(inputs), -1).clone(),
                                  'priority': priority[new_keep].clone()})
                updated[label] = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        self.entries = updated
        self.completed_tasks.add(int(task_id))

    def sample(self, device):
        labels = sorted(self.entries)
        # Balanced sampling with replacement; at most batch_size examples.
        selected = torch.randint(len(labels), (self.config.batch_size,))
        rows = []
        for class_idx in selected.tolist():
            entry = self.entries[labels[class_idx]]
            idx = int(torch.randint(len(entry['y']), (1,)))
            rows.append({k: v[idx] for k, v in entry.items() if k != 'priority'})
        return {k: torch.stack([r[k] for r in rows]).to(device) for k in rows[0]}

    def loss(self, model, x, y, current_classes):
        """Auxiliary loss on inference logits; mature gradient masks still apply.

        Current adapters learn to reject old inputs, while stored old logits
        constrain behavior on coordinates known at capture (never future ones).
        Historical adapters are not activated or modified by this objective.
        """
        zero = x.new_zeros(())
        if not len(self):
            return zero, {}
        cfg = self.config
        labels = sorted(set(self.entries) | set(int(c) for c in current_classes))
        support = torch.tensor(labels, device=x.device, dtype=torch.long)
        ce, dark, calibration = zero, zero, zero
        with inference_statistics(model):
            if cfg.ce_weight or cfg.logit_weight:
                batch = self.sample(x.device)
                logits = model(batch['x']).float()
                if cfg.ce_weight:
                    ce = F.cross_entropy(logits[:, support], torch.searchsorted(support, batch['y']))
                if cfg.logit_weight:
                    error = (logits - batch['logits'].float()).square()
                    valid = batch['valid']
                    dark = ((error * valid).sum(1) / valid.sum(1).clamp_min(1)).mean()
            if cfg.calibration_weight:
                logits = model(x).float()
                calibration = F.cross_entropy(logits[:, support], torch.searchsorted(support, y.long()))
        loss = cfg.ce_weight * ce + cfg.logit_weight * dark + cfg.calibration_weight * calibration
        return loss, {'replay_ce': float(ce.detach()), 'dark_mse': float(dark.detach()),
                      'calibration_ce': float(calibration.detach())}

    def state_dict(self):
        return {'version': 1, 'config': asdict(self.config),
                'completed_tasks': sorted(self.completed_tasks),
                'entries': {c: {k: v.detach().cpu().clone() for k, v in e.items()}
                            for c, e in self.entries.items()}}

    @classmethod
    def from_state(cls, config, state):
        # Old v1 checkpoints predate selection controls and imply priority.
        saved = ReplayConfig.from_dict({
            'denice_replay_' + k: v for k, v in state.get('config', {}).items()})
        if state.get('version') != 1 or saved != config:
            raise ValueError('Incompatible local replay continuation state/configuration.')
        memory = cls(config)
        memory.completed_tasks = set(state['completed_tasks'])
        memory.entries = {int(c): {k: v.detach().cpu().clone() for k, v in e.items()}
                          for c, e in state['entries'].items()}
        if len(memory) > config.capacity:
            raise ValueError('Replay continuation exceeds configured capacity.')
        return memory
