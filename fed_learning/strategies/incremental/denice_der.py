"""Trajectory-logit DER / DER++ with per-client uniform stream reservoir."""
from collections import Counter
import torch
import torch.nn.functional as F
from .denice_replay import inference_statistics


class DERReplay:
    def __init__(self, config, controls):
        self.config, self.controls = config, dict(controls)
        self.rows, self.seen = [], 0
        self.completed_tasks = set()

    def __len__(self):
        return len(self.rows)

    @property
    def entries(self):
        # Compatibility view for optional local router replay. Reservoir itself
        # is global within this client, with no per-class quota.
        result = {}
        for cls in sorted(set(int(row['y']) for row in self.rows)):
            rows = [r for r in self.rows if int(r['y']) == cls]
            result[cls] = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
        return result

    @torch.no_grad()
    def observe(self, x, y, logits, valid):
        # Algorithm R: kth arrival is retained with probability capacity/k.
        # Every presentation (including repeated local epochs) is an arrival.
        replacements = {}
        occupied = len(self.rows)
        for i in range(len(y)):
            self.seen += 1
            index = occupied if occupied < self.config.capacity else int(torch.randint(self.seen, (1,)))
            if index >= self.config.capacity:
                continue
            occupied = max(occupied, index + 1)
            replacements[index] = i
        if not replacements:
            return
        # Transfer only final survivors once per tensor, rather than synchronizing
        # CUDA separately for every stream item or overwritten replacement.
        slots = sorted(replacements)
        indices = torch.tensor([replacements[slot] for slot in slots], device=x.device)
        xs, ys = x[indices].detach().cpu(), y[indices.to(y.device)].detach().cpu().long()
        zs, mask = logits[indices.to(logits.device)].detach().cpu().float(), valid.detach().cpu().bool()
        for i, index in enumerate(slots):
            row = dict(x=xs[i].clone(), y=ys[i].clone(), logits=zs[i].clone(), valid=mask.clone())
            if index == len(self.rows):
                self.rows.append(row)
            else:
                self.rows[index] = row

    def sample(self, device):
        idx = torch.randperm(len(self.rows))[:min(self.config.batch_size, len(self.rows))]
        return {k: torch.stack([self.rows[int(i)][k] for i in idx]).to(device) for k in self.rows[0]}

    def loss(self, model, x, y, current_classes):
        zero = x.new_zeros((), dtype=torch.float32)
        if not self.rows:
            return zero, {}
        alpha = self.controls['alpha']
        beta = self.controls['beta'] if self.controls['method'] == 'derpp' else 0.
        dark, ce = zero, zero
        with inference_statistics(model):
            if alpha:
                batch = self.sample(x.device)
                error = (model(batch['x']).float() - batch['logits']).square() * batch['valid']
                value = error.sum(1)
                if self.controls['reduction'] == 'mean':
                    value = value / batch['valid'].sum(1).clamp_min(1)
                dark = value.mean()
            if beta:
                batch = self.sample(x.device)  # Independent draw, Algorithm 2.
                logits = model(batch['x']).float()
                if self.controls['logit_scope'] == 'seen':
                    mask = torch.as_tensor(model.unit_ranks['fc2'] > 0, device=x.device)
                    logits = logits.masked_fill(~mask, -1e4)
                ce = F.cross_entropy(logits, batch['y'])
        return alpha * dark + beta * ce, dict(replay_ce=float(ce.detach()), dark_mse=float(dark.detach()), calibration_ce=0.)

    def commit(self, model, x, y, task_id, batch_size=128):
        self.completed_tasks.add(int(task_id))  # No task-boundary overwrite of targets.

    def stats(self):
        return dict(buffer_type='stream_reservoir_'+self.controls['method'], has_replay=bool(self.rows),
                    capacity=self.config.capacity, size=len(self), stream_seen=self.seen,
                    class_counts=dict(Counter(int(row['y']) for row in self.rows)))

    def state_dict(self):
        return dict(version='der_stream_v1', controls=self.controls, capacity=self.config.capacity,
                    batch_size=self.config.batch_size, rows=self.rows, seen=self.seen,
                    completed_tasks=sorted(self.completed_tasks))

    @classmethod
    def from_state(cls, config, controls, state):
        if (state.get('version') != 'der_stream_v1' or state['controls'] != controls
                or state['capacity'] != config.capacity or state['batch_size'] != config.batch_size):
            raise ValueError('Incompatible DER continuation')
        obj = cls(config, controls)
        obj.rows = [{k: v.detach().cpu().clone() for k, v in r.items()} for r in state['rows']]
        obj.seen, obj.completed_tasks = int(state['seen']), set(state['completed_tasks'])
        if len(obj) > config.capacity or obj.seen < len(obj):
            raise ValueError('Invalid DER reservoir counters')
        return obj
