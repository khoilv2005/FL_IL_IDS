"""Opt-in training-time peer prototypes for personalized CANDLE V2.

One-round delayed, current-task prototypes; no raw examples cross clients.
Task-free cosine FC2 is shared by training/evaluation. V1 defaults are untouched.
"""
import copy
import json
import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

VERSION = 'candle_v2_training_prototypes_v1'
CONTROL_KEYS = ('candle_v2_align_weight', 'candle_v2_virtual_weight',
                'candle_v2_preservation_weight', 'candle_v2_logit_scale',
                'candle_v2_proto_per_class', 'candle_v2_proto_batch_size')


def validate(config):
    if not config.get('candle_v2_enabled', False):
        return
    if config.get('denice_cl_method', 'legacy') != 'legacy':
        raise ValueError('CANDLE V2 uses legacy NICE protection, without CGoFed/DER/EWC')
    for key in ('appliance_enabled', 'denice_appliance_enabled', 'denice_cme_after_each_task'):
        if config.get(key):
            raise ValueError(f'CANDLE V2 cannot combine with {key}')
    if config.get('denice_replay_capacity', 0) or config.get('denice_adapter_layers'):
        raise ValueError('CANDLE V2 task-free model requires no raw replay or context-specific adapters')
    for key in CONTROL_KEYS:
        value = config[key]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f'Invalid {key}')
    for key in ('candle_v2_logit_scale', 'candle_v2_proto_per_class', 'candle_v2_proto_batch_size'):
        if float(config[key]) <= 0:
            raise ValueError(f'{key} must be positive')
    if float(config['candle_v2_virtual_weight']) <= 0:
        raise ValueError('Main CANDLE V2 needs virtual prototype CE to learn imported rows')


def state(model, cid=None):
    value = getattr(model, 'candle_v2_state', None)
    if value is None:
        value = dict(version=VERSION, client_id=cid, task=-1, bank_round=-1,
                     bank={}, available=[], class_tasks={}, owned_history=[], imported={})
        model.candle_v2_state = value
    if value['version'] != VERSION:
        raise ValueError('Unsupported CANDLE V2 checkpoint state')
    if cid is not None and value.get('client_id') != int(cid):
        # Native bootstrap may clone a trained peer; its knowledge is inherited,
        # but its data ownership must never become the joining receiver's ownership.
        value.update(client_id=int(cid), inherited_from=value.get('client_id'),
                     owned_history=[], bank={}, bank_round=-1, imported={})
    return value


def features(model, x):
    h = model.penultimate_features(x)
    active = torch.as_tensor(np.asarray(model.unit_ranks['fc1']) > 0, device=h.device)
    return h * active.to(h.dtype)


def logits(model, h, allowed, scale):
    allowed = sorted(set(map(int, allowed)))
    if not allowed:
        raise ValueError('Empty CANDLE V2 class availability')
    weight = model.fc2.weight * model.weight_masks['fc2'].to(model.fc2.weight)
    active = torch.as_tensor(np.asarray(model.unit_ranks['fc1']) > 0, device=weight.device)
    weight = weight * active.to(weight.dtype)[None, :]
    h = h * active.to(h.dtype)
    output = float(scale) * F.linear(F.normalize(h.float(), dim=1, eps=1e-6),
                                    F.normalize(weight.float(), dim=1, eps=1e-6))
    valid = torch.zeros(output.shape[1], dtype=torch.bool, device=output.device)
    valid[allowed] = True
    return output.masked_fill(~valid, -1e4)


@contextmanager
def evaluation_mode(model):
    flags = [(m, m.training) for m in model.modules()]
    active = dict(model.active_adapters)
    try:
        model.eval()
        model.clear_active_adapters()
        yield
    finally:
        model.active_adapters = active
        for module, training in flags:
            module.training = training


@torch.no_grad()
def predict_logits(model, x, scale):
    value = state(model)
    if not value['available']:
        raise ValueError('Checkpoint has no successfully trained CANDLE V2 classes')
    return logits(model, features(model, x), value['available'], scale)


class CANDLEV2Service:
    def __init__(self, config, saved_config=None):
        validate(config)
        self.config = config
        self.live = {}
        self.protected_rows = {}
        if saved_config is not None:
            if not saved_config.get('candle_v2_enabled'):
                raise ValueError('Fresh training required: cannot resume V1 as CANDLE V2')
            for key in CONTROL_KEYS:
                if config[key] != saved_config.get(key):
                    raise ValueError(f'CANDLE V2 resume control changed: {key}')

    def training_hooks(self, cid, model, client, task, round_id, task_classes):
        value = state(model, cid)
        local = sorted(map(int, torch.unique(client.y_train).tolist()))
        if value['task'] != int(task):
            value.update(task=int(task), bank={}, bank_round=-1)
        value['owned_history'] = sorted(set(value['owned_history']) | set(local))
        candidate_bank = value['bank'] if value['bank_round'] == int(round_id) - 1 else {}
        active_fc1 = torch.as_tensor(np.asarray(model.unit_ranks['fc1']) > 0,
                                     dtype=model.fc2.weight.dtype, device=model.fc2.weight.device)
        active_cpu = active_fc1.detach().cpu().numpy()
        unsupported = [int(c) for c, row in candidate_bank.items()
                       if np.linalg.norm(np.asarray(row['prototype']) * active_cpu) <= 1e-6]
        bank = {c: row for c, row in candidate_bank.items() if int(c) not in unsupported}
        imported = sorted(int(c) for c in bank if int(c) not in local)
        for c in imported:
            if model.unit_ranks['fc2'][c] == 0:
                model.unit_ranks['fc2'][c] = 1
                # Avoid the singular gradient of a zero normalized row.
                # This is initialization in free output capacity, not a donor weight copy.
                with torch.no_grad():
                    initial = torch.tensor(bank[str(c)]['prototype'], device=active_fc1.device) * active_fc1
                    if initial.norm() <= 1e-6:
                        raise ValueError('Imported prototype has no support on allocated receiver FC1 coordinates')
                    model.fc2.weight[c].copy_(.1 * F.normalize(initial, dim=0))
            # Only current young rows may gain connections. Mature rows retain
            # their masks and gradient protection exactly.
            if model.unit_ranks['fc2'][c] == 1:
                model.weight_masks['fc2'][c] = active_fc1.to(model.weight_masks['fc2'].device)
                model.bias_masks['fc2'][c] = 1
        from fed_learning.strategies.incremental.nice import update_freeze_masks
        update_freeze_masks(model)
        trained_available = sorted(set(value['available']) | set(local) | set(imported))
        # Current task label IDs are protocol metadata, not future labels.
        # Keeping this denominator also avoids zero CE on one-class clients.
        # Untrained output rows remain gradient-frozen and unavailable at inference.
        allowed = sorted(set(trained_available) | set(map(int, task_classes)))
        old = [int(c) for c, t in value['class_tasks'].items() if int(t) < int(task)]
        teacher = None
        if len(old) > 1 and self.config['candle_v2_preservation_weight'] > 0:
            teacher = copy.deepcopy(model).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
        captured = {}
        audits = []
        bank_classes = sorted(map(int, bank))
        proto = (torch.tensor([bank[str(c)]['prototype'] for c in bank_classes],
                              device=model.fc2.weight.device, dtype=torch.float32)
                 if bank_classes else None)
        if proto is not None:
            proto = proto * active_fc1
        scale = self.config['candle_v2_logit_scale']

        def forward(x):
            h = features(model, x)
            captured['h'] = h
            captured['x'] = x
            return logits(model, model.dropout(h), allowed, scale)

        def auxiliary(current, x, y):
            h = captured.pop('h')
            captured.pop('x', None)
            align = h.float().sum() * 0.
            virtual = align
            preserve = align
            if proto is not None:
                present = torch.tensor([int(c) in bank_classes for c in y.detach().cpu().tolist()],
                                       device=y.device, dtype=torch.bool)
                if present.any():
                    target = torch.tensor([bank_classes.index(int(c)) for c in y[present].detach().cpu().tolist()],
                                          device=y.device, dtype=torch.long)
                    hn = F.normalize(h[present].float(), dim=1, eps=1e-6)
                    pn = F.normalize(proto, dim=1, eps=1e-6)
                    attraction = (1. - (hn * pn[target]).sum(1)).mean()
                    contrastive = F.cross_entropy(float(scale) * (hn @ pn.T), target)
                    align = .5 * (attraction + contrastive)
                missing_ids = [bank_classes.index(c) for c in imported]
                if missing_ids:
                    target = torch.tensor(imported, device=y.device, dtype=torch.long)
                    virtual = F.cross_entropy(logits(current, proto[missing_ids].detach(), allowed, scale), target)
            if teacher is not None:
                with torch.no_grad():
                    old_teacher = logits(teacher, features(teacher, x), old, scale)[:, old]
                old_student = logits(current, h, old, scale)[:, old]
                temperature = 2.
                preserve = F.kl_div(F.log_softmax(old_student / temperature, dim=1),
                                    F.softmax(old_teacher / temperature, dim=1),
                                    reduction='batchmean') * temperature**2
            total = (self.config['candle_v2_align_weight'] * align
                     + self.config['candle_v2_virtual_weight'] * virtual
                     + self.config['candle_v2_preservation_weight'] * preserve)
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite CANDLE V2 auxiliary objective')
            audits.append(dict(alignment=float(align.detach()), virtual_CE=float(virtual.detach()),
                               preservation=float(preserve.detach())))
            return total

        self.live[int(cid)] = dict(local=local, imported=imported, allowed=trained_available,
                                   training_denominator=allowed,
                                   unsupported_prototype_classes=unsupported,
                                   audits=audits, task=int(task), teacher=teacher)
        return dict(supervised_forward=forward, auxiliary_loss=auxiliary)

    def after_train(self, cid, model, result):
        live = self.live.pop(int(cid))
        value = state(model, cid)
        steps = int(result.get('optimizer_steps', 0))
        if steps:
            value['available'] = live['allowed']
            for c in live['allowed']:
                value['class_tasks'].setdefault(str(c), live['task'])
            for c in live['imported']:
                value['imported'][str(c)] = dict(task=live['task'],
                    donors=value['bank'][str(c)]['donors'], successful_training_steps=steps)
            # Native aggregation still uses genuine local data ownership.
            # Preserve locally learned imported rows separately from peer averaging.
            self.protected_rows[int(cid)] = {
                c: dict(weight=model.fc2.weight[c].detach().cpu().clone(),
                        bias=model.fc2.bias[c].detach().cpu().clone(),
                        mask=model.weight_masks['fc2'][c].detach().cpu().clone(),
                        bias_mask=model.bias_masks['fc2'][c].detach().cpu().clone())
                for c in live['imported'] if model.unit_ranks['fc2'][c] == 1}
        audit = dict(optimizer_steps=steps, local_classes=live['local'], imported_classes=live['imported'],
            available_count=len(value['available']),
            losses={key: sum(a[key] for a in live['audits']) / max(1, len(live['audits']))
                    for key in ('alignment', 'virtual_CE', 'preservation')},
            bank_round=value['bank_round'])
        audit['training_denominator_classes'] = live['training_denominator']
        audit['unsupported_prototype_classes'] = live['unsupported_prototype_classes']
        return audit

    @torch.no_grad()
    def after_round(self, models, clients, ids, summary, task, round_id, device):
        descriptors = {}
        restored_rows = 0
        per_class = int(self.config['candle_v2_proto_per_class'])
        batch_size = int(self.config['candle_v2_proto_batch_size'])
        for cid in ids:
            model = models[cid]
            client = clients[cid]
            for c, row in self.protected_rows.pop(int(cid), {}).items():
                model.fc2.weight[c].copy_(row['weight'].to(model.fc2.weight))
                model.fc2.bias[c].copy_(row['bias'].to(model.fc2.bias))
                model.weight_masks['fc2'][c].copy_(row['mask'].to(model.weight_masks['fc2']))
                model.bias_masks['fc2'][c].copy_(row['bias_mask'].to(model.bias_masks['fc2']))
                restored_rows += 1
            value = state(model, cid)
            if value['task'] != int(task):
                raise ValueError('Task state not initialized before prototype exchange')
            labels = sorted(map(int, torch.unique(client.y_train).tolist()))
            selected = []
            for c in labels:
                rows = torch.nonzero(client.y_train == c, as_tuple=False).flatten()
                positions = torch.linspace(0, len(rows)-1, min(per_class, len(rows))).long()
                selected.append(rows[positions])
            selected = torch.cat(selected)
            sums = {}
            counts = {}
            original_device = next(model.parameters()).device
            model.to(device)
            with evaluation_mode(model):
                for start in range(0, len(selected), batch_size):
                    ix = selected[start:start+batch_size]
                    x = client.X_train[ix].to(device)
                    y = client.y_train[ix].to(device)
                    h = features(model, x).float()
                    for c in torch.unique(y).tolist():
                        mask = y == c
                        sums[c] = sums.get(c, torch.zeros(h.shape[1], device=h.device)) + h[mask].sum(0)
                        counts[c] = counts.get(c, 0) + int(mask.sum())
            model.to(original_device)
            descriptors[int(cid)] = dict(task=int(task), round=int(round_id), owner=int(cid),
                classes={str(c): dict(prototype=(sums[c]/counts[c]).cpu().tolist(), count=counts[c]) for c in labels})
        bytes_sent = 0
        packets = 0
        imports = 0
        for cid in ids:
            native = summary['alpha_debug'].get(int(cid), summary['alpha_debug'].get(str(cid)))
            if native is None:
                raise ValueError('Missing graph alpha provenance')
            contributions = {}
            for peer, alpha in zip(native['group_ids'], native['alphas']):
                peer = int(peer)
                if peer == int(cid) or alpha <= 0 or peer not in descriptors:
                    continue
                packet = descriptors[peer]
                bytes_sent += len(json.dumps(packet, separators=(',', ':'), allow_nan=False).encode('utf-8'))
                packets += 1
                for c, data in packet['classes'].items():
                    if np.linalg.norm(np.asarray(data['prototype'], np.float32)) <= 1e-6:
                        continue
                    weight = float(alpha) * int(data['count'])
                    contributions.setdefault(c, []).append((peer, weight, np.asarray(data['prototype'], np.float32)))
            bank = {}
            local = set(descriptors[int(cid)]['classes'])
            for c, records in contributions.items():
                norm = sum(weight for _, weight, _ in records)
                mean = sum(weight * proto for _, weight, proto in records) / norm
                if not np.isfinite(mean).all():
                    raise FloatingPointError('Nonfinite peer prototype')
                if np.linalg.norm(mean) <= 1e-6:
                    continue
                bank[c] = dict(prototype=mean.tolist(), donors=[peer for peer, _, _ in records])
                imports += int(c not in local)
            value = state(models[cid], cid)
            value.update(bank=bank, bank_round=int(round_id), task=int(task))
        return dict(version=VERSION, proto_packets=packets, prototype_JSON_payload_bytes=bytes_sent,
            imported_rows_preserved_after_native_aggregation=restored_rows,
            next_round_missing_class_candidates=imports, raw_samples_transferred=0,
            native_dense_model_payload_estimate_bytes=summary.get('dense_model_transfer_upper_bound_bytes'),
            communication_scope='Actual canonical prototype JSON encoding + separate native dense estimate; excludes native capsule/transport headers, not full wire bytes')
