"""APPLIANCE: synchronous, temporary-leader class transfer simulation.

Private validation stays behind LocalValidator. Packets contain parameters and
metadata, never validation records. This is neither secure aggregation nor DP.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json

import numpy as np
import torch
import torch.nn.functional as F

from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes


def validate_config(config):
    if not config.get('denice_appliance_enabled', False):
        return
    if config.get('appliance_enabled', False):
        raise ValueError('The DENICE row overlay and upstream APPLIANCE service cannot run together')
    required = {'mode': 'decentralized', 'algorithm': 'denice',
                'denice_clustering_mode': 'adaptive_ap',
                'denice_shared_context_eval': False, 'denice_eval_route_mode': 'hard',
                'denice_age_merge_policy': 'none'}
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f'APPLIANCE requires {key}={value!r}')
    if config.get('task_start', 0) and not config.get('resume_state_path'):
        raise ValueError('Row overlay task_start > 0 requires a continuation checkpoint')
    if config.get('denice_capsule_mode') == 'paper':
        raise ValueError('APPLIANCE reserves validation exclusively for local checking; use training capsules')
    if config.get('denice_transfer_enabled') or config.get('denice_classifier_enabled'):
        raise ValueError('Disable other validation selectors for the APPLIANCE experiment')
    if config.get('denice_continual_width', 0) or config.get('denice_cl_method', 'legacy') != 'legacy':
        raise ValueError('APPLIANCE currently supports the native fc2 readout and legacy DENICE learner')
    if config.get('denice_clean_roles_dir'):
        raise ValueError('Use the documented NPZ loader; the separate clean-meta pipeline is not integrated')
    if not 0 < float(config.get('denice_validation_fraction', 0)) < 1:
        raise ValueError('APPLIANCE requires a nonempty held-out fraction')
    for key, default in [('appliance_min_evidence', 4), ('appliance_bank_per_class', 64),
                         ('appliance_batch_size', 128)]:
        v = config.get(key, default)
        if isinstance(v, bool) or int(v) != v or v < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key, default in [('appliance_eta', .5), ('appliance_min_recall', .7),
                         ('appliance_drop_probability', 0.)]:
        if not 0 <= float(config.get(key, default)) <= 1:
            raise ValueError(f'{key} must be in [0,1]')


def elect_leaders(labels, knowledge, available, failed=()):
    """Per-round election: most historically trained labels, then lowest ID."""
    leaders = {}
    for group in sorted(set(labels.values())):
        eligible = [cid for cid in labels if labels[cid] == group
                    and cid in available and cid not in failed]
        leaders[group] = min(eligible, key=lambda cid: (-len(knowledge.get(cid, ())), cid)) if eligible else None
    return leaders


def fingerprint(model, exact=False):
    """Structural compatibility, optionally exact feature extractor identity.

    Structural equality is only a necessary condition: validation must still
    reject semantically incompatible learned feature bases.
    """
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(f'{name}:{tuple(value.shape)}:{value.dtype}'.encode())
        if exact and not name.startswith('fc2.'):
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    for name, value in sorted(model.get_masks_state().items()):
        if not name.endswith('_fc2'):
            h.update(name.encode()); h.update(value.cpu().numpy().tobytes())
    for name, value in sorted(model.get_neuron_ages_state().items()):
        if name != 'fc2':
            h.update(name.encode()); h.update(np.asarray(value).tobytes())
    registry = getattr(model, 'adapter_registry', {})
    h.update(json.dumps(registry, sort_keys=True, default=str).encode())
    return h.hexdigest()


@dataclass
class Packet:
    task: int
    round: int
    source: int
    cluster: int
    class_id: int
    schema: str
    features: str
    version: str
    count: int
    row: torch.Tensor
    bias: torch.Tensor
    mask: torch.Tensor
    bias_mask: torch.Tensor


def valid_packet(packet, task, round_id, schema, versions):
    return (packet.task == task and packet.round == round_id and packet.schema == schema
            and versions.get(packet.source) == packet.version and packet.count > 0
            and torch.isfinite(packet.row).all().item()
            and torch.isfinite(packet.bias).all().item())


def aggregate_rows(packets, row, bias, mask, bias_mask):
    """Hierarchical count-weighted sums; normalize coordinates only once.

    Cluster sums retain their denominators: averaging cluster means uniformly
    would overweight small clusters and missing-coordinate contributions.
    """
    sums = {}
    for p in packets:
        if p.row.shape != row.shape or p.mask.shape != mask.shape:
            continue
        if p.cluster not in sums:
            sums[p.cluster] = [torch.zeros_like(row), torch.zeros_like(row),
                               torch.zeros_like(bias), torch.zeros_like(bias)]
        num, den, bn, bd = sums[p.cluster]
        weight = p.count * p.mask.to(row) * mask.to(row)
        bw = p.count * p.bias_mask.to(bias) * bias_mask.to(bias)
        num.add_(weight * p.row.to(row)); den.add_(weight)
        bn.add_(bw * p.bias.to(bias)); bd.add_(bw)
    if not sums:
        return row.clone(), bias.clone()
    num, den, bn, bd = [sum(v[k] for v in sums.values()) for k in range(4)]
    return (torch.where(den > 0, num / den.clamp_min(1), row),
            torch.where(bd > 0, bn / bd.clamp_min(1), bias))


class LocalValidator:
    """Bounded cumulative private holdout; not a training replay buffer."""
    def __init__(self, config):
        self.config = config
        self.bank = {}

    def add(self, x, y):
        cap = int(self.config.get('appliance_bank_per_class', 64))
        for label in torch.unique(y).tolist():
            # Input is already a deterministically randomized stratified fold.
            if int(label) not in self.bank:
                ix = torch.where(y == label)[0][:cap]
                self.bank[int(label)] = (x[ix].detach().cpu().clone(), y[ix].cpu().clone())

    def sufficient(self, label):
        minimum = int(self.config.get('appliance_min_evidence', 4))
        positives = len(self.bank.get(label, ([], []))[1])
        negatives = sum(len(y) for c, (_, y) in self.bank.items() if c != label)
        return positives >= minimum and negatives >= minimum

    @torch.no_grad()
    def score(self, model, detector, seen):
        device = next(model.parameters()).device
        modes = [(m, m.training) for m in model.modules()]
        active = deepcopy(getattr(model, 'active_adapters', {}))
        result = {}
        try:
            model.eval()
            for label, (x, y) in self.bank.items():
                loss, correct, predictions = 0., 0, []
                for start in range(0, len(y), int(self.config.get('appliance_batch_size', 128))):
                    stop = start + int(self.config.get('appliance_batch_size', 128))
                    logits, _ = _denice_routed_logits_with_episodes(
                        model, x[start:stop].to(device), detector, seen, str(device), route_mode='hard')
                    target = y[start:stop].to(device)
                    loss += float(F.cross_entropy(logits, target, reduction='sum'))
                    pred = logits.argmax(1)
                    correct += int((pred == target).sum())
                    predictions.extend(pred.cpu().tolist())
                result[label] = dict(ce=loss / len(y), recall=correct / len(y), predictions=predictions)
        finally:
            model.active_adapters = active
            for module, mode in modes:
                module.training = mode
        return result


def acceptable(before, after, label, min_gain=1e-6):
    """Strict target CE gain; no class CE/recall or target false-positive regression."""
    if label not in before or set(before) != set(after):
        return False
    for c in before:
        a, b = before[c], after[c]
        if not np.isfinite(b['ce']) or b['ce'] > a['ce'] + 1e-7 or b['recall'] < a['recall']:
            return False
        if c != label and b['predictions'].count(label) > a['predictions'].count(label):
            return False
    return after[label]['ce'] < before[label]['ce'] - min_gain


def shadow_check(model, detector, validator, seen, label, episode, row=None, bias=None):
    """Try registration or one predetermined blend, restore on every exit.

    A true model copy avoids accidental BN, adapter, router or mask mutation.
    Only row/bias and episode availability are committed after acceptance.
    """
    if not validator.sufficient(label):
        return False, 'unknown_gap'
    baseline = validator.score(model, detector, seen)
    shadow, router = deepcopy(model), deepcopy(detector)
    router.episode_classes[episode] = sorted(set(router.episode_classes.get(episode, [])) | {label})
    if row is not None:
        with torch.no_grad():
            shadow.fc2.weight[label].copy_(row.to(shadow.fc2.weight))
            shadow.fc2.bias[label].copy_(bias.to(shadow.fc2.bias))
    trial = validator.score(shadow, router, seen)
    if row is None and trial[label]['recall'] < float(validator.config.get('appliance_min_recall', .7)):
        return False, 'functional_gap'
    if not acceptable(baseline, trial, label):
        return False, 'validation_rejected'
    if row is not None:
        with torch.no_grad():
            model.fc2.weight[label].copy_(row.to(model.fc2.weight))
            model.fc2.bias[label].copy_(bias.to(model.fc2.bias))
    detector.episode_classes[episode] = router.episode_classes[episode]
    return True, 'registration' if row is None else 'transfer'


class ApplianceProtocol:
    def __init__(self, config):
        validate_config(config)
        config.setdefault('seed', config.get('random_seed', 42))
        self.config = config
        self.validators, self.knowledge, self.counts = {}, {}, {}

    def add_local_task(self, cid, x_validation, y_validation, y_train):
        self.validators.setdefault(cid, LocalValidator(self.config)).add(x_validation, y_validation)
        labels, counts = torch.unique(y_train.cpu(), return_counts=True)
        self.knowledge.setdefault(cid, set()).update(int(c) for c in labels)
        self.counts.setdefault(cid, {}).update({int(c): int(n) for c, n in zip(labels, counts)})

    def run_round(self, models, detectors, labels, task, round_id, task_classes):
        """Snapshot announcements -> cluster relays -> full mesh -> private commit.

        All clients in this process are trusted simulation actors. Failure flags
        affect this overlay only, not the already completed native DENICE round.
        """
        cfg = self.config
        all_classes = [int(c) for classes in task_classes.values() for c in classes]
        if len(all_classes) != len(set(all_classes)):
            raise ValueError('APPLIANCE requires disjoint class-incremental task mappings')
        available = set(labels) - set(cfg.get('appliance_unavailable_clients', []))
        failed = set(cfg.get('appliance_failed_servers', []))
        leaders = elect_leaders(labels, self.knowledge, available, failed)
        schema = hashlib.sha256(json.dumps(task_classes, sort_keys=True).encode()).hexdigest()
        seed = int(cfg.get('random_seed', cfg.get('seed', 42))) + task * 100000 + round_id
        rng = np.random.default_rng(seed)
        versions, signatures, packets = {}, {}, []
        seen = sorted({c for t, classes in task_classes.items() if int(t) <= task for c in classes})
        class_episode = {int(c): int(t) for t, classes in task_classes.items() for c in classes}
        audit = dict(task=task, round=round_id, leaders=leaders, topology='full_mesh',
                     dropped=0, packets=0, row_payload_bytes=0, decisions=[],
                     compatibility='exact' if cfg.get('appliance_exact_features', False) else 'structural_plus_validation')
        # Explicit logical mailboxes: one immutable source announcement per
        # reachable leader. This is an in-memory broadcast, not a socket layer.
        inboxes = {group: [] for group, leader in leaders.items() if leader is not None}
        for cid in sorted(available):
            model = models[cid]
            signatures[cid] = fingerprint(model, bool(cfg.get('appliance_exact_features', False)))
            digest = hashlib.sha256()
            for value in model.state_dict().values():
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            versions[cid] = f'{task}:{round_id}:{cid}:' + digest.hexdigest()
            if leaders.get(labels[cid]) is None or cid in failed:
                continue
            validator = self.validators[cid]
            scores = validator.score(model, detectors[cid], seen)
            ages = model.get_neuron_ages_state()['fc2']
            for label in sorted(self.knowledge[cid] & set(seen)):
                if not validator.sufficient(label) or ages[label] < 1:
                    continue
                if scores[label]['recall'] < float(cfg.get('appliance_min_recall', .7)):
                    continue
                if rng.random() < float(cfg.get('appliance_drop_probability', 0.)):
                    audit['dropped'] += 1
                    continue
                p = Packet(task, round_id, cid, labels[cid], label, schema, signatures[cid],
                           versions[cid], self.counts[cid][label],
                           model.fc2.weight[label].detach().cpu().clone(),
                           model.fc2.bias[label].detach().cpu().clone(),
                           model.weight_masks['fc2'][label].cpu().clone(),
                           model.bias_masks['fc2'][label].cpu().clone())
                if valid_packet(p, task, round_id, schema, versions):
                    packets.append(p)
                    audit['row_payload_bytes'] += sum(v.numel() * v.element_size() for v in (p.row, p.bias, p.mask, p.bias_mask))
        audit['packets'] = len(packets)
        for p in packets:
            for group in inboxes:
                inboxes[group].append(p)
        audit['leader_deliveries'] = sum(len(inbox) for inbox in inboxes.values())
        audit['inter_leader_deliveries'] = sum(
            p.cluster != group for group, inbox in inboxes.items() for p in inbox)
        # Full-mesh cluster leaders relay immutable packets. No multihop duplicate
        # contributions; each source appears once for a receiver/class.
        for cid in sorted(available - failed):
            if leaders.get(labels[cid]) is None:
                continue
            model, detector, validator = models[cid], detectors[cid], self.validators[cid]
            for label in seen:
                entry = dict(receiver=cid, class_id=label, accepted=False)
                if not validator.sufficient(label):
                    entry['reason'] = 'unknown_gap'
                elif model.get_neuron_ages_state()['fc2'][label] >= 2:
                    entry['reason'] = 'protected_row'
                else:
                    episode = class_episode[label]
                    baseline = validator.score(model, detector, seen)
                    if label not in detector.episode_classes.get(episode, []):
                        accepted, reason = shadow_check(model, detector, validator, seen, label, episode)
                        if accepted:
                            entry.update(accepted=True, reason=reason)
                            audit['decisions'].append(entry)
                            continue
                    if baseline[label]['recall'] >= float(cfg.get('appliance_min_recall', .7)):
                        entry['reason'] = 'no_functional_gap'
                    else:
                        donors = [p for p in inboxes[labels[cid]] if p.source != cid and p.class_id == label
                                  and p.features == signatures[cid]
                                  and valid_packet(p, task, round_id, schema, versions)]
                        if not donors:
                            entry['reason'] = 'no_compatible_donor'
                        else:
                            row, bias = model.fc2.weight[label].detach().cpu(), model.fc2.bias[label].detach().cpu()
                            target, target_bias = aggregate_rows(donors, row, bias,
                                model.weight_masks['fc2'][label].cpu(), model.bias_masks['fc2'][label].cpu())
                            eta = float(cfg.get('appliance_eta', .5))
                            accepted, reason = shadow_check(model, detector, validator, seen, label, episode,
                                torch.lerp(row, target, eta), torch.lerp(bias, target_bias, eta))
                            entry.update(accepted=accepted, reason=reason, donors=[p.source for p in donors])
                audit['decisions'].append(entry)
        audit['accepted'] = sum(v['accepted'] for v in audit['decisions'])
        return audit
