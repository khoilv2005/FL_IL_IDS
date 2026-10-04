"""Per-sample TIP-inspired routing. No training/aggregation hooks are installed."""
from __future__ import annotations

import hashlib
import numpy as np
import torch

from .denice_router_baselines import feature_matrix


def normalize_rows(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


class SubspaceRouter:
    def __init__(self, energy=0.95, max_rank=32, basis_mode='independent', reference_mode='rms'):
        if not 0 < energy <= 1 or max_rank < 1:
            raise ValueError('Invalid energy/rank')
        if basis_mode not in ('independent', 'residual'):
            raise ValueError(basis_mode)
        if reference_mode not in ('mean', 'rms'):
            raise ValueError(reference_mode)
        self.energy, self.max_rank = float(energy), int(max_rank)
        self.basis_mode, self.reference_mode = basis_mode, reference_mode

    def fit(self, features):
        if not features:
            raise ValueError('Cannot fit an empty task bank')
        self.tasks = np.asarray(sorted(features), dtype=np.int64)
        matrices = [feature_matrix(features[int(t)]) for t in self.tasks]
        if any(not len(z) for z in matrices):
            raise ValueError('Each fitted task needs samples')
        d = matrices[0].shape[1]
        union = np.empty((d, 0))
        self.bases, self.moments, self.diagnostics = [], [], []
        for task, z in zip(self.tasks, matrices):
            residual = z - (z @ union) @ union.T if self.basis_mode == 'residual' else z
            _, singular, vt = np.linalg.svd(residual, full_matrices=False)
            power = singular**2
            total = float(power.sum())
            tolerance = max(float(np.sum(z*z)) * 1e-12, 1e-24)
            rank = 0 if total <= tolerance else min(
                int(np.searchsorted(np.cumsum(power), self.energy * total)) + 1,
                self.max_rank, int(np.sum(power > tolerance)))
            if self.basis_mode == 'residual':
                rank = min(rank, d - union.shape[1])
            basis = vt[:rank].T.copy()
            if self.basis_mode == 'residual' and rank:
                basis -= union @ (union.T @ basis)
                basis = np.linalg.qr(basis, mode='reduced')[0]
                union = np.concatenate([union, basis], axis=1)
            self.bases.append(basis)
            self.moments.append(z.T @ z / len(z))
            self.diagnostics.append(dict(task=int(task), rank=rank, samples=len(z),
                                         residual_energy=total,
                                         original_energy=float(np.sum(z*z))))
        references = []
        for z, moment in zip(matrices, self.moments):
            if self.reference_mode == 'mean':
                references.append(self.relevance(z).mean(0))
            else:
                references.append([np.sqrt(max(float(np.sum(u * (moment @ u))), 0))
                                   for u in self.bases])
        self.references = np.asarray(references)
        return self

    def relevance(self, values):
        z = feature_matrix(values)
        return np.stack([np.linalg.norm(z @ u, axis=1) for u in self.bases], axis=1)

    def scores(self, values):
        return normalize_rows(self.relevance(values)) @ normalize_rows(self.references).T

    def predict(self, values):
        # Sorted task IDs give a deterministic tie/zero-vector fallback.
        return self.tasks[self.scores(values).argmax(1)]

    def state_dict(self):
        return dict(schema_version=1, method='tip', tasks=self.tasks, bases=self.bases,
                    moments=self.moments, references=self.references,
                    energy=self.energy, max_rank=self.max_rank, basis_mode=self.basis_mode,
                    reference_mode=self.reference_mode, diagnostics=self.diagnostics)


@torch.no_grad()
def continuous_features(model, inputs, batch_size=512):
    """Adapter-free fc1 features; restore all module modes and active adapters."""
    active = dict(model.active_adapters)
    modes = [(module, module.training) for module in model.modules()]
    result = []
    try:
        model.eval()
        model.clear_active_adapters()
        device = next(model.parameters()).device
        for start in range(0, len(inputs), batch_size):
            result.append(model.penultimate_features(inputs[start:start+batch_size].to(device))
                          .detach().float().cpu().numpy())
    finally:
        model.active_adapters = active
        for module, training in modes:
            module.training = training
    if not result:
        return np.empty((0, model.fc2.in_features), dtype=np.float32)
    values = np.concatenate(result)
    feature_matrix(values)
    return values


def encoder_fingerprint(model):
    """Conservative signature includes the entire model plus non-buffer masks."""
    digest = hashlib.sha256(b'fc1:adapter-free:identity:v1')
    collections = [('state', model.state_dict())]
    for name in ('weight_masks', 'freeze_masks'):
        collections.append((name, getattr(model, name, {})))
    for collection, values in collections:
        for key, value in sorted(values.items()):
            digest.update(f'{collection}:{key}'.encode())
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().contiguous().numpy()
                digest.update(str((array.dtype, array.shape)).encode())
                digest.update(array.tobytes())
            else:
                digest.update(repr(value).encode())
    return digest.hexdigest()
