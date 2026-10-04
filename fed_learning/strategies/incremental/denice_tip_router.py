"""Per-sample TIP-inspired routing. No training/aggregation hooks are installed."""
from __future__ import annotations

import hashlib
import numpy as np
import torch

from .denice_router_baselines import feature_matrix


def normalize_rows(values):
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def stable_feature_svd(values):
    """Right singular vectors with logged numerical fallbacks, not sample drops.

    Keep the original NumPy result when it converges. On failure, use the
    classical LAPACK driver on a scaled copy, then a symmetric Gram eigensolve.
    Scaling preserves directions and singular values are returned in input units.
    """
    from scipy import linalg

    matrix = feature_matrix(values)
    errors = []
    try:
        _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
        if not (np.isfinite(singular).all() and np.isfinite(vt).all()):
            raise np.linalg.LinAlgError('Non-finite NumPy SVD result')
        return singular, vt, dict(solver='numpy_svd', fallback_errors=[])
    except np.linalg.LinAlgError as exc:
        errors.append(str(exc))
    scale = float(np.max(np.abs(matrix)))
    if scale == 0:
        return np.zeros(0), np.empty((0, matrix.shape[1])), dict(
            solver='zero_matrix', fallback_errors=errors)
    scaled = np.array(matrix / scale, dtype=np.float64, order='F', copy=True)
    try:
        _, singular, vt = linalg.svd(scaled, full_matrices=False,
                                     lapack_driver='gesvd', check_finite=True)
        singular = singular * scale
        if not (np.isfinite(singular).all() and np.isfinite(vt).all()):
            raise np.linalg.LinAlgError('Non-finite gesvd result')
        return singular, vt, dict(solver='scipy_gesvd_scaled', fallback_errors=errors)
    except np.linalg.LinAlgError as exc:
        errors.append(str(exc))
    try:
        gram = scaled.T @ scaled
        eigenvalues, vectors = linalg.eigh((gram + gram.T)*0.5, driver='evr', check_finite=True)
        order = np.argsort(eigenvalues)[::-1][:min(matrix.shape)]
        singular = np.sqrt(np.maximum(eigenvalues[order], 0)) * scale
        vt = vectors[:, order].T
        if not (np.isfinite(singular).all() and np.isfinite(vt).all()):
            raise np.linalg.LinAlgError('Non-finite Gram eigensolve result')
        return singular, vt, dict(solver='scipy_gram_eigh_scaled', fallback_errors=errors)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            f'All feature decompositions failed for shape={matrix.shape}: {errors + [str(exc)]}') from exc


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
            try:
                singular, vt, solver_audit = stable_feature_svd(residual)
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise ValueError(f'TIP task={int(task)} basis={self.basis_mode} '
                                 f'reference={self.reference_mode} rank_cap={self.max_rank}: {exc}') from exc
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
                                         original_energy=float(np.sum(z*z)), **solver_audit))
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
