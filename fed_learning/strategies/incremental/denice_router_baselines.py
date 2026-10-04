"""Continuous-feature routing baselines for frozen-checkpoint diagnostics."""
from __future__ import annotations

import numpy as np


def feature_matrix(values):
    z = np.asarray(values, dtype=np.float64)
    if z.ndim != 2 or not z.shape[1] or not np.isfinite(z).all():
        raise ValueError('Expected a finite [samples, features] matrix')
    return z


class PrototypeRouter:
    """Nearest centroid or pooled, shrinkage-regularized Mahalanobis distance.

    Scores are negative distances, not calibrated task probabilities.
    """
    def __init__(self, method='centroid', shrinkage=0.1):
        if method not in ('centroid', 'mahalanobis'):
            raise ValueError(method)
        if not 0 <= shrinkage <= 1:
            raise ValueError('shrinkage must be in [0, 1]')
        self.method, self.shrinkage = method, float(shrinkage)

    def fit(self, features):
        if not features:
            raise ValueError('Cannot fit an empty task bank')
        self.tasks = np.asarray(sorted(features), dtype=np.int64)
        matrices = [feature_matrix(features[int(t)]) for t in self.tasks]
        if any(not len(z) for z in matrices):
            raise ValueError('Each fitted task needs samples')
        self.means = np.stack([z.mean(0) for z in matrices])
        self.cholesky = None
        self.jitter = 0.0
        if self.method == 'mahalanobis':
            residual = np.concatenate([z - mu for z, mu in zip(matrices, self.means)])
            covariance = residual.T @ residual / max(len(residual) - len(matrices), 1)
            scale = float(np.trace(covariance) / covariance.shape[0])
            covariance = ((1 - self.shrinkage) * covariance
                          + self.shrinkage * scale * np.eye(len(covariance)))
            for power in range(7):
                self.jitter = max(1e-6, scale * 1e-6) * 10**power
                try:
                    self.cholesky = np.linalg.cholesky(
                        covariance + self.jitter * np.eye(len(covariance)))
                    break
                except np.linalg.LinAlgError:
                    continue
            if self.cholesky is None:
                raise ValueError('Mahalanobis covariance remained singular')
        return self

    def scores(self, values):
        from scipy.linalg import solve_triangular
        z = feature_matrix(values)
        delta = z[:, None, :] - self.means[None, :, :]
        if self.cholesky is not None:
            shape = delta.shape
            delta = solve_triangular(self.cholesky, delta.reshape(-1, shape[-1]).T,
                                     lower=True).T.reshape(shape)
        return -np.sum(delta * delta, axis=-1)

    def predict(self, values):
        return self.tasks[self.scores(values).argmax(1)]

    def state_dict(self):
        return dict(schema_version=1, method=self.method, tasks=self.tasks,
                    means=self.means, cholesky=self.cholesky,
                    shrinkage=self.shrinkage, jitter=self.jitter)
