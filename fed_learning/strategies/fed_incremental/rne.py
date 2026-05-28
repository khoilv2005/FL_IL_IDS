"""Federated wrappers for RNE-based incremental learning."""

from ..incremental.rne import RNETrainer
from .der import DERAggregator


class RNEAggregator(DERAggregator):
    """RNE aggregation with DER-style frozen parameter protection."""

    pass
