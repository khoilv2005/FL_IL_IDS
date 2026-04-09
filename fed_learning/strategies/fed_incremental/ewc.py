"""
Federated wrappers for EWC-based incremental learning.
"""

from ...core import BaseTrainer
from ..federated.fedavg import FedAvgTrainer
from ..federated.fedprox import FedProxTrainer
from ..incremental.ewc import EWCMixin


class FedAvgEWCTrainer(EWCMixin, FedAvgTrainer):
    """FedAvg + EWC: Standard averaging with EWC regularization."""

    def __init__(self, **kwargs):
        ewc_args = {
            "ewc_lambda": kwargs.pop("ewc_lambda", 1000.0),
            "fisher_samples": kwargs.pop("fisher_samples", 200),
            "online_ewc": kwargs.pop("online_ewc", False),
            "gamma": kwargs.pop("gamma", 0.9),
            "temp_dir": kwargs.pop("temp_dir", "./temp_ewc_storage"),
            "debug_logging": kwargs.pop("debug_logging", False),
        }
        EWCMixin.__init__(self, **ewc_args)


class FedProxEWCTrainer(EWCMixin, FedProxTrainer):
    """FedProx + EWC: Proximal regularization with EWC."""

    def __init__(self, **kwargs):
        ewc_args = {
            "ewc_lambda": kwargs.pop("ewc_lambda", 1000.0),
            "fisher_samples": kwargs.pop("fisher_samples", 200),
            "online_ewc": kwargs.pop("online_ewc", False),
            "gamma": kwargs.pop("gamma", 0.9),
            "temp_dir": kwargs.pop("temp_dir", "./temp_ewc_storage"),
            "debug_logging": kwargs.pop("debug_logging", False),
        }
        self.mu = kwargs.pop("mu", 0.01)
        EWCMixin.__init__(self, **ewc_args)
