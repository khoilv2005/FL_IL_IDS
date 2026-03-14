"""Standalone EWC trainer for non-federated incremental learning."""

from ...core import BaseTrainer
from ..fed_incremental.ewc import EWCMixin


class EWCTrainer(EWCMixin, BaseTrainer):
    """
    EWC trainer dùng cho incremental learning thuần local.

    Loss nền là CrossEntropy từ `BaseTrainer`, sau đó được cộng thêm
    regularization EWC từ `EWCMixin`.
    """

    def __init__(self, **kwargs):
        ewc_args = {
            "ewc_lambda": kwargs.pop("ewc_lambda", 1000.0),
            "fisher_samples": kwargs.pop("fisher_samples", 200),
            "online_ewc": kwargs.pop("online_ewc", False),
            "gamma": kwargs.pop("gamma", 0.9),
            "temp_dir": kwargs.pop("temp_dir", "./temp_ewc_storage"),
        }
        EWCMixin.__init__(self, **ewc_args)
