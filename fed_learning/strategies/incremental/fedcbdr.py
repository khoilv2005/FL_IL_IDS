"""Standalone CBDR trainer for non-federated incremental learning."""

from ..fed_incremental.fedcbdr import (
    FedCBDRTrainer,
    ReplayBuffer,
    LeverageScoreCalculator,
)


class CBDRTrainer(FedCBDRTrainer):
    """
    Wrapper tên gọn cho Class-wise Balancing Data Replay ở chế độ local.
    """

    pass


__all__ = ["CBDRTrainer", "ReplayBuffer", "LeverageScoreCalculator"]
