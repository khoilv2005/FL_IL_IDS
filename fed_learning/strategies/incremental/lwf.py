"""Standalone LwF trainer for non-federated incremental learning."""

from ..fed_incremental.fedlwf import FedLwFTrainer


class LwFTrainer(FedLwFTrainer):
    """
    Wrapper tên gọn cho Learning without Forgetting ở chế độ local.

    Toàn bộ logic CE + distillation được kế thừa từ `FedLwFTrainer`.
    """

    pass
