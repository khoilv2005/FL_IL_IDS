"""
Federated wrappers for DER-based incremental learning.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set

from ...core import BaseAggregator
from ..incremental.der import DERTrainer


class DERAggregator(BaseAggregator):
    """
    DER aggregation — weighted average với bảo vệ tham số đã đóng băng.

    Ý tưởng là:
    - vẫn average như FedAvg
    - nhưng các tham số extractor cũ đã freeze sẽ được khôi phục từ
      global model để tránh drift do sai số số học.
    """

    def __init__(self):
        self.trainable_keys: Set[str] = set()

    def set_trainable_keys(self, keys):
        """
        Ghi nhận những parameter nào thực sự trainable ở task hiện tại.

        Chỉ các key này mới cần được tin tưởng sau bước aggregate.
        """
        self.trainable_keys = set(keys)

    def aggregate(
        self,
        results: list,
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> OrderedDict:
        """
        Aggregate update từ client và khôi phục frozen params nếu cần.

        Luồng:
        1. weighted average toàn bộ params
        2. param nào không trainable thì chép lại từ global model
        """
        # Standard weighted average (reuse from BaseAggregator)
        agg = self._weighted_average(results)

        if agg is None:
            # All training threads failed (empty results) — return global params unchanged
            return global_params if global_params is not None else OrderedDict()

        # Restore frozen params from global to prevent drift
        if global_params is not None and self.trainable_keys:
            for k in agg:
                if k not in self.trainable_keys:
                    agg[k] = global_params[k].clone()

        return agg

    def set_task(self, task_id: int):
        """Hàm tương thích để pipeline chung có thể gọi mà không lỗi."""
        pass
