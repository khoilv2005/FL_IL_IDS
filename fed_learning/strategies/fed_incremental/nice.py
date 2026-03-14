"""
Federated wrappers for NICE-based incremental learning.
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from ...core import BaseAggregator
from ..incremental.nice import (
    NICETrainer,
    pick_top_neurons,
    select_learner_units,
    drop_young_to_learner,
    grow_all_to_young,
    increase_unit_ranks,
    update_freeze_masks,
)


class NICEAggregator(BaseAggregator):
    """
    NICE Aggregator - FedAvg với bảo vệ tham số ở mức từng neuron.

    Sau khi average, aggregator sẽ khôi phục lại các tham số của neuron mature
    từ global model để tránh việc tri thức cũ bị ghi đè bởi neuron trẻ.
    """

    def __init__(self):
        self._frozen_keys: Set[str] = set()
        self._freeze_masks: Dict[str, np.ndarray] = {}

    def set_frozen_keys(self, keys: List[str]):
        """Đánh dấu các parameter key bị freeze hoàn toàn, không được average."""
        self._frozen_keys = set(keys)

    def set_freeze_masks(self, freeze_masks: Dict[str, np.ndarray]):
        """Set per-layer freeze masks for partial freezing.

        Freeze mask cho phép partial freezing theo từng neuron thay vì chỉ
        freeze cả layer.
        """
        self._freeze_masks = {k: np.array(v) for k, v in freeze_masks.items()}

    def aggregate(
        self,
        results: List[Dict],
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> OrderedDict:
        """Aggregate và khôi phục tham số của neuron mature nếu cần."""
        agg_results = []
        for r in results:
            params = r.get("params", r.get("masked_params"))
            agg_results.append(
                {
                    "params": params,
                    "num_samples": r["num_samples"],
                }
            )

        averaged = self._weighted_average(agg_results)

        # Restore frozen parameters from global model
        if global_params is not None:
            # 1. Fully frozen keys (all neurons mature)
            for key in self._frozen_keys:
                if key in global_params and key in averaged:
                    averaged[key] = global_params[key].clone()

            # 2. Per-neuron freezing for partially-mature layers
            if self._freeze_masks:
                for key in averaged:
                    if key not in global_params:
                        continue

                    layer_name = key.split(".")[0]
                    if layer_name == "gru" or layer_name not in self._freeze_masks:
                        continue

                    freeze = self._freeze_masks[layer_name]
                    if not np.any(freeze):
                        continue

                    mask = torch.tensor(freeze, dtype=torch.bool)

                    if "weight" in key and averaged[key].dim() >= 2:
                        if len(freeze) == averaged[key].shape[0]:
                            # Restore frozen rows from global params
                            averaged[key][mask] = global_params[key][mask].clone()
                    elif "bias" in key:
                        if len(freeze) == averaged[key].shape[0]:
                            averaged[key][mask] = global_params[key][mask].clone()

        return averaged

    def set_task(self, task_id: int):
        """Compatibility interface."""
        pass
