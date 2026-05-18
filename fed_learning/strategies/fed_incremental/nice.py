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

    @staticmethod
    def _restore_rows(
        averaged: OrderedDict,
        global_params: OrderedDict,
        key: str,
        freeze: np.ndarray,
    ) -> None:
        if key not in averaged or key not in global_params or not np.any(freeze):
            return
        tensor = averaged[key]
        if tensor.dim() == 0 or tensor.shape[0] != len(freeze):
            return
        mask = torch.tensor(freeze, dtype=torch.bool, device=tensor.device)
        averaged[key][mask] = global_params[key].to(tensor.device)[mask].clone()

    @staticmethod
    def _restore_gru_rows(
        averaged: OrderedDict,
        global_params: OrderedDict,
        key: str,
        freeze: np.ndarray,
    ) -> None:
        if key not in averaged or key not in global_params or not np.any(freeze):
            return
        tensor = averaged[key]
        hidden = len(freeze)
        if tensor.dim() == 0 or tensor.shape[0] != 3 * hidden:
            return
        gate_mask = torch.tensor(
            np.tile(freeze, 3), dtype=torch.bool, device=tensor.device
        )
        averaged[key][gate_mask] = global_params[key].to(tensor.device)[
            gate_mask
        ].clone()

    def _restore_batchnorm_channels(
        self, averaged: OrderedDict, global_params: OrderedDict
    ) -> None:
        bn_map = {"conv1": "bn1", "conv2": "bn2", "conv3": "bn3"}
        for conv_layer, bn_name in bn_map.items():
            freeze = self._freeze_masks.get(conv_layer)
            if freeze is None or not np.any(freeze):
                continue
            for suffix in ("weight", "bias", "running_mean", "running_var"):
                self._restore_rows(
                    averaged, global_params, f"{bn_name}.{suffix}", freeze
                )

    def _restore_gru_channels(
        self, averaged: OrderedDict, global_params: OrderedDict
    ) -> None:
        freeze = self._freeze_masks.get("gru")
        if freeze is None or not np.any(freeze):
            return
        for key in list(averaged.keys()):
            if key.startswith("gru.weight_") or key.startswith("gru.bias_"):
                self._restore_gru_rows(averaged, global_params, key, freeze)

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

                self._restore_batchnorm_channels(averaged, global_params)
                self._restore_gru_channels(averaged, global_params)

        return averaged

    def set_task(self, task_id: int):
        """Compatibility interface."""
        pass
