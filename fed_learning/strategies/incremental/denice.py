"""
DeNICE trainer (plan section 2 / 13).

DeNICE reuses the NICE phase-based training loop verbatim (CE loss on the
Let_Learner-masked output + mature-gradient freezing). The only additions are:

    - CANC configuration (capacity controller defaults).
    - Novelty layer weights.

The micro-adapter parameters live inside :class:`DeNICEModel` and are optimized
together with the plastic NICE neurons during the standard phase loop, so no
loss/optimizer changes are needed here.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .nice import NICETrainer
from .denice_capacity import CANCConfig
from .denice_novelty import DEFAULT_LAYER_WEIGHTS


class DeNICETrainer(NICETrainer):
    """NICE trainer + CANC/novelty configuration for DeNICE."""

    def __init__(
        self,
        tau: float = 0.95,
        max_phases: int = 5,
        phase_epochs: int = 5,
        memo_per_class: int = 50,
        canc_config: Optional[CANCConfig] = None,
        novelty_layer_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(
            tau=tau,
            max_phases=max_phases,
            phase_epochs=phase_epochs,
            memo_per_class=memo_per_class,
            **kwargs,
        )
        self.canc_config = canc_config or CANCConfig()
        self.novelty_layer_weights = dict(
            novelty_layer_weights or DEFAULT_LAYER_WEIGHTS
        )
