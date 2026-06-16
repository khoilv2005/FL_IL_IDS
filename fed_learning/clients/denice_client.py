"""
DeNICE Client - NICE phase-based training with active micro-adapters.

The DeNICE client behaves exactly like :class:`NICEClient` (phase-based
tau-greedy training, connection pruning, mature gradient freezing). The
difference is purely in the model: a :class:`DeNICEModel` applies whichever
micro-adapters are currently *active* during ``forward_output``. Adapter
parameters are part of ``model.parameters()`` so they are optimized together
with the plastic NICE neurons; inactive adapters receive no gradient and stay
frozen automatically.

The CANC decision (which adapters to activate, whether to freeze low layers) is
made by the task loop before ``train`` is called, then applied to the model via
``model.add_adapter`` / ``model.set_active_adapter``.
"""

from __future__ import annotations

from typing import Any, Dict

from .nice_client import NICEClient


class DeNICEClient(NICEClient):
    """NICE client whose model carries capacity-aware micro-adapters."""

    def train(self, *args, **kwargs) -> Dict[str, Any]:
        result = super().train(*args, **kwargs)

        model = self.model
        if hasattr(model, "get_adapter_registry_state"):
            result["adapter_registry"] = model.get_adapter_registry_state()
            result["adapter_param_count"] = int(model.adapter_param_count())
            result["active_adapters"] = dict(getattr(model, "active_adapters", {}))
        return result

    def get_buffer_stats(self) -> Dict:
        return {
            "buffer_type": "none",
            "has_replay": False,
            "note": "DeNICE is replay-free (NICE + micro-adapters)",
        }
