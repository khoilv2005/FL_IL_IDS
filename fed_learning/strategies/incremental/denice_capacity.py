"""
Capacity-Aware Neurogenesis Controller (CANC) - plan section 2.2 and section 3.

Tracks per-layer capacity and decides, before training a new task, whether to:

    NICE_ONLY              keep training NICE normally
    HIGH_LAYER_ONLY        freeze low conv layers, train high layers only
    ADD_ADAPTER            instantiate / enable a micro-adapter for the layer
    EMERGENCY_LOW_ADAPTER  enable an adapter on a low layer (last resort)
    GRACEFUL_RECYCLING     retire low-importance mature neurons, revive later

Capacity pressure (plan section 2.2)::

    kappa_i,l,t = alpha*(1 - rho0_i,l) + beta*u_i,l + gamma*dL_val + delta*nu_i,t

``dL_val`` defaults to 0 when no old validation/prototype check is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np

# CANC actions
NICE_ONLY = "NICE_ONLY"
HIGH_LAYER_ONLY = "HIGH_LAYER_ONLY"
ADD_ADAPTER = "ADD_ADAPTER"
EMERGENCY_LOW_ADAPTER = "EMERGENCY_LOW_ADAPTER"
GRACEFUL_RECYCLING = "GRACEFUL_RECYCLING"

LOW_LAYERS = {"conv1", "conv2"}
HIGH_LAYERS = {"conv3", "gru", "fc1", "fc2"}

# Adapter priority (highest first), plan Rule 4.
ADAPTER_PRIORITY: List[str] = ["fc1", "gru", "conv3", "conv2", "conv1"]


@dataclass
class CANCConfig:
    """Heuristic CANC defaults (plan section 2.2)."""

    epsilon_free: float = 0.30
    epsilon_adapter: float = 0.10
    xi_novelty: float = 0.35
    xi_high_novelty: float = 0.55
    xi_consume: float = 0.80
    kappa_mid: float = 0.45
    kappa_high: float = 0.75
    kappa_adapter: float = 0.60
    kappa_recycle: float = 0.85
    epsilon_recycle_free: float = 0.0

    alpha: float = 0.45
    beta: float = 0.25
    gamma: float = 0.15
    delta: float = 0.30

    # Which layers may receive an adapter. MVP = fc1 only.
    enabled_adapter_layers: List[str] = field(default_factory=lambda: ["fc1"])
    enable_recycling: bool = False
    recycle_ratio: float = 0.02
    recycle_min: int = 1
    recycle_max_per_layer: int = 8
    recycle_grace_tasks: int = 1
    recycle_usage_recent_threshold: float = 0.10
    recycle_max_old_metric_drop: float = 0.02
    recycle_require_old_check: bool = True

    @classmethod
    def from_dict(cls, config: Optional[Dict]) -> "CANCConfig":
        config = config or {}
        defaults = cls()
        adapter_layers = _parse_adapter_layers(
            config.get("denice_adapter_layers", defaults.enabled_adapter_layers)
        )
        return cls(
            epsilon_free=float(config.get("denice_epsilon_free", defaults.epsilon_free)),
            epsilon_adapter=float(config.get("denice_epsilon_adapter", defaults.epsilon_adapter)),
            xi_novelty=float(config.get("denice_xi_novelty", defaults.xi_novelty)),
            xi_high_novelty=float(config.get("denice_xi_high_novelty", defaults.xi_high_novelty)),
            xi_consume=float(config.get("denice_xi_consume", defaults.xi_consume)),
            kappa_mid=float(config.get("denice_kappa_mid", defaults.kappa_mid)),
            kappa_high=float(config.get("denice_kappa_high", defaults.kappa_high)),
            kappa_adapter=float(config.get("denice_kappa_adapter", defaults.kappa_adapter)),
            kappa_recycle=float(config.get("denice_kappa_recycle", defaults.kappa_recycle)),
            epsilon_recycle_free=float(
                config.get("denice_epsilon_recycle_free", defaults.epsilon_recycle_free)
            ),
            alpha=float(config.get("denice_alpha", defaults.alpha)),
            beta=float(config.get("denice_beta", defaults.beta)),
            gamma=float(config.get("denice_gamma", defaults.gamma)),
            delta=float(config.get("denice_delta", defaults.delta)),
            enabled_adapter_layers=adapter_layers,
            enable_recycling=bool(config.get("denice_enable_recycling", defaults.enable_recycling)),
            recycle_ratio=float(config.get("denice_recycle_ratio", defaults.recycle_ratio)),
            recycle_min=int(config.get("denice_recycle_min", defaults.recycle_min)),
            recycle_max_per_layer=int(
                config.get("denice_recycle_max_per_layer", defaults.recycle_max_per_layer)
            ),
            recycle_grace_tasks=int(
                config.get("denice_recycle_grace_tasks", defaults.recycle_grace_tasks)
            ),
            recycle_usage_recent_threshold=float(
                config.get(
                    "denice_recycle_usage_recent_threshold",
                    defaults.recycle_usage_recent_threshold,
                )
            ),
            recycle_max_old_metric_drop=float(
                config.get(
                    "denice_recycle_max_old_metric_drop",
                    defaults.recycle_max_old_metric_drop,
                )
            ),
            recycle_require_old_check=bool(
                config.get(
                    "denice_recycle_require_old_check",
                    defaults.recycle_require_old_check,
                )
            ),
        )


# Layers that have an age-tracked neuron pool we care about for capacity.
CAPACITY_LAYERS: List[str] = ["conv1", "conv2", "conv3", "gru", "fc1"]


def _parse_adapter_layers(value) -> List[str]:
    """Parse adapter layer config from sequence or comma-separated string."""
    if value is None:
        return ["fc1"]
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",")]
    elif isinstance(value, Iterable):
        raw = [str(part).strip() for part in value]
    else:
        raw = [str(value).strip()]

    valid = set(ADAPTER_PRIORITY)
    parsed: List[str] = []
    for layer in raw:
        if not layer:
            continue
        if layer not in valid:
            raise ValueError(
                f"Unsupported DeNICE adapter layer '{layer}'. "
                f"Supported layers: {ADAPTER_PRIORITY}"
            )
        if layer not in parsed:
            parsed.append(layer)
    return parsed or ["fc1"]


def compute_capacity_state(model) -> Dict[str, Dict[str, float]]:
    """Per-layer capacity stats from ``model.unit_ranks``.

    Returns dict ``layer -> {rho0, rhom, free, mature, learner, total}`` where
    ``rho0 = free/total`` (plan rho0) and ``rhom = mature/total`` (plan rho^m).
    """
    state: Dict[str, Dict[str, float]] = {}
    unit_ranks = getattr(model, "unit_ranks", {})
    for name in CAPACITY_LAYERS:
        ranks = unit_ranks.get(name)
        if ranks is None:
            continue
        ranks = np.asarray(ranks)
        total = int(ranks.size)
        if total == 0:
            continue
        free = int((ranks == 0).sum())
        learner = int((ranks == 1).sum())
        mature = int((ranks >= 2).sum())
        retired = int((ranks < 0).sum())
        state[name] = {
            "rho0": free / total,
            "rhom": mature / total,
            "free": float(free),
            "learner": float(learner),
            "mature": float(mature),
            "retired": float(retired),
            "total": float(total),
        }
    return state


def compute_consumption(
    prev_ages: Optional[Dict[str, np.ndarray]],
    cur_ages: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Consumption u_i,l = selected_learner / candidate_learner for the last task.

    A neuron is a *candidate learner* if it was free (age 0) before the task and
    a *selected learner* if it is now used (age >= 1). When no previous ages are
    available (task 0), consumption is 0.
    """
    consumption: Dict[str, float] = {}
    if not prev_ages:
        return {name: 0.0 for name in CAPACITY_LAYERS}

    for name in CAPACITY_LAYERS:
        prev = prev_ages.get(name)
        cur = cur_ages.get(name)
        if prev is None or cur is None:
            consumption[name] = 0.0
            continue
        prev = np.asarray(prev)
        cur = np.asarray(cur)
        if prev.shape != cur.shape:
            consumption[name] = 0.0
            continue
        candidate = prev == 0
        n_candidate = int(candidate.sum())
        if n_candidate == 0:
            consumption[name] = 0.0
            continue
        selected = int(((cur >= 1) & candidate).sum())
        consumption[name] = selected / n_candidate
    return consumption


class CapacityController:
    """CANC decision engine (plan section 3, Rules 1-4)."""

    def __init__(self, config: Optional[CANCConfig] = None):
        self.config = config or CANCConfig()

    def pressure(
        self, rho0: float, u: float, novelty: float, val_loss_delta: float = 0.0
    ) -> float:
        c = self.config
        return float(
            c.alpha * (1.0 - rho0)
            + c.beta * u
            + c.gamma * max(0.0, float(val_loss_delta))
            + c.delta * novelty
        )

    def decide_layer(
        self,
        layer: str,
        rho0: float,
        rhom: float,
        u: float,
        novelty: float,
        val_loss_delta: float = 0.0,
    ) -> Dict[str, float]:
        """Return ``{action, kappa}`` for a single layer."""
        c = self.config
        kappa = self.pressure(rho0, u, novelty, val_loss_delta)
        is_low = layer in LOW_LAYERS

        # Rule 4: emergency adapter on a depleted low layer under strong shift.
        if is_low and rho0 < c.epsilon_adapter and novelty >= c.xi_high_novelty:
            return {"action": EMERGENCY_LOW_ADAPTER, "kappa": kappa}

        # Rule 3: add micro-adapter.
        if kappa >= c.kappa_adapter or (
            rho0 < c.epsilon_adapter and (novelty >= c.xi_novelty or u >= c.xi_consume)
        ):
            return {"action": ADD_ADAPTER, "kappa": kappa}

        # Rule 2: low layer depleted but task is still close -> reuse, train high.
        if is_low and rho0 < c.epsilon_free and novelty < c.xi_novelty:
            return {"action": HIGH_LAYER_ONLY, "kappa": kappa}

        # Rule 1: NICE only.
        return {"action": NICE_ONLY, "kappa": kappa}

    def plan_task(
        self,
        capacity_state: Dict[str, Dict[str, float]],
        novelty: float,
        consumption: Optional[Dict[str, float]] = None,
        val_loss_delta: Optional[Dict[str, float] | float] = None,
        is_first_task: bool = False,
    ) -> Dict:
        """Produce per-layer actions and the concrete adapter/freeze plan.

        For task 0 (``is_first_task``) the plan is always NICE_ONLY with no
        adapters (plan section 6, Task 0).
        """
        consumption = consumption or {}
        layers: Dict[str, Dict[str, float]] = {}

        if is_first_task:
            for name, st in capacity_state.items():
                layers[name] = {
                    "action": NICE_ONLY,
                    "kappa": self.pressure(st["rho0"], 0.0, 0.0),
                    "rho0": st["rho0"],
                    "rhom": st["rhom"],
                    "retired": st.get("retired", 0.0),
                    "u": 0.0,
                    "novelty": 0.0,
                }
            return {
                "layers": layers,
                "adapters_to_add": [],
                "freeze_low_layers": False,
                "recycle_layers": [],
                "novelty": 0.0,
            }

        for name, st in capacity_state.items():
            u = float(consumption.get(name, 0.0))
            if isinstance(val_loss_delta, dict):
                d_loss = float(val_loss_delta.get(name, 0.0))
            else:
                d_loss = float(val_loss_delta or 0.0)
            decision = self.decide_layer(
                name, st["rho0"], st["rhom"], u, novelty, d_loss
            )
            layers[name] = {
                **decision,
                "rho0": st["rho0"],
                "rhom": st["rhom"],
                "retired": st.get("retired", 0.0),
                "u": u,
                "novelty": novelty,
                "val_loss_delta": d_loss,
            }

        freeze_low = any(
            info["action"] == HIGH_LAYER_ONLY for name, info in layers.items()
            if name in LOW_LAYERS
        )

        adapters_to_add = self._resolve_adapters(layers)
        recycle_layers = self._resolve_recycling(layers, adapters_to_add)

        return {
            "layers": layers,
            "adapters_to_add": adapters_to_add,
            "recycle_layers": recycle_layers,
            "freeze_low_layers": freeze_low,
            "novelty": novelty,
        }

    def _resolve_adapters(self, layers: Dict[str, Dict[str, float]]) -> List[str]:
        """Map per-layer adapter actions to concrete (enabled) adapter layers.

        - ``ADD_ADAPTER`` on layer ``l`` -> add adapter on ``l`` if enabled.
        - ``EMERGENCY_LOW_ADAPTER`` -> add the highest-priority enabled adapter
          layer (plan Rule 4: fc1 -> gru -> conv3 -> conv2 -> conv1).
        """
        enabled = [l for l in ADAPTER_PRIORITY if l in self.config.enabled_adapter_layers]
        to_add: List[str] = []

        for name, info in layers.items():
            if info["action"] == ADD_ADAPTER and name in enabled and name not in to_add:
                to_add.append(name)

        if any(info["action"] == EMERGENCY_LOW_ADAPTER for info in layers.values()):
            for layer in enabled:
                if layer not in to_add:
                    to_add.append(layer)
                    break

        # Order by priority for deterministic behavior.
        to_add.sort(key=lambda l: ADAPTER_PRIORITY.index(l))
        return to_add

    def _resolve_recycling(
        self, layers: Dict[str, Dict[str, float]], adapters_to_add: List[str]
    ) -> List[str]:
        """Resolve Phase-4 recycling layers after adapter fallback is exhausted."""
        c = self.config
        if not c.enable_recycling:
            return []
        if adapters_to_add:
            return []

        recycle_layers: List[str] = []
        for layer in ADAPTER_PRIORITY:
            info = layers.get(layer)
            if not info:
                continue
            if float(info.get("rho0", 1.0)) > c.epsilon_recycle_free:
                continue
            if float(info.get("kappa", 0.0)) < c.kappa_recycle:
                continue
            if float(info.get("rhom", 0.0)) <= 0.0:
                continue
            info["action"] = GRACEFUL_RECYCLING
            recycle_layers.append(layer)
        return recycle_layers
