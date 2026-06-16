"""
Tests for DeNICE = NICE + CANC + capacity-aware micro-adapters.

Covers the plan's Phase 1 (model/adapter, novelty, CANC) and Phase 3
(capsule, dynamic-K clustering, age-aware aggregation).
"""

from collections import OrderedDict

import numpy as np
import torch

from fed_learning.models.denice_model import (
    DeNICEModel,
    MicroAdapter,
    default_rank,
    adapter_key,
    parse_adapter_key,
)
from fed_learning.strategies.incremental.denice_capacity import (
    CANCConfig,
    CapacityController,
    compute_capacity_state,
    compute_consumption,
    NICE_ONLY,
    ADD_ADAPTER,
    EMERGENCY_LOW_ADAPTER,
    GRACEFUL_RECYCLING,
)
from fed_learning.strategies.incremental.denice_recycling import apply_graceful_recycling
from fed_learning.strategies.incremental.denice_novelty import NoveltyEstimator
from fed_learning.strategies.incremental.nice import select_learner_units
from fed_learning.strategies.incremental import get_incremental_strategy


INPUT_SHAPE = (16, 1)
NUM_CLASSES = 12


def _make_model():
    torch.manual_seed(0)
    return DeNICEModel(INPUT_SHAPE, NUM_CLASSES)


def _dummy_batch(n=8):
    torch.manual_seed(1)
    return torch.randn(n, INPUT_SHAPE[0])


# ---------------------------------------------------------------------------
# Micro-adapter / model
# ---------------------------------------------------------------------------
class TestMicroAdapter:
    def test_rank_rule(self):
        assert default_rank(256) == 16          # 256/16
        assert default_rank(100) == max(4, 6)   # 100/16 = 6
        assert default_rank(8) == 4             # max(4, 0)

    def test_adapter_zero_init_is_identity(self):
        adapter = MicroAdapter(256)
        h = torch.randn(4, 256)
        out = adapter(h)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_adapter_key_roundtrip(self):
        key = adapter_key(3, "fc1", 16, 1)
        assert parse_adapter_key(key) == (3, "fc1", 16, 1)


class TestDeNICEModel:
    def test_forward_shape(self):
        model = _make_model()
        out = model(_dummy_batch())
        assert out.shape == (8, NUM_CLASSES)

    def test_add_adapter_registry(self):
        model = _make_model()
        key = model.add_adapter(context_id=0, layer_name="fc1")
        assert key in model.adapters
        assert key in model.adapter_registry
        assert model.adapter_registry[key]["layer_name"] == "fc1"
        assert model.adapter_registry[key]["rank"] == default_rank(256)
        assert model.has_adapter(0, "fc1")
        # Idempotent.
        assert model.add_adapter(context_id=0, layer_name="fc1") == key
        assert len(model.adapters) == 1

    def test_adapter_zero_residual_keeps_output(self):
        model = _make_model()
        model.eval()
        x = _dummy_batch()
        before = model(x)
        model.add_adapter(context_id=0, layer_name="fc1", set_active=True)
        after = model(x)
        # U is zero-initialized -> adapter is a no-op until trained.
        assert torch.allclose(before, after, atol=1e-5)

    def test_active_adapter_changes_output_once_trained(self):
        model = _make_model()
        model.eval()
        x = _dummy_batch()
        key = model.add_adapter(context_id=1, layer_name="fc1", set_active=True)
        with torch.no_grad():
            model.adapters[key].U.weight.fill_(0.1)
        changed = model(x)
        model.clear_active_adapters()
        base = model(x)
        assert not torch.allclose(base, changed, atol=1e-4)

    def test_conv3_and_gru_adapters_forward(self):
        model = _make_model()
        model.add_adapter(context_id=0, layer_name="conv3", set_active=True)
        model.add_adapter(context_id=0, layer_name="gru", set_active=True)
        out = model(_dummy_batch())
        assert out.shape == (8, NUM_CLASSES)

    def test_set_active_context(self):
        model = _make_model()
        model.add_adapter(context_id=2, layer_name="fc1", set_active=False)
        model.add_adapter(context_id=2, layer_name="gru", set_active=False)
        model.set_active_context(2)
        assert "fc1" in model.active_adapters
        assert "gru" in model.active_adapters
        model.set_active_context(99)  # no adapters for this context
        assert model.active_adapters == {}

    def test_adapters_in_state_dict(self):
        model = _make_model()
        model.add_adapter(context_id=0, layer_name="fc1")
        sd = model.state_dict()
        assert any("adapters" in k for k in sd)

    def test_retire_and_revive_recycled_neurons(self):
        model = _make_model()
        model.unit_ranks["fc1"][:3] = 2
        retired = model.retire_neurons("fc1", [0, 1], task_id=2)
        assert retired == [0, 1]
        assert model.unit_ranks["fc1"][0] == -1
        assert model.weight_masks["fc1"][0].sum().item() == 0

        revived = model.revive_retired_neurons(task_id=3, grace_tasks=1)
        assert revived == {"fc1": [0, 1]}
        assert model.unit_ranks["fc1"][0] == 0
        assert model.weight_masks["fc1"][0].sum().item() == model.weight_masks["fc1"].shape[1]

    def test_select_learner_units_does_not_select_retired(self):
        model = _make_model()
        model.unit_ranks["fc1"][:] = 0
        model.unit_ranks["fc1"][0] = -1
        select_learner_units(model, tau=1.0, data=_dummy_batch(8))
        assert model.unit_ranks["fc1"][0] == -1
        assert np.all(model.unit_ranks["fc1"][1:] == 1)

    def test_apply_graceful_recycling_retires_low_importance_mature(self):
        model = _make_model()
        model.unit_ranks["fc1"][:] = 2
        plan = {"recycle_layers": ["fc1"]}
        summary = apply_graceful_recycling(
            model,
            _dummy_batch(12),
            task_id=1,
            canc_plan=plan,
            config={
                "denice_enable_recycling": True,
                "denice_recycle_ratio": 0.01,
                "denice_recycle_min": 2,
                "denice_recycle_max_per_layer": 3,
            },
        )
        assert summary["total_retired"] == 3
        assert int((model.unit_ranks["fc1"] == -1).sum()) == 3


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------
class TestNovelty:
    def test_first_task_is_fully_novel(self):
        est = NoveltyEstimator()
        model = _make_model()
        x = _dummy_batch(16)
        est.calibrate_thresholds(model, x)
        info = est.compute_novelty(model, x)
        assert info["novelty"] == 1.0  # no history yet

    def test_identical_prototype_has_low_novelty(self):
        est = NoveltyEstimator()
        model = _make_model()
        x = _dummy_batch(16)
        est.calibrate_thresholds(model, x)
        proto = est.compute_prototype(model, x)
        est.store_prototype(0, proto)
        info = est.novelty_from_prototype(proto, exclude_task=None)
        # Same prototype -> cosine ~1 -> novelty ~0.
        assert info["novelty"] < 0.05


# ---------------------------------------------------------------------------
# CANC
# ---------------------------------------------------------------------------
class TestCANC:
    def test_capacity_state(self):
        model = _make_model()
        model.unit_ranks["fc1"][:] = 0
        model.unit_ranks["fc1"][:64] = 2  # 64/256 mature
        state = compute_capacity_state(model)
        assert abs(state["fc1"]["rhom"] - 64 / 256) < 1e-6
        assert abs(state["fc1"]["rho0"] - 192 / 256) < 1e-6

    def test_first_task_all_nice_only(self):
        ctrl = CapacityController(CANCConfig())
        model = _make_model()
        state = compute_capacity_state(model)
        plan = ctrl.plan_task(state, novelty=1.0, is_first_task=True)
        assert plan["adapters_to_add"] == []
        assert all(info["action"] == NICE_ONLY for info in plan["layers"].values())

    def test_depleted_high_novelty_adds_adapter(self):
        ctrl = CapacityController(CANCConfig(enabled_adapter_layers=["fc1"]))
        # fc1 depleted (rho0 ~ 0), high novelty -> ADD_ADAPTER.
        decision = ctrl.decide_layer("fc1", rho0=0.02, rhom=0.9, u=0.0, novelty=0.6)
        assert decision["action"] == ADD_ADAPTER

    def test_emergency_low_adapter(self):
        ctrl = CapacityController(CANCConfig())
        decision = ctrl.decide_layer("conv1", rho0=0.0, rhom=1.0, u=0.0, novelty=0.7)
        assert decision["action"] == EMERGENCY_LOW_ADAPTER

    def test_plan_resolves_only_enabled_layers(self):
        ctrl = CapacityController(CANCConfig(enabled_adapter_layers=["fc1"]))
        layers = {
            "fc1": {"action": ADD_ADAPTER},
            "gru": {"action": ADD_ADAPTER},
        }
        adapters = ctrl._resolve_adapters(layers)
        assert adapters == ["fc1"]  # gru not enabled in MVP

    def test_config_parses_phase2_adapter_layers_from_string(self):
        cfg = CANCConfig.from_dict({"denice_adapter_layers": "fc1,gru,conv3"})
        assert cfg.enabled_adapter_layers == ["fc1", "gru", "conv3"]

    def test_phase2_plan_resolves_fc1_gru_conv3(self):
        ctrl = CapacityController(CANCConfig(enabled_adapter_layers=["fc1", "gru", "conv3"]))
        layers = {
            "fc1": {"action": ADD_ADAPTER},
            "gru": {"action": ADD_ADAPTER},
            "conv3": {"action": ADD_ADAPTER},
            "conv1": {"action": EMERGENCY_LOW_ADAPTER},
        }
        adapters = ctrl._resolve_adapters(layers)
        assert adapters == ["fc1", "gru", "conv3"]

    def test_consumption(self):
        prev = {"fc1": np.zeros(256, dtype=np.int32)}
        cur = {"fc1": np.zeros(256, dtype=np.int32)}
        cur["fc1"][:128] = 1  # half consumed
        u = compute_consumption(prev, cur)
        assert abs(u["fc1"] - 0.5) < 1e-6

    def test_phase4_recycling_requires_enable_flag_and_no_adapter(self):
        ctrl = CapacityController(
            CANCConfig(
                enable_recycling=True,
                enabled_adapter_layers=[],
                kappa_recycle=0.7,
            )
        )
        state = {
            "fc1": {
                "rho0": 0.0,
                "rhom": 1.0,
                "free": 0,
                "learner": 0,
                "mature": 256,
                "retired": 0,
                "total": 256,
            }
        }
        plan = ctrl.plan_task(state, novelty=1.0, consumption={"fc1": 0.0})
        assert plan["recycle_layers"] == ["fc1"]
        assert plan["layers"]["fc1"]["action"] == GRACEFUL_RECYCLING

    def test_phase4_recycling_is_off_by_default(self):
        ctrl = CapacityController(CANCConfig(enabled_adapter_layers=[]))
        state = {
            "fc1": {
                "rho0": 0.0,
                "rhom": 1.0,
                "free": 0,
                "learner": 0,
                "mature": 256,
                "retired": 0,
                "total": 256,
            }
        }
        plan = ctrl.plan_task(state, novelty=1.0, consumption={"fc1": 0.0})
        assert plan["recycle_layers"] == []


# ---------------------------------------------------------------------------
# Strategy registration
# ---------------------------------------------------------------------------
class TestRegistration:
    def test_denice_trainer_registered(self):
        trainer = get_incremental_strategy("denice")
        assert type(trainer).__name__ == "DeNICETrainer"
        assert hasattr(trainer, "canc_config")


# ---------------------------------------------------------------------------
# Decentralized: capsule + clustering + aggregation
# ---------------------------------------------------------------------------
class TestDecentralized:
    def _capsule(self, client_id, labels, proto_seed):
        from fed_learning.strategies.decentralized import ContextCapsule

        rng = np.random.default_rng(proto_seed)
        proto = {name: rng.random(8) for name in ["conv1", "conv2", "conv3", "gru"]}
        age = {name: (rng.random(8) > 0.5).astype(np.float32) for name in ["conv1", "conv2", "conv3", "gru"]}
        imp = {name: rng.random(8) for name in ["conv1", "conv2", "conv3", "gru"]}
        cap = {name: {"young": 0.5, "learner": 0.2, "mature": 0.3} for name in ["conv1", "conv2", "conv3", "gru"]}
        return ContextCapsule(
            client_id=client_id,
            task_id=0,
            round_id=0,
            activation_prototypes=proto,
            age_mask=age,
            neuron_importance=imp,
            capacity_histogram=cap,
            label_histogram={c: 1.0 for c in labels},
            label_set=labels,
            sample_count=100,
            reliability=0.9,
            context_detector_summary={},
        )

    def test_build_capsule_from_model(self):
        from fed_learning.strategies.decentralized import build_context_capsule

        model = _make_model()
        x = _dummy_batch(16)
        cap = build_context_capsule(
            model,
            x,
            client_id=0,
            task_id=0,
            round_id=0,
            label_histogram={0: 0.5, 1: 0.5},
            label_set=[0, 1],
            sample_count=16,
            reliability=0.8,
        )
        assert cap.proto_vector().size > 0
        assert set(cap.activation_prototypes.keys()) == {"conv1", "conv2", "conv3", "gru"}

    def test_dynamic_ap_two_groups(self):
        from fed_learning.strategies.decentralized import dynamic_ap_cluster

        # Two clear groups: clients {0,1} share label set A, {2,3} share label set B.
        caps = [
            self._capsule(0, [0, 1], proto_seed=1),
            self._capsule(1, [0, 1], proto_seed=1),
            self._capsule(2, [5, 6], proto_seed=99),
            self._capsule(3, [5, 6], proto_seed=99),
        ]
        result = dynamic_ap_cluster(caps)
        assert result["K_t"] >= 1
        assert result["labels"].shape == (4,)

    def test_aggregation_weights_normalize(self):
        from fed_learning.strategies.decentralized import aggregation_weights

        alpha = aggregation_weights([1.0, 1.0], [100, 100], [1.0, 1.0])
        assert abs(alpha.sum() - 1.0) < 1e-6

    def test_aggregation_weights_collapse_keeps_self(self):
        from fed_learning.strategies.decentralized import aggregation_weights

        alpha = aggregation_weights([0.0, 0.0], [0, 0], [0, 0], self_index=0)
        assert alpha[0] == 1.0

    def test_age_aware_aggregate_protects_mature(self):
        from fed_learning.strategies.decentralized import age_aware_aggregate

        target = OrderedDict({"fc2.weight": torch.zeros(4, 3)})
        ages = {"fc2": np.array([2, 2, 0, 0])}  # first 2 neurons mature
        delta = OrderedDict({"fc2.weight": torch.ones(4, 3)})
        out = age_aware_aggregate(target, ages, [delta], np.array([1.0]))
        # Mature rows unchanged (0), plastic rows updated (1).
        assert torch.allclose(out["fc2.weight"][:2], torch.zeros(2, 3))
        assert torch.allclose(out["fc2.weight"][2:], torch.ones(2, 3))

    def test_aggregate_adapters_matches_key(self):
        from fed_learning.strategies.decentralized import aggregate_adapters

        key = adapter_key(0, "fc1", 16, 1)
        target = {key: OrderedDict({"U.weight": torch.zeros(8, 4)})}
        nb_match = {key: OrderedDict({"U.weight": torch.ones(8, 4)})}
        nb_other = {adapter_key(9, "fc1", 16, 1): OrderedDict({"U.weight": torch.ones(8, 4)})}
        merged = aggregate_adapters(target, [nb_match, nb_other], [1.0, 1.0])
        # Average of target(0) and matching neighbor(1) = 0.5; non-matching skipped.
        assert torch.allclose(merged[key]["U.weight"], torch.full((8, 4), 0.5))


# ---------------------------------------------------------------------------
# Training smoke test (one short task with adapter)
# ---------------------------------------------------------------------------
class TestTrainingSmoke:
    def test_denice_client_one_phase(self):
        from fed_learning.clients.denice_client import DeNICEClient
        from fed_learning.strategies.incremental.denice import DeNICETrainer

        torch.manual_seed(0)
        n = 64
        X = torch.randn(n, INPUT_SHAPE[0])
        y = torch.randint(0, 3, (n,))
        model = _make_model()
        for c in range(3):
            model.unit_ranks["fc2"][c] = 1
        model.add_adapter(context_id=0, layer_name="fc1", set_active=True)

        client = DeNICEClient(0, X, y, max_phases=1, phase_epochs=1, tau=1.0)
        client.setup_for_gpu(model, "cpu")
        trainer = DeNICETrainer(max_phases=1, phase_epochs=1)

        result = client.train(
            trainer=trainer, epochs=1, batch_size=16, lr=1e-3, is_last_task=True
        )
        assert "adapter_registry" in result
        assert result["adapter_param_count"] > 0
        assert np.isfinite(result["loss"])
