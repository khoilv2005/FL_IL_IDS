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
from fed_learning.training.denice_delta_checkpoint import (
    cpu_client_model_states,
    load_denice_checkpoint,
    save_delta_round_checkpoint,
    save_task_base_checkpoint,
)


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

    def test_delta_checkpoint_reconstructs_round_state(self, tmp_path):
        model = _make_model()
        models = {7: model}
        client_ids = [7]
        config = {
            "input_shape": INPUT_SHAPE,
            "num_classes": NUM_CLASSES,
            "total_classes": NUM_CLASSES,
            "algorithm": "denice",
            "mode": "decentralized",
        }
        base_path = tmp_path / "checkpoint_task_0_base.pt"
        round_path = tmp_path / "checkpoint_task_0_round_0.pt"
        save_task_base_checkpoint(
            str(base_path),
            task_id=0,
            client_ids=client_ids,
            models=models,
            client_algorithm_states={7: {"denice": {}}},
            config=config,
            seen_classes=list(range(NUM_CLASSES)),
        )
        previous = cpu_client_model_states(client_ids, models)
        with torch.no_grad():
            first_param = next(model.parameters())
            first_param.add_(0.25)
        expected = cpu_client_model_states(client_ids, models)[7]
        save_delta_round_checkpoint(
            str(round_path),
            task_id=0,
            round_id=0,
            base_path=str(base_path),
            previous_round_path=None,
            client_ids=client_ids,
            models=models,
            previous_model_states=previous,
            client_algorithm_states={7: {"denice": {}}},
            config=config,
            seen_classes=list(range(NUM_CLASSES)),
            cluster={},
            metrics={},
        )

        reconstructed = load_denice_checkpoint(str(round_path))
        actual = reconstructed["client_model_states"][7]
        assert reconstructed["checkpoint_type"] == "denice_reconstructed_round"
        assert actual.keys() == expected.keys()
        for key in expected:
            if torch.is_floating_point(expected[key]):
                assert torch.allclose(actual[key].float(), expected[key].float(), atol=2e-3)
            else:
                assert torch.equal(actual[key], expected[key])

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
            canc_plan={**plan, "old_metric_delta": 0.0},
            config={
                "denice_enable_recycling": True,
                "denice_recycle_ratio": 0.01,
                "denice_recycle_min": 2,
                "denice_recycle_max_per_layer": 3,
                "denice_recycle_usage_recent_threshold": 1.0,
            },
        )
        assert summary["total_retired"] == 3
        assert int((model.unit_ranks["fc1"] == -1).sum()) == 3

    def test_recycling_requires_old_metric_check(self):
        model = _make_model()
        model.unit_ranks["fc1"][:] = 2
        plan = {"recycle_layers": ["fc1"]}
        summary = apply_graceful_recycling(
            model,
            _dummy_batch(12),
            task_id=1,
            canc_plan=plan,
            config={"denice_enable_recycling": True},
        )
        assert summary["blocked"] is True
        assert summary["total_retired"] == 0
        assert int((model.unit_ranks["fc1"] == -1).sum()) == 0


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

    def test_capacity_pressure_includes_validation_loss_delta(self):
        ctrl = CapacityController(
            CANCConfig(alpha=0.0, beta=0.0, gamma=0.5, delta=0.0)
        )
        decision = ctrl.decide_layer(
            "fc1", rho0=1.0, rhom=0.0, u=0.0, novelty=0.0, val_loss_delta=2.0
        )
        assert abs(decision["kappa"] - 1.0) < 1e-6

    def test_runner_computes_runtime_validation_loss_delta(self):
        from fed_learning.training.decentralized_denice_il import _compute_val_loss_delta

        model = _make_model()
        ref_bank = {0: (_dummy_batch(6), torch.zeros(6, dtype=torch.long))}
        delta = _compute_val_loss_delta(
            model,
            ref_bank,
            baseline=0.0,
            device=torch.device("cpu"),
            batch_size=4,
        )
        assert delta > 0.0


    def test_eval_client_selection_prefers_full_seen_class_coverage(self):
        from fed_learning.training.decentralized_denice_il import _select_eval_clients

        class Detector:
            def __init__(self, classes):
                self.episode_classes = {0: classes}

        detectors = {
            1: Detector([6, 7, 8, 9, 10, 11]),
            2: Detector([6, 7, 8, 9, 10, 11]),
            3: Detector([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
            4: Detector([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        }
        selected = _select_eval_clients(
            [1, 2, 3, 4],
            2,
            context_detectors=detectors,
            seen_classes=list(range(12)),
            require_full_coverage=True,
        )
        assert selected == [3, 4]

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
            labels=torch.tensor([0] * 8 + [1] * 8),
        )
        assert cap.proto_vector().size > 0
        assert set(cap.activation_prototypes.keys()) == {"conv1", "conv2", "conv3", "gru"}
        assert set(cap.class_activation_prototypes.keys()) == {0, 1}
        assert set(cap.class_activation_prototypes[0].keys()) == {"conv1", "conv2", "conv3", "gru"}

    def test_capsule_context_detector_summary_contains_q_fields(self):
        from fed_learning.servers.nice_server import ContextDetector
        from fed_learning.strategies.decentralized import build_context_capsule

        model = _make_model()
        model.eval()
        det = ContextDetector(memo_per_class=10)
        det.episode_classes[0] = [0, 1]
        det.episode_classes[1] = [2, 3]
        det.push_activations(model, _dummy_batch(12), episode=0)
        det.push_activations(model, _dummy_batch(12) + 2.0, episode=1)
        det.train_models(1)
        cap = build_context_capsule(
            model,
            _dummy_batch(8),
            client_id=0,
            task_id=0,
            round_id=0,
            label_histogram={0: 1.0},
            label_set=[0],
            sample_count=8,
            reliability=0.8,
            context_detector=det,
            labels=torch.zeros(8, dtype=torch.long),
        )
        q = cap.context_detector_summary
        assert q["episode_classes"] == {0: [0, 1], 1: [2, 3]}
        assert q["memory_counts"][0] == 12
        assert set(q["threshold_layers"]) == {"conv1", "conv2", "conv3", "gru"}
        assert "activation_stats" in q
        assert "threshold_values" in q
        assert q["learners"][0]["coef_hash"]

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

    def test_class_prototype_similarity_requires_shared_class(self):
        from fed_learning.strategies.decentralized import (
            ContextCapsule,
            class_prototype_similarity,
        )

        proto_a = {name: np.ones(4, dtype=np.float32) for name in ["conv1", "conv2", "conv3", "gru"]}
        proto_b = {name: np.ones(4, dtype=np.float32) for name in ["conv1", "conv2", "conv3", "gru"]}
        base = dict(
            task_id=0,
            round_id=0,
            activation_prototypes={},
            age_mask={name: np.ones(4, dtype=np.float32) for name in ["conv1", "conv2", "conv3", "gru"]},
            neuron_importance={name: np.ones(4, dtype=np.float32) for name in ["conv1", "conv2", "conv3", "gru"]},
            capacity_histogram={name: {"young": 0.0, "learner": 0.0, "mature": 1.0} for name in ["conv1", "conv2", "conv3", "gru"]},
            sample_count=100,
            reliability=1.0,
            context_detector_summary={},
        )
        cap_a = ContextCapsule(
            client_id=0,
            label_histogram={0: 1.0},
            label_set=[0],
            class_activation_prototypes={0: proto_a},
            **base,
        )
        cap_b = ContextCapsule(
            client_id=1,
            label_histogram={1: 1.0},
            label_set=[1],
            class_activation_prototypes={1: proto_b},
            **base,
        )
        cap_c = ContextCapsule(
            client_id=2,
            label_histogram={0: 1.0},
            label_set=[0],
            class_activation_prototypes={0: proto_b},
            **base,
        )
        assert class_prototype_similarity(cap_a, cap_b) == 0.0
        assert class_prototype_similarity(cap_a, cap_c) > 0.99

    def test_aggregation_weights_normalize(self):
        from fed_learning.strategies.decentralized import aggregation_weights

        alpha = aggregation_weights([1.0, 1.0], [100, 100], [1.0, 1.0])
        assert abs(alpha.sum() - 1.0) < 1e-6

    def test_aggregation_weights_collapse_keeps_self(self):
        from fed_learning.strategies.decentralized import aggregation_weights

        alpha = aggregation_weights([0.0, 0.0], [0, 0], [0, 0], self_index=0)
        assert alpha[0] == 1.0

    def test_aggregation_weights_self_floor_limits_overwrite(self):
        from fed_learning.strategies.decentralized import aggregation_weights

        alpha = aggregation_weights(
            [1.0, 1.0],
            [10, 1_000_000],
            [1.0, 1.0],
            self_index=0,
            count_transform="log",
            self_floor=0.25,
        )
        assert alpha[0] >= 0.25
        assert abs(alpha.sum() - 1.0) < 1e-6

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

    def test_aggregate_adapters_preserves_self_alpha(self):
        """A target alpha of .25 and neighbor alpha of .75 must yield 7.5."""
        from fed_learning.strategies.decentralized import aggregate_adapters

        key = adapter_key(0, "fc1", 16, 1)
        target = {key: OrderedDict({"U.weight": torch.zeros(1, 1)})}
        neighbor = {key: OrderedDict({"U.weight": torch.full((1, 1), 10.0)})}
        merged = aggregate_adapters(
            target, [neighbor], [0.75], target_weight=0.25
        )
        assert torch.allclose(merged[key]["U.weight"], torch.full((1, 1), 7.5))

    def test_consensus_age_merge_does_not_union_disjoint_peer_maturity(self):
        from fed_learning.strategies.decentralized import merge_neuron_ages

        target = {"fc1": np.array([0, 0, 0, 0])}
        neighbors = [
            {"fc1": np.array([2, 0, 0, 0])},
            {"fc1": np.array([0, 2, 0, 0])},
            {"fc1": np.array([0, 0, 2, 0])},
        ]
        merged = merge_neuron_ages(
            target,
            neighbors,
            neighbor_weights=[0.25, 0.25, 0.25],
            consensus_threshold=0.5,
        )
        assert np.array_equal(merged["fc1"], target["fc1"])

    def test_consensus_age_merge_promotes_only_supported_maturity(self):
        from fed_learning.strategies.decentralized import merge_neuron_ages

        target = {"fc1": np.array([2, 0, 0])}
        neighbors = [
            {"fc1": np.array([0, 2, 0])},
            {"fc1": np.array([0, 2, 2])},
        ]
        merged = merge_neuron_ages(
            target,
            neighbors,
            neighbor_weights=[0.30, 0.30],
            consensus_threshold=0.50,
        )
        assert np.array_equal(merged["fc1"], np.array([2, 2, 0]))

    def test_invalid_cluster_without_prior_valid_is_self_only(self):
        from fed_learning.training.decentralized_denice_il import (
            _effective_cluster_assignment,
        )

        result = {"labels": np.array([5, 5, 9]), "valid": False, "edges": np.ones((3, 3))}
        labels, edges, policy, reason, state = _effective_cluster_assignment(
            result, [10, 20, 30]
        )
        assert np.array_equal(labels, np.array([0, 1, 2]))
        assert edges is None
        assert policy == "self_only"
        assert reason is not None
        assert state is None

    def test_invalid_cluster_reuses_only_compatible_previous_valid_state(self):
        from fed_learning.training.decentralized_denice_il import (
            _effective_cluster_assignment,
        )

        result = {"labels": np.array([9, 9, 9]), "valid": False, "edges": np.ones((3, 3))}
        prior = {"client_ids": (10, 20, 30), "labels": np.array([0, 0, 1])}
        labels, edges, policy, _reason, _state = _effective_cluster_assignment(
            result, [10, 20, 30], prior
        )
        assert np.array_equal(labels, np.array([0, 0, 1]))
        assert edges is None
        assert policy == "previous_valid"

        labels, _edges, policy, _reason, _state = _effective_cluster_assignment(
            result, [20, 10, 30], prior
        )
        assert np.array_equal(labels, np.array([0, 1, 2]))
        assert policy == "self_only"

    def test_bootstrap_clone_preserves_model_and_adapter_structure(self):
        from fed_learning.training.decentralized_denice_il import (
            _bootstrap_denice_model,
            _param_max_abs_diff,
            _state_dict,
        )

        source = _make_model()
        source.add_adapter(0, "fc1", set_active=False)
        with torch.no_grad():
            source.fc1.weight.fill_(0.125)
        cloned = _bootstrap_denice_model(
            source,
            {"input_shape": INPUT_SHAPE, "num_classes": NUM_CLASSES},
            torch.device("cpu"),
        )
        assert cloned.get_adapter_registry_state() == source.get_adapter_registry_state()
        assert _param_max_abs_diff(_state_dict(cloned), _state_dict(source)) == 0.0

    def test_rejoining_catch_up_keeps_receiver_mature_rows(self):
        from fed_learning.training.decentralized_denice_il import (
            _catch_up_rejoining_model,
        )

        target = _make_model()
        source = _make_model()
        target.unit_ranks["fc1"][0] = 2
        with torch.no_grad():
            target.fc1.weight.zero_()
            source.fc1.weight.fill_(1.0)
        _catch_up_rejoining_model(target, source, torch.device("cpu"))
        assert torch.allclose(target.fc1.weight[0], torch.zeros_like(target.fc1.weight[0]))
        assert torch.allclose(target.fc1.weight[1], torch.ones_like(target.fc1.weight[1]))

    def test_invalid_cluster_self_only_aggregation_smoke(self, monkeypatch):
        """Invalid AP labels must not alter peer state or spread peer maturity."""
        import fed_learning.training.decentralized_denice_il as runner

        models = {0: _make_model(), 1: _make_model()}
        models[1].unit_ranks["fc1"][0] = 2
        before = {
            cid: OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())
            for cid, model in models.items()
        }
        capsules = {
            0: self._capsule(0, [0, 1], 1),
            1: self._capsule(1, [0, 1], 1),
        }
        monkeypatch.setattr(
            runner,
            "dynamic_ap_cluster",
            lambda *_args, **_kwargs: {
                "labels": np.array([0, 0]), "edges": np.ones((2, 2), dtype=int),
                "valid": False, "K_t": 1, "silhouette": 0.1,
                "similarity": np.eye(2),
            },
        )
        summary = runner._aggregate_round(
            client_ids=[0, 1],
            models=models,
            capsules=capsules,
            config={"denice_age_merge_policy": "consensus"},
            device=torch.device("cpu"),
        )
        assert summary["effective_policy"] == "self_only"
        assert summary["groups"] == {0: [0], 1: [1]}
        assert all(
            torch.allclose(value, before[cid][name])
            for cid, model in models.items()
            for name, value in model.state_dict().items()
        )
        assert models[0].unit_ranks["fc1"][0] == 0

    def test_capacity_reserve_keeps_current_task_units_plastic(self):
        from fed_learning.training.decentralized_denice_il import (
            _enforce_minimum_free_capacity,
        )

        model = _make_model()
        start = model.get_neuron_ages_state()
        model.unit_ranks["fc1"][:] = 1
        released = _enforce_minimum_free_capacity(model, start, 0.10)
        assert released["fc1"] >= int(np.ceil(0.10 * len(model.unit_ranks["fc1"])))
        assert float((model.unit_ranks["fc1"] == 0).mean()) >= 0.10

    def test_coordinate_median_ignores_outlier_neighbor(self):
        """Robust aggregation (Đề xuất §7): median is unmoved by one outlier."""
        from fed_learning.strategies.decentralized import (
            AggregationConfig,
            age_aware_aggregate,
        )

        target = OrderedDict({"fc2.weight": torch.zeros(4, 3)})
        ages = {"fc2": np.array([0, 0, 0, 0])}  # all plastic
        deltas = [
            OrderedDict({"fc2.weight": torch.ones(4, 3)}),
            OrderedDict({"fc2.weight": torch.ones(4, 3)}),
            OrderedDict({"fc2.weight": torch.full((4, 3), 100.0)}),  # Byzantine
        ]
        cfg = AggregationConfig(method="coordinate_median")
        out = age_aware_aggregate(target, ages, deltas, np.array([1.0, 1.0, 1.0]), cfg)
        # Median of {1, 1, 100} = 1, outlier ignored.
        assert torch.allclose(out["fc2.weight"], torch.ones(4, 3))

    def test_trimmed_mean_drops_extremes(self):
        from fed_learning.strategies.decentralized import (
            AggregationConfig,
            age_aware_aggregate,
        )

        target = OrderedDict({"fc2.weight": torch.zeros(2, 2)})
        ages = {"fc2": np.array([0, 0])}
        vals = [-50.0, 1.0, 1.0, 1.0, 50.0]
        deltas = [OrderedDict({"fc2.weight": torch.full((2, 2), v)}) for v in vals]
        cfg = AggregationConfig(method="trimmed_mean", trim_ratio=0.2)
        out = age_aware_aggregate(
            target, ages, deltas, np.ones(len(vals)), cfg
        )
        # Trim 1 each side -> mean of {1,1,1} = 1.
        assert torch.allclose(out["fc2.weight"], torch.ones(2, 2))

    def test_weighted_mean_is_default_and_unchanged(self):
        """Default path must stay the faithful alpha-weighted sum."""
        from fed_learning.strategies.decentralized import (
            AggregationConfig,
            age_aware_aggregate,
        )

        target = OrderedDict({"fc2.weight": torch.zeros(2, 2)})
        ages = {"fc2": np.array([0, 0])}
        deltas = [
            OrderedDict({"fc2.weight": torch.full((2, 2), 2.0)}),
            OrderedDict({"fc2.weight": torch.full((2, 2), 4.0)}),
        ]
        out = age_aware_aggregate(
            target, ages, deltas, np.array([0.25, 0.75]), AggregationConfig()
        )
        # 0.25*2 + 0.75*4 = 3.5
        assert torch.allclose(out["fc2.weight"], torch.full((2, 2), 3.5))

    def test_collaboration_group_respects_context_edges(self):
        """G_i = {j | same cluster AND s_ij > delta} (Đề xuất §6)."""
        from fed_learning.strategies.decentralized import collaboration_group

        labels = np.array([0, 0, 0])  # all same cluster
        # Client 0 is only context-connected to client 1 (not 2).
        neighbors = [1]
        group = collaboration_group(0, labels, neighbors=neighbors)
        assert group == [0, 1]
        # Without edge filtering the whole cluster collaborates.
        assert collaboration_group(0, labels) == [0, 1, 2]

    def test_centroid_distance_marks_far_context_outlier(self):
        from fed_learning.training.decentralized_denice_il import _cosine_distance_to_centroid

        d = _cosine_distance_to_centroid(
            {
                0: np.array([1.0, 0.0]),
                1: np.array([1.0, 0.0]),
                2: np.array([-1.0, 0.0]),
            }
        )
        assert d[2] > d[0]


# ---------------------------------------------------------------------------
# Shared/global context detector (route-accuracy fix, plan/protocol 15 + 23.3)
# ---------------------------------------------------------------------------
class TestSharedContextDetector:
    def _episode_data(self, n, shift):
        torch.manual_seed(100 + int(shift * 10))
        return torch.randn(n, INPUT_SHAPE[0]) + float(shift)

    def _binary(self, detector, model, data):
        acts = {
            name: np.asarray(act.detach().cpu().tolist())
            for name, act in model.get_context_activations_per_sample(data).items()
        }
        return detector.binarize_layer_activations(acts)

    def test_threshold_calibrated_on_first_push_even_if_episode_nonzero(self):
        """Late-joining client (first task >= 1) must still get thresholds."""
        from fed_learning.servers.nice_server import ContextDetector

        model = _make_model()
        model.eval()
        det = ContextDetector(memo_per_class=10)
        det.episode_classes[2] = [4, 5]
        # Client's FIRST push is episode 2, not 0.
        det.push_activations(model, self._episode_data(16, 3.0), episode=2)
        assert det.binarize_thresholds is not None
        assert set(det.binarize_thresholds) == {"conv1", "conv2", "conv3", "gru"}

    def test_local_detector_cannot_route_missing_episode(self):
        """Reproduces Claude's finding: a client missing an episode misroutes it."""
        from fed_learning.servers.nice_server import ContextDetector

        model = _make_model()
        model.eval()
        ep1 = self._episode_data(40, 5.0)

        det_a = ContextDetector(memo_per_class=20)
        det_a.episode_classes[0] = [0, 1]
        det_a.push_activations(model, self._episode_data(40, -5.0), episode=0)
        det_a.train_models(0)

        # det_a never saw episode 1 -> it can only ever predict episode 0.
        preds = det_a.predict_episodes_batch(self._binary(det_a, model, ep1))
        assert set(int(p) for p in np.unique(preds)).issubset({0})

    def test_pooled_detector_covers_and_routes_all_episodes(self):
        from fed_learning.servers.nice_server import (
            ContextDetector,
            build_pooled_context_detector,
        )

        model = _make_model()
        model.eval()
        ep0 = self._episode_data(60, -5.0)
        ep1 = self._episode_data(60, 5.0)

        det_a = ContextDetector(memo_per_class=30)
        det_a.episode_classes[0] = [0, 1]
        det_a.push_activations(model, ep0, episode=0)
        det_a.train_models(0)

        det_b = ContextDetector(memo_per_class=30)
        det_b.episode_classes[1] = [2, 3]
        det_b.push_activations(model, ep1, episode=1)
        det_b.train_models(1)

        pooled = build_pooled_context_detector([det_a, det_b], memo_per_class=30)
        assert set(pooled.episode_classes) == {0, 1}
        assert set(pooled.activation_memory) == {0, 1}

        pred0 = pooled.predict_episodes_batch(self._binary(pooled, model, ep0))
        pred1 = pooled.predict_episodes_batch(self._binary(pooled, model, ep1))
        # Majority of each episode routes to its own id (separable inputs).
        assert float((pred0 == 0).mean()) > 0.6
        assert float((pred1 == 1).mean()) > 0.6

    def test_strict_pooling_rejects_mismatched_calibration(self):
        from fed_learning.servers.nice_server import (
            ContextDetector,
            build_pooled_context_detector,
        )

        model = _make_model()
        model.eval()
        det_a = ContextDetector(memo_per_class=10)
        det_a.episode_classes[0] = [0]
        det_a.push_activations(model, self._episode_data(12, -3.0), episode=0)
        det_b = ContextDetector(memo_per_class=10)
        det_b.episode_classes[1] = [1]
        det_b.push_activations(model, self._episode_data(12, 3.0), episode=1)

        assert det_a.calibration_signature() != det_b.calibration_signature()
        assert build_pooled_context_detector(
            [det_a, det_b], require_compatible_calibration=True
        ) is None

    def test_context_detector_checkpoint_preserves_router_state(self):
        from fed_learning.servers.nice_server import ContextDetector
        from fed_learning.training.checkpoint_state import (
            restore_context_detector,
            snapshot_context_detector,
        )

        model = _make_model()
        model.eval()
        source = ContextDetector(
            memo_per_class=10,
            router_mode="multiclass",
            calibration_provenance="reference-encoder-v1",
        )
        source.episode_classes[0] = [0, 1]
        source.push_activations(model, self._episode_data(16, -2.0), episode=0)
        source.train_models(0)
        state = snapshot_context_detector(source)

        restored = ContextDetector(memo_per_class=1)
        restore_context_detector(restored, state)
        assert restored.router_mode == "multiclass"
        assert restored.calibration_provenance == "reference-encoder-v1"
        assert restored.multiclass_episodes == source.multiclass_episodes

    def test_cluster_context_bank_uses_only_group_clients(self):
        from fed_learning.servers.nice_server import ContextDetector
        from fed_learning.training.decentralized_denice_il import (
            _build_shared_context_detector,
        )

        model = _make_model()
        model.eval()

        det_a = ContextDetector(memo_per_class=20)
        det_a.episode_classes[0] = [0, 1]
        det_a.push_activations(model, self._episode_data(30, -5.0), episode=0)
        det_a.train_models(0)

        det_b = ContextDetector(memo_per_class=20)
        det_b.episode_classes[1] = [2, 3]
        det_b.push_activations(model, self._episode_data(30, 5.0), episode=1)
        det_b.train_models(1)

        bank_a = _build_shared_context_detector(
            {10: det_a, 20: det_b},
            memo_per_class=20,
            max_per_episode=None,
            seed=0,
            client_ids=[10],
            require_compatible_calibration=False,
        )
        bank_all = _build_shared_context_detector(
            {10: det_a, 20: det_b},
            memo_per_class=20,
            max_per_episode=None,
            seed=0,
            client_ids=None,
            require_compatible_calibration=False,
        )

        assert set(bank_a.episode_classes) == {0}
        assert set(bank_a.activation_memory) == {0}
        assert set(bank_all.episode_classes) == {0, 1}
        assert set(bank_all.activation_memory) == {0, 1}


class TestDeNICEEvaluation:
    def test_checkpoint_loader_normalizes_task_end_schema(self, monkeypatch):
        import eval_checkpoint

        monkeypatch.setattr(
            eval_checkpoint,
            "load_denice_checkpoint",
            lambda _path: {"task": 3, "algorithm": "denice"},
        )
        assert eval_checkpoint._load_checkpoint("unused.pt")["task_id"] == 3

    def test_evaluation_exposes_route_diagnostics_and_nomask_path(self):
        from fed_learning.servers.nice_server import ContextDetector
        from fed_learning.training.denice_eval import evaluate_denice_model

        model = _make_model()
        detector = ContextDetector(memo_per_class=10)
        data = {"X_test": _dummy_batch(8), "y_test": torch.zeros(8, dtype=torch.long)}
        metrics = evaluate_denice_model(
            model,
            data,
            device="cpu",
            context_detector=detector,
            seen_classes=list(range(NUM_CLASSES)),
            route_mode="nomask",
            include_route_diagnostics=True,
        )
        assert set(metrics["route_confusion"]) == set()
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert np.isfinite(metrics["loss"])

    def test_representative_ensemble_returns_normalized_metrics(self):
        from fed_learning.servers.nice_server import ContextDetector
        from fed_learning.training.denice_eval import evaluate_denice_ensemble

        model_a = _make_model()
        model_b = _make_model()
        detector_a = ContextDetector(memo_per_class=10)
        detector_b = ContextDetector(memo_per_class=10)
        data = {"X_test": _dummy_batch(8), "y_test": torch.zeros(8, dtype=torch.long)}
        metrics = evaluate_denice_ensemble(
            [(model_a, detector_a), (model_b, detector_b)],
            data,
            device="cpu",
            seen_classes=list(range(NUM_CLASSES)),
            route_mode="nomask",
        )
        assert metrics["ensemble_size"] == 2.0
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert np.isfinite(metrics["loss"])


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
