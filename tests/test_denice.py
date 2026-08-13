"""
Tests for DeNICE = NICE + CANC + capacity-aware micro-adapters.

Covers the plan's Phase 1 (model/adapter, novelty, CANC) and Phase 3
(capsule, dynamic-K clustering, age-aware aggregation).
"""

from collections import OrderedDict
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
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
    compact_algorithm_states,
    load_denice_checkpoint,
    save_delta_round_checkpoint,
    save_task_base_checkpoint,
)
from fed_learning.training.decentralized_denice_il import (
    _limit_eval_samples,
    _router_update_schedule,
    _should_run_post_task_eval,
    _should_refresh_router_after_aggregation,
    _should_update_local_router_context,
)


INPUT_SHAPE = (16, 1)
NUM_CLASSES = 12


def _make_model():
    torch.manual_seed(0)
    return DeNICEModel(INPUT_SHAPE, NUM_CLASSES)


def _dummy_batch(n=8):
    torch.manual_seed(1)
    return torch.randn(n, INPUT_SHAPE[0])


class TestDeNICEEvaluationScheduling:
    def test_debug_jsonl_appends_one_compact_record_per_call(self, tmp_path):
        from fed_learning.training.decentralized_denice_il import _append_jsonl

        output = tmp_path / "debug.jsonl"
        _append_jsonl(str(output), {"task": 0, "round": 0})
        _append_jsonl(str(output), {"task": 0, "round": 1})
        rows = [json.loads(line) for line in output.read_text().splitlines()]

        assert rows == [{"task": 0, "round": 0}, {"task": 0, "round": 1}]

    def test_post_task_eval_can_be_limited_to_final_task(self):
        config = {"denice_post_task_eval_tasks": [5]}
        assert not _should_run_post_task_eval(0, config)
        assert _should_run_post_task_eval(5, config)

    def test_post_task_eval_is_unrestricted_when_task_list_is_absent(self):
        assert _should_run_post_task_eval(0, {})

    def test_limited_eval_keeps_each_class_represented(self):
        X = torch.arange(60).reshape(30, 2)
        y = torch.tensor([0] * 10 + [1] * 10 + [2] * 10)
        _, sampled_y, sample_info = _limit_eval_samples(X, y, max_samples=9, seed=7)
        assert sample_info == {"limited": True, "total": 30, "used": 9}
        assert torch.bincount(sampled_y, minlength=3).tolist() == [3, 3, 3]

    def test_task_end_router_schedule_only_samples_once_and_refreshes_final(self):
        assert _router_update_schedule({"denice_router_update_schedule": "task_end"}) == "task_end"
        assert _should_update_local_router_context(0, "task_end")
        assert not _should_update_local_router_context(1, "task_end")
        assert not _should_refresh_router_after_aggregation(1, 3, 9999, "task_end")
        assert _should_refresh_router_after_aggregation(2, 3, 9999, "task_end")

    def test_task_end_schedule_refreshes_before_explicit_mid_round_eval(self):
        assert _should_refresh_router_after_aggregation(1, 5, 2, "task_end")

    def test_every_round_router_schedule_preserves_legacy_behavior(self):
        assert _router_update_schedule({}) == "every_round"
        assert _should_update_local_router_context(7, "every_round")
        assert _should_refresh_router_after_aggregation(7, 20, 9999, "every_round")

    def test_unknown_router_schedule_is_rejected(self):
        import pytest

        with pytest.raises(ValueError, match="denice_router_update_schedule"):
            _router_update_schedule({"denice_router_update_schedule": "sometimes"})


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
    def test_delta_metadata_preserves_raw_router_references_as_float(self):
        states = compact_algorithm_states(
            {7: {"denice": {"context_detector": {
                "activation_memory": {0: np.asarray([[0.0, 1.0]], dtype=np.float32)},
                "reference_input_memory": {0: np.asarray([[0.125, 0.875]], dtype=np.float32)},
            }}}}
        )
        detector_state = states[7]["denice"]["context_detector"]
        assert detector_state["activation_memory"][0].dtype == np.uint8
        assert detector_state["reference_input_memory"][0].dtype == np.float32
        assert np.allclose(detector_state["reference_input_memory"][0], [[0.125, 0.875]])

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

    def test_clone_roundtrip_is_deep_and_preserves_history(self):
        est = NoveltyEstimator()
        model = _make_model()
        x = _dummy_batch(16)
        est.calibrate_thresholds(model, x)
        est.store_prototype(0, est.compute_prototype(model, x))

        cloned = est.clone()

        assert cloned.has_history()
        assert cloned.thresholds == est.thresholds
        assert np.array_equal(cloned.prototypes[0]["conv1"], est.prototypes[0]["conv1"])
        cloned.prototypes[0]["conv1"][0] = 123.0
        assert est.prototypes[0]["conv1"][0] != 123.0

    def test_decentralized_continuation_roundtrip_preserves_non_tensor_state(self, tmp_path):
        from fed_learning.servers.nice_server import ContextDetector
        from fed_learning.training.checkpoint_state import restore_denice_state
        from fed_learning.training.decentralized_denice_il import (
            _build_denice_continuation_state,
            _load_denice_continuation_state,
        )

        model = _make_model()
        model.add_adapter(0, "fc1", set_active=False)
        model.unit_ranks["fc1"][:2] = [2, 1]
        model.freeze_masks = {"fc1": np.array([True, False] + [False] * 254)}
        detector = ContextDetector(memo_per_class=3, router_mode="multiclass")
        detector.episode_classes = {0: [0, 1]}
        novelty = NoveltyEstimator()
        x = _dummy_batch(8)
        novelty.calibrate_thresholds(model, x)
        novelty.store_prototype(0, novelty.compute_prototype(model, x))

        payload = _build_denice_continuation_state(
            task_id=0,
            config={"total_classes": NUM_CLASSES, "input_shape": INPUT_SHAPE},
            models={7: model}, context_detectors={7: detector},
            novelty_estimators={7: novelty}, prev_ages={7: model.get_neuron_ages_state()},
            old_ref_banks={7: {0: (x, torch.zeros(len(x), dtype=torch.long))}},
            old_ref_loss_baselines={7: 0.5}, last_active_task={7: 0},
            history={"task_accuracies": [{"task": 0}], "task_forgetting": [], "round_metrics": []},
            cluster_history=[{"task": 0}], adapter_history=[], debug_history=[],
        )
        path = tmp_path / "continuation_state_task_0.pt"
        torch.save(payload, path)
        restored_payload = _load_denice_continuation_state(str(path))

        restored_model = _make_model()
        restored_detector = ContextDetector(memo_per_class=1)
        denice_state = restored_payload["client_algorithm_states"][7]
        restore_denice_state(restored_model, restored_detector, denice_state)
        missing, unexpected = restored_model.load_state_dict(
            restored_payload["client_model_states"][7], strict=False
        )
        restored_novelty = NoveltyEstimator()
        restored_novelty.load_state(restored_payload["novelty_states"][7])

        assert not missing and not unexpected
        assert restored_model.get_adapter_registry_state() == model.get_adapter_registry_state()
        assert np.array_equal(restored_model.unit_ranks["fc1"], model.unit_ranks["fc1"])
        assert np.array_equal(restored_model.freeze_masks["fc1"], model.freeze_masks["fc1"])
        assert restored_detector.episode_classes == {0: [0, 1]}
        assert restored_novelty.has_history()
        assert restored_payload["meta"]["resume_from_task"] == 1

    def test_decentralized_split_run_matches_continuous_smoke(self, tmp_path):
        """A phase boundary must not silently reset DeNICE client state."""
        from fed_learning.training.decentralized_denice_il import (
            _load_denice_continuation_state,
            run_decentralized_denice_il,
        )

        data_dir = tmp_path / "tiny_split"
        data_dir.mkdir()
        metadata = {
            "task_structure": {"total_classes": 4, "task_classes": {"0": [0, 1], "1": [2, 3]}},
            "client_allocation": {"task_active_clients": {"0": [0, 1], "1": [0, 1]}},
        }
        (data_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        rng = np.random.default_rng(9)
        for cid in (0, 1):
            X = rng.normal(size=(8, 16, 1)).astype(np.float32)
            y = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
            np.savez(data_dir / f"client_{cid}_train.npz", X_train=X, y_train=y)
        np.savez(
            data_dir / "global_test_data.npz",
            X_test=rng.normal(size=(8, 16, 1)).astype(np.float32),
            y_test=np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64),
        )

        def config(output_dir, **overrides):
            return {
                "algorithm": "denice", "mode": "decentralized", "data_dir": str(data_dir),
                "output_dir": str(output_dir), "total_classes": 4,
                "rounds_per_task": 1, "batch_size": 2, "learning_rate": 0.001,
                "random_seed": 31, "denice_max_clients": 2,
                "denice_post_task_eval": False, "round_checkpoint_every": None,
                "save_resume_after_task": True, "nice_phase_epochs": 1,
                "denice_batch_sampling": "class_balanced",
                **overrides,
            }

        continuous = run_decentralized_denice_il(config(tmp_path / "continuous", task_end=1))
        split_first = run_decentralized_denice_il(config(tmp_path / "split", task_end=0))
        split_path = split_first["output_dir"]
        resumed = run_decentralized_denice_il(config(
            tmp_path / "ignored", task_end=1,
            resume_state_path=os.path.join(split_path, "continuation_state_task_0.pt"),
        ))

        continuous_state = _load_denice_continuation_state(
            os.path.join(continuous["output_dir"], "continuation_state_task_1.pt")
        )
        resumed_state = _load_denice_continuation_state(
            os.path.join(resumed["output_dir"], "continuation_state_task_1.pt")
        )
        assert [row["task"] for row in resumed_state["history"]["task_accuracies"]] == [0, 1]
        assert continuous_state["last_active_task"] == resumed_state["last_active_task"]
        for cid in continuous_state["client_ids"]:
            for name, value in continuous_state["client_model_states"][cid].items():
                assert torch.allclose(value, resumed_state["client_model_states"][cid][name], atol=1e-6)

    def test_late_client_without_history_is_not_global_first_task(self):
        from fed_learning.servers.nice_server import ContextDetector
        from fed_learning.training.decentralized_denice_il import _prepare_client_task

        model = _make_model()
        model.unit_ranks["fc1"][:] = 2
        client = SimpleNamespace(
            X_train=_dummy_batch(8),
            y_train=torch.tensor([2, 2, 2, 2, 3, 3, 3, 3]),
        )
        trainer = get_incremental_strategy("denice")
        prepared = _prepare_client_task(
            cid=9, task_id=1, num_tasks=2, new_classes=[2, 3], model=model,
            client=client, trainer=trainer, config={"denice_adapter_layers": ["fc1"]},
            device=torch.device("cpu"), context_detector=ContextDetector(memo_per_class=2),
            novelty_estimator=NoveltyEstimator(), prev_ages=None,
        )

        assert prepared["plan"]["is_global_first_task"] is False
        assert prepared["plan"]["has_novelty_baseline"] is False
        assert prepared["plan"]["novelty"] == 1.0
        assert "fc1" in prepared["plan"]["adapters_to_add"]

    def test_nice_loss_semantics_diagnostic_reports_full_denominator_effect(self):
        from fed_learning.training.decentralized_denice_il import _nice_loss_semantics_diagnostic

        model = _make_model()
        model.unit_ranks["fc2"][:] = 0
        model.unit_ranks["fc2"][:2] = 1
        diagnostic = _nice_loss_semantics_diagnostic(
            model, _dummy_batch(8), torch.tensor([0, 1] * 4), torch.device("cpu")
        )

        assert diagnostic["learner_output_count"] == 2
        assert diagnostic["nonlearner_output_count"] == NUM_CLASSES - 2
        assert diagnostic["all_targets_are_learner_outputs"] is True
        assert diagnostic["mean_nonlearner_probability_mass"] > 0.0
        assert diagnostic["full_output_ce"] >= diagnostic["learner_only_ce"]

    def test_run_validator_requires_complete_protocol_evidence(self, tmp_path):
        from tools.validate_denice_run import validate_denice_run

        run_dir = tmp_path / "run"
        eval_dir = run_dir / "d1_evaluation"
        eval_dir.mkdir(parents=True)
        (run_dir / "config.json").write_text(json.dumps({
            "algorithm": "denice", "mode": "decentralized", "task_start": 0,
            "task_end": 1, "rounds_per_task": 1, "random_seed": 42,
        }), encoding="utf-8")
        (run_dir / "training_history.json").write_text(json.dumps({
            "task_accuracies": [{"task": 0}, {"task": 1}],
            "round_metrics": [{"task": 0, "round": 0, "train_loss": 1.0},
                              {"task": 1, "round": 0, "train_loss": 1.0}],
        }), encoding="utf-8")
        (run_dir / "checkpoint_task_1.pt").write_bytes(b"checkpoint")
        required = [
            "e0_backbone_nomask", "e1_pred_adapter_nomask", "e2_oracle_adapter_nomask",
            "e3_oracle_routed_system_ceiling", "e3b_oracle_hard_no_adapter", "e4_pred_hard",
        ]
        record = {
            "accuracy": 0.5, "f1_macro": 0.4, "loss": 1.0,
            "evaluation_sampling": {"source_index_sha256": "abc"},
            "coverage_protocol": {
                "requested_sample_count": 10, "assigned_sample_count": 10,
                "unsupported_sample_count": 0, "partial_coverage": False,
            },
        }
        (eval_dir / "p6_evaluation_summary.json").write_text(json.dumps({
            "policies": required,
            "summary": {"coverage_aware_local": {name: record for name in required}},
        }), encoding="utf-8")

        valid = validate_denice_run(run_dir, require_evaluation=True)
        assert valid["valid"] is True

        (run_dir / "checkpoint_task_1.pt").unlink()
        (run_dir / "checkpoint_task_1_round_0.pt").write_bytes(b"round checkpoint")
        valid_round_only = validate_denice_run(
            run_dir, expected_rounds_per_task=1, require_evaluation=True
        )
        assert valid_round_only["valid"] is True
        assert valid_round_only["evidence"]["final_checkpoint"].endswith(
            "checkpoint_task_1_round_0.pt"
        )

        record["coverage_protocol"]["partial_coverage"] = True
        (eval_dir / "p6_evaluation_summary.json").write_text(json.dumps({
            "policies": required,
            "summary": {"coverage_aware_local": {name: record for name in required}},
        }), encoding="utf-8")
        invalid = validate_denice_run(run_dir, require_evaluation=True)
        assert invalid["valid"] is False
        assert any("partial coverage" in error for error in invalid["errors"])

    def test_d1_analyzer_requires_confirmation_before_opening_d2(self, tmp_path):
        from tools.analyze_denice_d1 import analyze_d1

        def make_manifest(root, seed):
            variants = {}
            for name, f1 in (("peer_default", 0.08), ("self_only", 0.10), ("peer_self_floor_050", 0.09)):
                run_dir = root / name
                eval_dir = run_dir / "d1_evaluation"
                eval_dir.mkdir(parents=True)
                (run_dir / "cluster_history.json").write_text(json.dumps([{
                    "plastic_fc2_row_audit": {"0": [{
                        "row_drift_l2": 2.0,
                        "peer_alpha_supported": 0.0,
                        "peer_alpha_unsupported": 0.5,
                    }]}
                }]), encoding="utf-8")
                coverage = {
                    "requested_sample_count": 10, "assigned_sample_count": 10,
                    "unsupported_sample_count": 0, "partial_coverage": False,
                }
                summary = {
                    "summary": {"coverage_aware_local": {
                        "e3_oracle_routed_system_ceiling": {
                            "accuracy": 0.2, "f1_macro": f1,
                            "evaluation_sampling": {"source_index_sha256": "same"},
                            "coverage_protocol": coverage,
                        },
                        "e4_pred_hard": {"accuracy": 0.1, "f1_macro": 0.05},
                    }}
                }
                summary_path = eval_dir / "p6_evaluation_summary.json"
                summary_path.write_text(json.dumps(summary), encoding="utf-8")
                checkpoint = run_dir / "checkpoint_task_2_round_4.pt"
                checkpoint.write_bytes(b"x")
                variants[name] = {
                    "checkpoint": str(checkpoint), "evaluation_summary": str(summary_path),
                }
            manifest = root / "d1_manifest.json"
            manifest.write_text(json.dumps({"seed": seed, "variants": variants}), encoding="utf-8")
            return manifest

        first = make_manifest(tmp_path / "seed42", 42)
        no_confirmation = analyze_d1(first)
        assert no_confirmation["decision"] == "KEEP_D2_CLOSED"
        assert no_confirmation["conditions"]["material_negative_transfer"] is True

        second = make_manifest(tmp_path / "seed43", 43)
        confirmed = analyze_d1(first, confirmation_manifest_path=second)
        assert confirmed["decision"] == "OPEN_D2"
        assert confirmed["d2_eligible"] is True


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

    def test_canc_pressure_components_sum_to_kappa_and_emit_thresholds(self):
        ctrl = CapacityController(CANCConfig(alpha=0.4, beta=0.2, gamma=0.1, delta=0.3))
        decision = ctrl.decide_layer("fc1", rho0=0.5, rhom=0.2, u=0.25, novelty=0.4, val_loss_delta=0.1)
        assert decision["kappa"] == pytest.approx(
            decision["pressure_capacity"] + decision["pressure_consumption"]
            + decision["pressure_validation"] + decision["pressure_novelty"]
        )
        plan = ctrl.plan_task({"fc1": {"rho0": 0.5, "rhom": 0.2}}, novelty=0.4)
        assert {"kappa_mid", "kappa_high", "kappa_adapter"}.issubset(plan["thresholds"])

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

    def test_observed_silhouette_is_valid_at_calibrated_threshold(self, monkeypatch):
        import fed_learning.strategies.decentralized.denice_clustering as clustering

        caps = [self._capsule(i, [0, 1], proto_seed=i) for i in range(4)]
        similarity = np.ones((4, 4), dtype=np.float64)
        edges = np.ones((4, 4), dtype=np.int8)
        np.fill_diagonal(edges, 0)
        monkeypatch.setattr(
            clustering,
            "build_similarity_matrix",
            lambda *_args, **_kwargs: (similarity, edges),
        )
        monkeypatch.setattr(
            clustering,
            "affinity_propagation",
            lambda *_args, **_kwargs: (np.array([0, 0, 1, 1]), [0, 2]),
        )
        monkeypatch.setattr(
            clustering,
            "silhouette_score_from_similarity",
            lambda *_args, **_kwargs: 0.2365,
        )

        result = clustering.dynamic_ap_cluster(
            caps, config=clustering.ClusteringConfig(theta_s=0.20)
        )

        assert result["valid"] is True
        assert result["K_t"] == 2

    def test_nan_silhouette_remains_invalid_after_threshold_calibration(self, monkeypatch):
        import fed_learning.strategies.decentralized.denice_clustering as clustering

        caps = [self._capsule(i, [0, 1], proto_seed=i) for i in range(4)]
        similarity = np.ones((4, 4), dtype=np.float64)
        edges = np.ones((4, 4), dtype=np.int8)
        np.fill_diagonal(edges, 0)
        monkeypatch.setattr(
            clustering,
            "build_similarity_matrix",
            lambda *_args, **_kwargs: (similarity, edges),
        )
        monkeypatch.setattr(
            clustering,
            "affinity_propagation",
            lambda *_args, **_kwargs: (np.array([0, 0, 1, 1]), [0, 2]),
        )
        monkeypatch.setattr(
            clustering,
            "silhouette_score_from_similarity",
            lambda *_args, **_kwargs: float("nan"),
        )

        result = clustering.dynamic_ap_cluster(
            caps, config=clustering.ClusteringConfig(theta_s=0.20)
        )

        assert result["valid"] is False

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

    def test_valid_cluster_reports_positive_peer_aggregation(self, monkeypatch):
        """Peer alpha excludes self and proves collaboration is not a no-op."""
        import fed_learning.training.decentralized_denice_il as runner

        models = {cid: _make_model() for cid in range(4)}
        with torch.no_grad():
            for cid, model in models.items():
                model.fc1.weight.fill_(float(cid))
        receiver_before = models[0].fc1.weight.detach().clone()
        capsules = {
            cid: self._capsule(cid, [0, 1], cid + 1) for cid in models
        }
        edges = np.ones((4, 4), dtype=np.int8)
        np.fill_diagonal(edges, 0)
        monkeypatch.setattr(
            runner,
            "dynamic_ap_cluster",
            lambda *_args, **_kwargs: {
                "labels": np.array([0, 0, 1, 1]),
                "edges": edges,
                "valid": True,
                "K_t": 2,
                "silhouette": 0.3,
                "similarity": np.ones((4, 4)),
            },
        )

        summary = runner._aggregate_round(
            client_ids=list(models),
            models=models,
            capsules=capsules,
            config={"denice_age_merge_policy": "consensus"},
            device=torch.device("cpu"),
        )

        assert summary["effective_K_t"] == 2
        assert summary["group_size_stats"]["mean"] == 2.0
        assert summary["peer_aggregated_client_count"] == 4
        assert summary["peer_alpha_sum_stats"]["mean"] > 0.0
        assert all(
            info["peer_alpha_sum"] > 0.0
            for info in summary["alpha_debug"].values()
        )
        assert not torch.allclose(models[0].fc1.weight, receiver_before)
        assert torch.count_nonzero(models[0].fc1.weight).item() > 0

    def test_aggregation_uses_clustering_effective_similarity(self, monkeypatch):
        """Adaptive score provenance must survive from AP to peer alpha."""
        import fed_learning.training.decentralized_denice_il as runner

        models = {cid: _make_model() for cid in range(2)}
        capsules = {cid: self._capsule(cid, [0, 1], cid + 1) for cid in models}
        monkeypatch.setattr(
            runner,
            "dynamic_ap_cluster",
            lambda *_args, **_kwargs: {
                "labels": np.array([0, 0]),
                "edges": np.array([[0, 1], [1, 0]], dtype=np.int8),
                "valid": True,
                "K_t": 1,
                "silhouette": 0.3,
                # Sparse AP matrix is intentionally not the score source.
                "similarity": np.array([[-1e9, -1e9], [-1e9, -1e9]]),
                "effective_similarity": np.array([[0.0, 0.8], [0.6, 0.0]]),
                "effective_weights": {"label": 1.0},
            },
        )

        summary = runner._aggregate_round(
            client_ids=[0, 1], models=models, capsules=capsules,
            config={}, device=torch.device("cpu"),
        )

        assert summary["alpha_debug"][0]["similarities"] == [1.0, 0.8]
        assert summary["alpha_debug"][1]["similarities"] == [0.6, 1.0]
        assert all(
            row["similarity_source"] == "clustering_effective_similarity"
            for row in summary["alpha_debug"].values()
        )

    def test_d1_row_drift_audit_records_supported_peer_weight(self, monkeypatch):
        import fed_learning.training.decentralized_denice_il as runner

        models = {cid: _make_model() for cid in range(2)}
        for model in models.values():
            model.unit_ranks["fc2"][1] = 1
        capsules = {cid: self._capsule(cid, [0, 1], cid + 1) for cid in models}
        edges = np.array([[0, 1], [1, 0]], dtype=np.int8)
        monkeypatch.setattr(
            runner, "dynamic_ap_cluster", lambda *_args, **_kwargs: {
                "labels": np.array([0, 0]), "edges": edges, "valid": True,
                "K_t": 1, "silhouette": 0.3, "similarity": np.ones((2, 2)),
                "effective_similarity": np.array([[0.0, 0.8], [0.8, 0.0]]),
            }
        )
        summary = runner._aggregate_round(
            client_ids=[0, 1], models=models, capsules=capsules,
            config={"denice_d1_row_drift_audit": True}, device=torch.device("cpu"),
        )

        rows = summary["plastic_fc2_row_audit"][0]
        row = next(item for item in rows if item["class_id"] == 1)
        assert row["peer_alpha_supported"] > 0.0
        assert row["peer_alpha_unsupported"] == 0.0

    def test_self_only_ablation_blocks_peer_state_but_keeps_cluster_evidence(self, monkeypatch):
        """D1's control must not leak peer params, adapters, or ages."""
        import fed_learning.training.decentralized_denice_il as runner

        models = {cid: _make_model() for cid in range(2)}
        with torch.no_grad():
            models[0].fc1.weight.zero_()
            models[1].fc1.weight.fill_(3.0)
        before = {
            cid: OrderedDict((name, value.detach().clone()) for name, value in model.state_dict().items())
            for cid, model in models.items()
        }
        capsules = {cid: self._capsule(cid, [0, 1], cid + 1) for cid in models}
        monkeypatch.setattr(
            runner,
            "dynamic_ap_cluster",
            lambda *_args, **_kwargs: {
                "labels": np.array([0, 0]),
                "edges": np.ones((2, 2), dtype=np.int8),
                "valid": True,
                "K_t": 1,
                "silhouette": 0.3,
                "similarity": np.ones((2, 2)),
            },
        )

        summary = runner._aggregate_round(
            client_ids=[0, 1],
            models=models,
            capsules=capsules,
            config={"denice_aggregation_mode": "self_only"},
            device=torch.device("cpu"),
        )

        assert summary["raw_K_t"] == 1
        assert summary["aggregation_mode"] == "self_only"
        assert summary["groups"] == {0: [0], 1: [1]}
        assert summary["peer_aggregated_client_count"] == 0
        assert summary["peer_alpha_sum_stats"]["max"] == 0.0
        assert all(
            torch.allclose(value, before[cid][name])
            for cid, model in models.items()
            for name, value in model.state_dict().items()
        )

    def test_collaboration_guard_triggers_on_second_collapsed_round(self):
        from fed_learning.training.decentralized_denice_il import (
            _update_collaboration_guard,
        )

        collapsed = {
            "effective_K_t": 4,
            "group_size_stats": {"min": 1.0, "mean": 1.0, "max": 1.0},
            "peer_alpha_sum_stats": {"min": 0.0, "mean": 0.0, "max": 0.0},
            "peer_aggregated_client_count": 0,
        }
        config = {
            "denice_collaboration_guard_mode": "error",
            "denice_max_consecutive_self_only_rounds": 2,
            "denice_min_mean_peer_alpha": 0.05,
        }
        first, streak = _update_collaboration_guard(collapsed, 4, 0, config)
        second, streak = _update_collaboration_guard(collapsed, 4, streak, config)

        assert first["collapsed"] is True
        assert first["triggered"] is False
        assert second["triggered"] is True
        assert streak == 2
        assert "has_positive_peer_weight" in second["failed_checks"]

    def test_valid_peer_aggregation_resets_collaboration_guard_streak(self):
        from fed_learning.training.decentralized_denice_il import (
            _update_collaboration_guard,
        )

        valid = {
            "effective_K_t": 2,
            "group_size_stats": {"min": 2.0, "mean": 2.0, "max": 2.0},
            "peer_alpha_sum_stats": {"min": 0.2, "mean": 0.4, "max": 0.6},
            "peer_aggregated_client_count": 4,
        }
        guard, streak = _update_collaboration_guard(
            valid,
            4,
            1,
            {
                "denice_collaboration_guard_mode": "error",
                "denice_min_mean_peer_alpha": 0.05,
            },
        )

        assert guard["collapsed"] is False
        assert guard["triggered"] is False
        assert guard["failed_checks"] == []
        assert streak == 0

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
    def test_smoke_client_limit_preserves_each_present_class(self):
        from fed_learning.training.decentralized_denice_il import (
            _stratified_limit_client_task_data,
        )

        X = torch.arange(30, dtype=torch.float32).reshape(30, 1)
        y = torch.tensor([0] * 25 + [1] * 3 + [2] * 2, dtype=torch.long)
        X_limited, y_limited = _stratified_limit_client_task_data(X, y, 9, seed=5)
        X_again, y_again = _stratified_limit_client_task_data(X, y, 9, seed=5)

        assert len(y_limited) == 9
        assert set(y_limited.tolist()) == {0, 1, 2}
        assert torch.equal(X_limited, X_again)
        assert torch.equal(y_limited, y_again)

    def test_context_reference_memory_refreshes_after_model_change(self):
        from fed_learning.servers.nice_server import ContextDetector
        from fed_learning.training.checkpoint_state import (
            restore_context_detector,
            snapshot_context_detector,
        )

        model = _make_model()
        detector = ContextDetector(memo_per_class=4, router_mode="multiclass")
        detector.episode_classes = {0: [0], 1: [1]}
        ref0 = _dummy_batch(8)
        ref1 = _dummy_batch(8) + 1.5
        detector.push_activations(model, ref0, 0, reference_data=ref0)
        detector.push_activations(model, ref1, 1, reference_data=ref1)
        detector.train_models(1)
        old_memory = detector.activation_memory[0].copy()
        with torch.no_grad():
            model.conv1.weight.add_(0.5)
        detector.mark_router_stale("test_model_change")
        summary = detector.refresh_activation_memory(
            model, task_id=1, round_id=4, batch_size=3
        )

        assert summary["refreshed_episode_count"] == 2
        assert summary["reference_sample_count"] == 16
        assert summary["encode_time"] >= 0.0
        assert summary["fit_time"] >= 0.0
        assert summary["total_time"] == summary["encode_time"] + summary["fit_time"]
        assert detector.router_state_fresh is True
        assert detector.router_last_refresh_task == 1
        assert detector.router_last_refresh_round == 4
        assert detector.router_stale_reason is None
        assert detector.activation_memory[0].shape == old_memory.shape
        state = snapshot_context_detector(detector)
        restored = ContextDetector(memo_per_class=1)
        restore_context_detector(restored, state)
        assert set(restored.reference_input_memory) == {0, 1}
        assert restored.router_state_fresh is True
        assert restored.router_last_refresh_task == 1
        assert restored.router_last_refresh_round == 4

    def test_vectorized_binary_activation_matches_legacy_list_conversion(self):
        from fed_learning.servers.nice_server import ContextDetector

        model = _make_model().eval()
        detector = ContextDetector(memo_per_class=4)
        data = _dummy_batch(7)
        acts = model.get_context_activations_per_sample(data)
        legacy = detector.binarize_layer_activations(
            {
                name: np.asarray(value.detach().cpu().tolist())
                for name, value in acts.items()
            }
        )
        vectorized = detector._binarize_per_sample(model, data)

        assert np.array_equal(vectorized, legacy)

    def test_router_reference_quota_is_separate_from_novelty_memory(self):
        from fed_learning.servers.nice_server import ContextDetector
        from fed_learning.training.local_task_loop import (
            _update_local_nice_context_memory,
        )

        model = _make_model()
        detector = ContextDetector(
            memo_per_class=50,
            router_reference_per_class=2,
            router_mode="multiclass",
        )
        data = _dummy_batch(12)
        labels = torch.tensor([0] * 6 + [1] * 6)
        profile = _update_local_nice_context_memory(
            detector,
            model,
            data,
            labels,
            task_id=0,
            task_classes=[0, 1],
            device="cpu",
            fit_router=False,
        )

        assert detector.memo_per_class == 50
        assert detector.router_reference_per_class == 2
        assert profile["reference_sample_count"] == 4
        assert detector.reference_input_memory[0].shape[0] == 4
        assert detector.router_state_fresh is False

    def test_old_checkpoint_without_freshness_metadata_restores_as_stale(self):
        from fed_learning.servers.nice_server import ContextDetector
        from fed_learning.training.checkpoint_state import restore_context_detector

        detector = ContextDetector(memo_per_class=1)
        restore_context_detector(detector, {"episode_classes": {0: [0]}})

        assert detector.router_state_fresh is False
        assert detector.router_stale_reason == "checkpoint_missing_freshness_metadata"

    def test_balanced_router_audit_subset_has_equal_episode_quota(self):
        import eval_checkpoint

        labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.long)
        indices, episodes = eval_checkpoint._balanced_episode_indices(
            labels,
            {0: [0, 1], 1: [2]},
            samples_per_episode=2,
            seed=9,
        )

        assert len(indices) == 4
        assert (episodes == 0).sum() == 2
        assert (episodes == 1).sum() == 2
        episode_tensor = torch.tensor(episodes.tolist(), dtype=torch.long)
        assert set(labels[indices[episode_tensor == 0]].tolist()).issubset({0, 1})
        assert set(labels[indices[episode_tensor == 1]].tolist()) == {2}

    def test_current_router_audit_ignores_future_dataset_episodes(self, monkeypatch):
        import eval_checkpoint
        from fed_learning.servers.nice_server import ContextDetector

        detector = ContextDetector(memo_per_class=2, router_mode="multiclass")
        detector.activation_memory = {
            0: np.zeros((2, 2), dtype=np.float32),
            1: np.ones((2, 2), dtype=np.float32),
        }
        detector.episode_classes = {0: [0], 1: [1]}
        detector.train_models(1)

        checkpoint = {
            "client_ids": [7],
            "client_model_states": {7: {}},
            "client_algorithm_states": {7: {"denice": {"context_detector": {
                "episode_classes": {0: [0], 1: [1]},
                "activation_memory": {0: np.zeros((2, 2)), 1: np.ones((2, 2))},
            }}}},
        }
        monkeypatch.setattr(
            eval_checkpoint,
            "_make_denice_client_model",
            lambda *_args, **_kwargs: (object(), detector),
        )
        monkeypatch.setattr(
            eval_checkpoint,
            "_binary_current_context_features",
            lambda *_args, **_kwargs: np.zeros((4, 2), dtype=np.float32),
        )
        audit = eval_checkpoint.audit_denice_router_current_features(
            checkpoint,
            torch.zeros((4, 1)),
            torch.tensor([0, 0, 1, 1]),
            {0: [0], 1: [1], 2: [2]},
            device="cpu",
            samples_per_episode=2,
        )

        assert audit["required_episodes"] == [0, 1]
        assert audit["full_coverage_client_count"] == 1

    def test_router_state_audit_reports_coverage_and_holdout(self):
        import eval_checkpoint
        from fed_learning.servers.nice_server import ContextDetector
        from fed_learning.training.checkpoint_state import snapshot_context_detector

        detector = ContextDetector(memo_per_class=4, router_mode="multiclass")
        detector.activation_memory = {
            0: np.zeros((8, 3), dtype=np.float32),
            1: np.ones((8, 3), dtype=np.float32),
        }
        detector.context_masks = {0: np.ones(3, dtype=bool), 1: np.ones(3, dtype=bool)}
        detector.episode_classes = {0: [0], 1: [1]}
        detector.train_models(1)
        checkpoint = {
            "config": {"denice_router_mode": "multiclass", "memo_per_class": 4},
            "client_algorithm_states": {
                7: {"denice": {"context_detector": snapshot_context_detector(detector)}}
            },
        }
        audit = eval_checkpoint.audit_denice_router_states(checkpoint, seed=1)

        assert audit["client_count"] == 1
        assert audit["eligible_holdout_client_count"] == 1
        assert audit["memory_episode_coverage_histogram"] == {"0,1": 1}
        assert audit["clients"][0]["refit_holdout"]["balanced_accuracy"] == 1.0

    def test_coverage_aware_partition_never_assigns_unsupported_episode(self):
        import eval_checkpoint

        test_y = torch.tensor([0, 1, 2, 3, 0, 2, 3], dtype=torch.long)
        task_classes = {0: [0, 1], 1: [2, 3]}
        coverage = {
            10: {"supported_episodes": [0]},
            20: {"supported_episodes": [1]},
        }
        partitions, debug = eval_checkpoint._build_coverage_aware_partitions(
            test_y,
            [10, 20],
            task_classes,
            coverage,
            seed=7,
        )

        assert debug["unsupported_sample_count"] == 0
        assert set(test_y[partitions[10]].tolist()).issubset({0, 1})
        assert set(test_y[partitions[20]].tolist()).issubset({2, 3})
        assert sum(len(indices) for indices in partitions.values()) == len(test_y)

    def test_coverage_aware_partition_fails_closed_when_episode_is_unsupported(self):
        import eval_checkpoint

        _partitions, debug = eval_checkpoint._build_coverage_aware_partitions(
            torch.tensor([0, 2], dtype=torch.long),
            [10],
            {0: [0], 1: [2]},
            {10: {"supported_episodes": [0]}},
            seed=7,
        )

        with pytest.raises(ValueError, match="incomplete support"):
            eval_checkpoint._validate_coverage_partition(
                debug, allow_partial_coverage=False
            )
        eval_checkpoint._validate_coverage_partition(debug, allow_partial_coverage=True)
        assert debug["partial_coverage"] is True
        assert debug["assigned_sample_count"] == 1
        assert debug["requested_sample_count"] == 2

    def test_class_balanced_test_subset_has_fixed_reproducible_support(self):
        import eval_checkpoint

        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
        first, first_debug = eval_checkpoint._select_class_balanced_test_indices(
            labels, samples_per_class=3, seed=19
        )
        second, second_debug = eval_checkpoint._select_class_balanced_test_indices(
            labels, samples_per_class=3, seed=19
        )

        assert torch.equal(first, second)
        assert first_debug == second_debug
        assert torch.bincount(labels[first], minlength=3).tolist() == [3, 3, 3]
        assert first_debug["sampling_protocol"] == "class_balanced_global_test_without_replacement"
        assert first_debug["unique_source_sample_count"] == 9

    def test_class_balanced_test_subset_requires_quota_or_explicit_replacement(self):
        import eval_checkpoint

        labels = torch.tensor([0, 0, 1], dtype=torch.long)
        with pytest.raises(ValueError, match="class-balanced evaluation needs 2 examples for class 1"):
            eval_checkpoint._select_class_balanced_test_indices(
                labels, samples_per_class=2, seed=3
            )
        indices, debug = eval_checkpoint._select_class_balanced_test_indices(
            labels, samples_per_class=2, seed=3, with_replacement=True
        )
        assert torch.bincount(labels[indices], minlength=2).tolist() == [2, 2]
        assert debug["unique_source_sample_count"] < len(indices)
        assert debug["with_replacement"] is True

    def test_partitioned_local_eval_uses_each_global_sample_once(self, monkeypatch):
        import eval_checkpoint

        class DummyModel:
            pass

        monkeypatch.setattr(
            eval_checkpoint,
            "_make_denice_client_model",
            lambda *_args, **_kwargs: (DummyModel(), object()),
        )

        def routed_logits(_model, X_batch, _detector, _seen, _device, _mode, _topk, **_kwargs):
            labels = X_batch[:, 0].long()
            logits = torch.full((len(labels), 3), -10.0)
            logits[torch.arange(len(labels)), labels] = 10.0
            return logits, None

        monkeypatch.setattr(eval_checkpoint, "_denice_routed_logits_with_episodes", routed_logits)
        X = torch.arange(9, dtype=torch.float32).remainder(3).unsqueeze(1)
        y = X[:, 0].long()
        metrics = eval_checkpoint._evaluate_denice_partitioned_clients(
            {"config": {"eval_batch_size": 2}, "seen_classes": [0, 1, 2]},
            [3, 7, 9],
            X,
            y,
            device="cpu",
            router_mode="multiclass",
            route_mode="hard",
            route_topk=1,
            eval_seed=42,
            task_id=5,
            include_prediction_trace=True,
        )

        assert sum(metrics["partition_sizes"].values()) == len(y)
        assert metrics["partition_count"] == 3
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert set(metrics["debug"]["per_class"]) == {"0", "1", "2"}
        assert len(metrics["debug"]["worst_partitions_by_accuracy"]) == 3
        trace = metrics["debug"]["prediction_trace"]
        assert len(trace["source_test_indices"]) == len(y)
        assert trace["targets"] == trace["predictions"]
        assert len(trace["trace_sha256"]) == 64

    def test_oracle_hard_masks_to_true_episode_classes(self, monkeypatch):
        from fed_learning.servers.nice_server import ContextDetector
        import fed_learning.training.denice_eval as denice_eval

        model = _make_model()
        detector = ContextDetector(memo_per_class=10, router_mode="multiclass")
        detector.episode_classes = {0: [0, 1], 1: [2, 3]}
        X = _dummy_batch(2)
        monkeypatch.setattr(
            denice_eval,
            "_route_episodes_with_scores",
            lambda *_args: (np.asarray([1, 0]), None),
        )
        logits, _ = denice_eval._denice_routed_logits_with_episodes(
            model,
            X,
            detector,
            seen_classes=[0, 1, 2, 3],
            device="cpu",
            inference_policy="oracle_hard",
            oracle_episodes=np.asarray([0, 1]),
        )

        assert torch.all(logits[0, 2:4] == -100.0)
        assert torch.all(logits[1, :2] == -100.0)

    def test_oracle_hard_no_adapter_keeps_oracle_mask_without_adapter(self, monkeypatch):
        from fed_learning.servers.nice_server import ContextDetector
        import fed_learning.training.denice_eval as denice_eval

        model = _make_model()
        model.add_adapter(0, "fc1", set_active=False)
        detector = ContextDetector(memo_per_class=10, router_mode="multiclass")
        detector.episode_classes = {0: [0, 1], 1: [2, 3]}
        calls = []
        monkeypatch.setattr(
            model, "set_active_context", lambda episode: calls.append(int(episode))
        )
        monkeypatch.setattr(
            denice_eval,
            "_route_episodes_with_scores",
            lambda *_args: (np.asarray([1, 0]), None),
        )
        logits, _ = denice_eval._denice_routed_logits_with_episodes(
            model, _dummy_batch(2), detector, seen_classes=[0, 1, 2, 3],
            device="cpu", inference_policy="oracle_hard_no_adapter",
            oracle_episodes=np.asarray([0, 1]),
        )

        assert calls == []
        assert torch.all(logits[0, 2:4] == -100.0)
        assert torch.all(logits[1, :2] == -100.0)

    def test_adaptive_route_records_fallback_and_preserves_seen_classes(self, monkeypatch):
        from fed_learning.servers.nice_server import ContextDetector
        import fed_learning.training.denice_eval as denice_eval

        model = _make_model()
        detector = ContextDetector(memo_per_class=10, router_mode="multiclass")
        detector.episode_classes = {0: [0, 1], 1: [2, 3]}
        X = _dummy_batch(3)
        monkeypatch.setattr(
            denice_eval,
            "_route_episodes_with_scores",
            lambda *_args: (
                np.asarray([0, 1, 1]),
                np.asarray([
                    [0.90, 0.10, 0.00],
                    [0.55, 0.45, 0.00],
                    [0.30, 0.40, 0.30],
                ]),
            ),
        )
        diagnostics = {}
        logits, _ = denice_eval._denice_routed_logits_with_episodes(
            model,
            X,
            detector,
            seen_classes=[0, 1, 2, 3],
            device="cpu",
            route_mode="adaptive",
            route_topk=2,
            routing_diagnostics=diagnostics,
            adaptive_high_confidence=0.75,
            adaptive_low_confidence=0.45,
        )

        assert diagnostics["adaptive_hard_sample_count"] == 1
        assert diagnostics["adaptive_topk_sample_count"] == 1
        assert diagnostics["adaptive_nomask_sample_count"] == 1
        assert torch.all(logits[0, 2:4] == -100.0)
        assert torch.all(logits[1, :4] > -100.0)
        assert torch.all(logits[2, :4] > -100.0)

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

    def test_representative_ensemble_oracle_hard_preserves_true_episode_classes(self, monkeypatch):
        from fed_learning.servers.nice_server import ContextDetector
        import fed_learning.training.denice_eval as denice_eval

        model = _make_model()
        detector = ContextDetector(memo_per_class=10, router_mode="multiclass")
        detector.episode_classes = {0: [0, 1], 1: [2, 3]}
        monkeypatch.setattr(
            denice_eval,
            "_route_episodes_with_scores",
            lambda *_args: (np.asarray([1, 0]), None),
        )
        metrics = denice_eval.evaluate_denice_ensemble(
            [(model, detector)],
            {"X_test": _dummy_batch(2), "y_test": torch.tensor([0, 2])},
            device="cpu",
            seen_classes=[0, 1, 2, 3],
            inference_policy="oracle_hard",
            class_to_episode={0: 0, 1: 0, 2: 1, 3: 1},
        )

        assert metrics["inference_policy"] == "oracle_hard"
        assert metrics["oracle_mask_violation_count"] == 0

    def test_p6_runner_has_complete_policy_matrix(self):
        from run_denice_p6_eval import POLICIES

        assert set(POLICIES) == {
            "e0_backbone_nomask",
            "e1_pred_adapter_nomask",
            "e2_oracle_adapter_nomask",
            "e3_oracle_routed_system_ceiling",
            "e3b_oracle_hard_no_adapter",
            "e4_pred_hard",
            "e5_topk2",
            "e5_topk3",
            "e6_adaptive",
        }

    def test_representative_global_requires_full_router_coverage(self, monkeypatch):
        import eval_checkpoint

        checkpoint = {
            "config": {"data_dir": "unused", "eval_batch_size": 8},
            "algorithm": "denice",
            "task_id": 1,
            "round_id": 0,
            "client_ids": [1, 2],
            "client_model_states": {1: {}, 2: {}},
            "client_algorithm_states": {
                1: {"denice": {"context_detector": {
                    "episode_classes": {0: [0]}, "activation_memory": {0: [[0.0]]},
                }}},
                2: {"denice": {"context_detector": {
                    "episode_classes": {0: [0], 1: [1]},
                    "activation_memory": {0: [[0.0]], 1: [[1.0]]},
                }}},
            },
            "seen_classes": [0, 1],
        }

        class DummyLoader:
            task_classes = {0: [0], 1: [1]}
            def __init__(self, *_args, **_kwargs): pass
            def get_test_data(self, *_args, **_kwargs):
                return torch.zeros((2, 1)), torch.tensor([0, 1])

        monkeypatch.setattr(eval_checkpoint, "_load_checkpoint", lambda *_args: checkpoint)
        monkeypatch.setattr(eval_checkpoint, "IncrementalDataLoader", DummyLoader)
        monkeypatch.setattr(
            eval_checkpoint, "_make_denice_client_model", lambda *_args, **_kwargs: (object(), object())
        )
        monkeypatch.setattr(
            eval_checkpoint, "evaluate_denice_ensemble", lambda pairs, *_args, **_kwargs: {"ensemble_size": len(pairs)}
        )
        monkeypatch.setattr(eval_checkpoint.hashlib, "sha256", lambda *_args: type("H", (), {"hexdigest": lambda self: "h"})())
        monkeypatch.setattr(eval_checkpoint.Path, "read_bytes", lambda *_args: b"x")

        result = eval_checkpoint.evaluate_checkpoint(
            "unused.pt", data_dir="unused", evaluation_mode="representative_global"
        )
        assert result["representative_client_ids"] == [2]
        assert result["representative_coverage_debug"]["required_episodes"] == [0, 1]

    def test_p6_summary_validator_accepts_three_verified_seeds(self, tmp_path, monkeypatch):
        import summarize_denice_p6

        policies = tuple(summarize_denice_p6.POLICIES)
        protocols = tuple(summarize_denice_p6.PROTOCOLS)
        run_dirs = []
        for seed in (42, 43, 44):
            run_dir = tmp_path / f"seed_{seed}"
            run_dir.mkdir()
            metric = {
                "accuracy": 0.50,
                "f1_macro": 0.40,
                "f1_weighted": 0.45,
                "loss": 1.0,
                "route_accuracy": 0.80,
                "oracle_mask_violation_count": 0,
                "checkpoint_sha256": f"checkpoint-{seed}",
                "config_sha256": f"config-{seed}",
            }
            summary = {
                "training_seed": seed,
                "summary": {
                    protocol: {policy: dict(metric) for policy in policies}
                    for protocol in protocols
                },
            }
            summary["summary"]["coverage_aware_oracle_gap"] = 0.02
            (run_dir / "p6_evaluation_summary.json").write_text(json.dumps(summary))
            # The validator reads this detailed E4 artifact for collapse checks.
            (run_dir / "coverage_aware_local_e4_pred_hard.json").write_text(
                json.dumps({"metrics": {"debug": {"route_confusion": {"0": {"0": 3, "1": 2}, "1": {"0": 2, "1": 3}}}}})
            )
            run_dirs.append(str(run_dir))
        output = tmp_path / "final.json"
        monkeypatch.setattr(
            "sys.argv",
            ["summarize_denice_p6.py", "--run-dirs", *run_dirs, "--output", str(output)],
        )
        summarize_denice_p6.main()
        report = json.loads(output.read_text())

        assert report["gates"]["three_distinct_seeds"]
        assert report["gates"]["oracle_mask_violations_zero"]
        assert report["gates"]["predicted_hard_oracle_gap_le_5_points"]


# ---------------------------------------------------------------------------
# D3 imbalance controls
# ---------------------------------------------------------------------------
class TestDeNICEImbalanceControls:
    def test_class_balanced_batches_keep_epoch_budget_and_balance_batches(self):
        from fed_learning.clients.denice_client import class_balanced_batch_indices

        torch.manual_seed(7)
        labels = torch.tensor([0] * 8 + [1] * 2)
        batches = list(class_balanced_batch_indices(labels, batch_size=4))

        assert [len(batch) for batch in batches] == [4, 4, 2]
        for batch in batches:
            counts = torch.bincount(labels[batch], minlength=2)
            assert int((counts.max() - counts.min()).item()) <= 1
        sampled = torch.cat(batches)
        sampled_counts = torch.bincount(labels[sampled], minlength=2)
        assert sampled_counts.tolist() == [5, 5]

    def test_class_weights_are_smoothed_clipped_and_ignore_absent_classes(self):
        from fed_learning.clients.denice_client import build_denice_class_weights

        labels = torch.tensor([0] * 10 + [1] * 2)
        weights, audit = build_denice_class_weights(
            labels,
            4,
            mode="inverse_frequency",
            smoothing=1.0,
            effective_beta=0.999,
            clip_min=0.25,
            clip_max=2.0,
        )

        assert weights is not None
        assert weights[1] > weights[0]
        assert weights[2:].tolist() == [1.0, 1.0]
        assert 0.25 <= audit["weight_min"] <= audit["weight_max"] <= 2.0
        assert audit["class_counts"] == {0: 10, 1: 2}

    def test_d3_controls_reject_unregistered_combination(self):
        from fed_learning.clients.denice_client import normalize_denice_imbalance_config

        with pytest.raises(ValueError, match="one-factor"):
            normalize_denice_imbalance_config(
                {
                    "denice_batch_sampling": "class_balanced",
                    "denice_class_weight_mode": "effective_number",
                }
            )

    def test_d3_analyzer_requires_aligned_trace_and_gates_candidate(self, tmp_path):
        from tools.analyze_denice_d3 import analyze

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "metadata.json").write_text(json.dumps({
            "task_structure": {"task_classes": {"0": [0], "1": [1]}}
        }), encoding="utf-8")
        trace_base = {"source_test_indices": [0, 1, 2, 3], "client_ids": [0, 0, 1, 1],
                      "targets": [0, 0, 1, 1], "predictions": [0, 1, 1, 0]}
        trace_win = {**trace_base, "predictions": [0, 0, 1, 1]}
        def artifact(trace):
            return {"metrics": {"f1_macro": 0.5, "debug": {"prediction_trace": trace,
                "per_class": {"0": {"accuracy": 1.0}, "1": {"accuracy": 1.0}}}}}
        manifest = {"seed": 3, "data_dir": str(data_dir), "task_end": 1, "variants": {}}
        for name, trace in (("baseline", trace_base), ("class_balanced_batches", trace_win),
                            ("effective_number_ce", trace_win)):
            evaluation = tmp_path / name
            evaluation.mkdir()
            for policy in ("e3_oracle_routed_system_ceiling", "e4_pred_hard"):
                payload = artifact(trace)
                payload["metrics"]["f1_macro"] = 0.5 if name == "baseline" else 1.0
                (evaluation / f"coverage_aware_local_{policy}.json").write_text(
                    json.dumps(payload), encoding="utf-8")
            manifest["variants"][name] = {"evaluation_dir": str(evaluation)}
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report = analyze(manifest_path, bootstrap_replicates=30)
        assert report["decision"] == "CANDIDATE_FOR_CONFIRMATION_SEED"
        assert report["gates"]["bootstrap_positive_candidates"]
        assert report["recommended_candidate"] in {
            "class_balanced_batches", "effective_number_ce"
        }


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
            trainer=trainer,
            epochs=1,
            batch_size=16,
            lr=1e-3,
            is_last_task=True,
            denice_batch_sampling="class_balanced",
        )
        assert "adapter_registry" in result
        assert result["adapter_param_count"] > 0
        assert result["imbalance_control"]["batch_sampling"] == "class_balanced"
        assert result["imbalance_control"]["sampling_epochs"]
        assert np.isfinite(result["loss"])
