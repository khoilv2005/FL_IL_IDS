from collections import OrderedDict
import shutil

import torch

from fed_learning.strategies.fed_incremental.cgofed import (
    CGoFedAggregator,
    CGoFedTrainer,
    is_model_artifact,
    load_model_state,
)
from fed_learning.clients.fedcbdr_client import FedCBDRClient
from fed_learning.strategies.incremental.ewc import EWCTrainer
from fed_learning.training.resume_state import (
    load_continuation_state,
    restore_client_state,
    restore_aggregator_state,
    restore_trainer_state,
    save_continuation_state,
    snapshot_aggregator_state,
    snapshot_client_state,
    snapshot_trainer_state,
)


class _DummyBuffer:
    def __init__(self):
        self.total_samples = 0
        self.class_buffers = {}


class _DummyClient:
    def __init__(self):
        self.client_id = 0
        self.X_train = torch.randn(4, 2)
        self.y_train = torch.tensor([0, 1, 0, 1])
        self.num_samples = 4
        self.model = None
        self.device = None
        self.use_amp = False
        self.current_task = 0
        self.seen_classes = set()
        self.old_model = None
        self.old_model_state = OrderedDict()
        self.replay_buffer = _DummyBuffer()


def test_resume_state_roundtrip_for_nested_client_objects():
    source = _DummyClient()
    source.current_task = 2
    source.seen_classes = {0, 1, 2}
    source.old_model_state = OrderedDict(weight=torch.tensor([1.0, 2.0]))
    source.replay_buffer.total_samples = 7
    source.replay_buffer.class_buffers = {
        0: {"X": torch.ones(2, 3), "y": torch.zeros(2, dtype=torch.long)}
    }

    state = snapshot_client_state(source)

    target = _DummyClient()
    target.current_task = 99
    restore_client_state(target, state)

    assert target.current_task == 2
    assert target.seen_classes == {0, 1, 2}
    assert torch.equal(target.old_model_state["weight"], torch.tensor([1.0, 2.0]))
    assert target.replay_buffer.total_samples == 7
    assert 0 in target.replay_buffer.class_buffers
    assert torch.equal(
        target.replay_buffer.class_buffers[0]["X"], torch.ones(2, 3)
    )


def test_ewc_resume_state_rebuilds_latest_fisher(tmp_path):
    first_temp = tmp_path / "ewc_phase1"
    second_temp = tmp_path / "ewc_phase2"

    trainer = EWCTrainer(temp_dir=str(first_temp))
    trainer.current_task = 2
    trainer.seen_classes = {0, 1, 2}
    trainer.best_acc_per_task = {0: 0.9, 1: 0.8}
    trainer.current_acc_per_task = {0: 0.85, 1: 0.75}
    trainer.last_af = 0.05
    trainer._cached_fisher_acc = {"layer.weight": torch.ones(3)}
    trainer._cached_optimal_params = {"layer.weight": torch.zeros(3)}

    state = trainer.get_resume_state()

    restored = EWCTrainer(temp_dir=str(second_temp))
    restored.load_resume_state(state)
    restored.set_task(3, [3])

    fisher = restored._get_prev_fisher_acc()

    assert fisher is not None
    assert torch.equal(fisher["layer.weight"], torch.ones(3))
    assert 2 in restored.ewc_data
    assert restored.last_af == 0.05
    assert 3 in restored.seen_classes


def test_cgofed_trainer_resume_rebuilds_basis_files(tmp_path):
    phase1 = tmp_path / "cgofed_phase1"
    phase2 = tmp_path / "cgofed_phase2"
    phase1.mkdir()

    basis_path = phase1 / "task_0_fc1_basis.pt"
    importance_path = phase1 / "task_0_fc1_importance.pt"
    torch.save(torch.randn(8, 3), basis_path)
    torch.save(torch.tensor([0.5, 0.6, 0.7]), importance_path)

    trainer = CGoFedTrainer(temp_dir=str(phase1))
    trainer.current_task = 2
    trainer.seen_classes = {0, 1, 2}
    trainer.layer_bases = {
        "task_0": {
            "fc1": {
                "basis": str(basis_path),
                "importance": str(importance_path),
                "shape": (8, 3),
            }
        }
    }

    state = snapshot_trainer_state(trainer)

    restored = CGoFedTrainer(temp_dir=str(phase2))
    restore_trainer_state(restored, state)

    restored_info = restored.layer_bases["task_0"]["fc1"]
    assert restored_info["basis"].startswith(str(phase2))
    assert restored_info["importance"].startswith(str(phase2))
    assert torch.equal(
        torch.load(restored_info["basis"], map_location="cpu"),
        torch.load(basis_path, map_location="cpu"),
    )
    assert torch.equal(
        torch.load(restored_info["importance"], map_location="cpu"),
        torch.load(importance_path, map_location="cpu"),
    )


def test_cgofed_aggregator_resume_materializes_historical_models():
    model_state = OrderedDict(weight=torch.tensor([1.0, 2.0]))
    aggregator = CGoFedAggregator(rounds_per_task=20)
    aggregator.current_task = 2
    aggregator.task_global_models = {0: model_state}
    aggregator.client_historical_models = {7: {0: model_state}}
    aggregator._current_historical_models = {0: model_state}
    aggregator.task_representations = {0: torch.tensor([0.1, 0.2])}
    aggregator.task_representation_matrices = {
        0: {
            "_artifact_type": "cgofed_representation_artifact",
            "signature": torch.tensor([1.0, 2.0]),
            "shape": (4, 2),
            "mean_vector": torch.tensor([0.5, 0.5]),
            "mean_norm": 0.7071,
        }
    }

    state = snapshot_aggregator_state(aggregator)

    restored = CGoFedAggregator(rounds_per_task=20)
    restore_aggregator_state(restored, state)

    assert isinstance(restored.task_global_models[0], OrderedDict)
    assert isinstance(restored.client_historical_models[7][0], OrderedDict)
    assert isinstance(restored._current_historical_models[0], OrderedDict)
    assert torch.equal(restored.task_global_models[0]["weight"], model_state["weight"])


def test_cgofed_aggregator_resume_accepts_legacy_model_artifact_refs(tmp_path):
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    model_state = OrderedDict(weight=torch.tensor([1.0, 2.0]))
    artifact_path = history_dir / "task0.pt"
    torch.save(model_state, artifact_path)

    legacy_state = {
        "cross_task_weight": 0.3,
        "top_k": 2,
        "rounds_per_task": 20,
        "current_task": 2,
        "_round_in_task": 0,
        "client_representations": {},
        "task_global_models": {
            0: {
                "_artifact_type": "cgofed_model_artifact",
                "path": str(artifact_path),
                "key": "task_0",
            }
        },
        "client_historical_models": {
            7: {
                0: {
                    "_artifact_type": "cgofed_model_artifact",
                    "path": str(artifact_path),
                    "key": "client_7_task_0",
                }
            }
        },
        "task_representations": {},
        "task_representation_matrices": {},
        "_current_similarity_weights": {},
        "_current_historical_models": {
            0: {
                "_artifact_type": "cgofed_model_artifact",
                "path": str(artifact_path),
                "key": "task_0",
            }
        },
    }

    restored = CGoFedAggregator(rounds_per_task=20)
    restored.load_resume_state(legacy_state)

    assert is_model_artifact(restored.task_global_models[0])
    assert is_model_artifact(restored.client_historical_models[7][0])
    assert is_model_artifact(restored._current_historical_models[0])
    assert torch.equal(load_model_state(restored.task_global_models[0])["weight"], model_state["weight"])
    assert torch.equal(
        load_model_state(restored.client_historical_models[7][0])["weight"], model_state["weight"]
    )
    assert torch.equal(
        load_model_state(restored._current_historical_models[0])["weight"], model_state["weight"]
    )


def test_cgofed_continuation_bundles_model_artifacts_for_lazy_resume(tmp_path):
    history_dir = tmp_path / "history"
    history_dir.mkdir()
    model_state = OrderedDict(weight=torch.tensor([1.0, 2.0]))
    artifact_path = history_dir / "task0.pt"
    torch.save(model_state, artifact_path)

    aggregator = CGoFedAggregator(rounds_per_task=20)
    aggregator.task_global_models = {
        0: {
            "_artifact_type": "cgofed_model_artifact",
            "path": str(artifact_path),
            "key": "task_0",
        }
    }
    aggregator.client_historical_models = {7: {0: aggregator.task_global_models[0]}}
    aggregator._current_historical_models = {0: aggregator.task_global_models[0]}

    state = {
        "aggregator_state": snapshot_aggregator_state(aggregator),
    }

    output_dir = tmp_path / "resume_bundle"
    continuation_path = save_continuation_state(str(output_dir), 3, state)

    moved_dir = tmp_path / "moved_bundle"
    moved_dir.mkdir()
    moved_path = moved_dir / "continuation_state_task_3.pt"
    shutil.copy2(continuation_path, moved_path)
    shutil.copytree(
        output_dir / "continuation_artifacts_task_3",
        moved_dir / "continuation_artifacts_task_3",
    )

    artifact_path.unlink()
    shutil.rmtree(output_dir)
    loaded = load_continuation_state(str(moved_path))

    restored = CGoFedAggregator(rounds_per_task=20)
    restore_aggregator_state(restored, loaded["aggregator_state"])

    restored_ref = restored.task_global_models[0]
    assert is_model_artifact(restored_ref)
    assert str(moved_dir) in restored_ref["path"]
    assert torch.equal(load_model_state(restored_ref)["weight"], model_state["weight"])


def test_fedcbdr_client_resume_keeps_replay_buffer_without_raw_dataset_dump():
    client = FedCBDRClient(
        client_id=3,
        X_train=torch.randn(6, 4),
        y_train=torch.tensor([0, 0, 1, 1, 2, 2]),
        buffer_size=10,
        leverage_rank=7,
    )
    client.current_task = 2
    client.current_classes = {4, 5}
    client.seen_classes = {0, 1, 2, 4, 5}
    client.replay_buffer.add_samples(
        X=torch.randn(3, 4),
        y=torch.tensor([0, 1, 1]),
        importance_scores=torch.tensor([0.2, 0.3, 0.5]),
    )

    state = snapshot_client_state(client)

    assert "X_original" not in state["attrs"]
    assert "y_original" not in state["attrs"]
    assert "replay_buffer" in state["attrs"]

    restored = FedCBDRClient(
        client_id=3,
        X_train=torch.zeros(1, 4),
        y_train=torch.tensor([0]),
        buffer_size=1,
        leverage_rank=1,
    )
    restore_client_state(restored, state)

    assert restored.current_task == 2
    assert restored.current_classes == {4, 5}
    assert restored.seen_classes == {0, 1, 2, 4, 5}
    assert restored.leverage_calculator.rank == 7
    assert restored.replay_buffer.total_samples == 3
    assert restored.replay_buffer.num_classes == 2
