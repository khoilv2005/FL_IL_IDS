from collections import OrderedDict
import os

import torch

from fed_learning.clients.client import FederatedClient
from fed_learning.servers.plexus_server import PlexusServer
from fed_learning.strategies import get_strategy, list_strategies
from fed_learning.strategies.federated.plexus import SampleManager, PlexusAggregator
from fed_learning.plexus.aggregator import PlexusAggregator as PurePlexusAggregator
from fed_learning.plexus.orchestrator import NodeWrapper
from fed_learning.models.cnn_gru import CNN_GRU_Model
from fed_learning.training import decentralized_plexus_il as decentralized_il


def test_plexus_strategy_registered():
    trainer, aggregator = get_strategy(
        "plexus",
        plexus_sample_size=4,
        plexus_num_aggregators=1,
        plexus_success_fraction=0.8,
    )

    assert trainer.__class__.__name__ == "PlexusTrainer"
    assert aggregator.__class__.__name__ == "PlexusAggregator"
    assert "plexus" in list_strategies()


def test_sample_manager_keeps_aggregator_inside_sample():
    manager = SampleManager(sample_size=4, num_aggregators=1)
    peer_ids = list(range(10))
    bandwidths = {pid: float(pid) for pid in peer_ids}

    expected_sample = manager.get_ordered_sample_list(7, peer_ids)[:4]
    sample = manager.get_sample(7, peer_ids, bandwidths)
    aggregators = manager.get_aggregators(7, peer_ids, bandwidths)

    assert sample == expected_sample
    assert len(sample) == 4
    assert aggregators[0] in sample
    assert aggregators[0] == max(sample, key=lambda pid: bandwidths[pid])


def test_plexus_aggregator_uses_all_results_passed_by_server():
    aggregator = PlexusAggregator(
        sample_size=4,
        num_aggregators=1,
        success_fraction=0.5,
    )
    results = [
        {
            "client_id": i,
            "num_samples": 1,
            "params": OrderedDict([("w", torch.tensor([value], dtype=torch.float32))]),
        }
        for i, value in enumerate([0.0, 10.0, 10.0, 10.0])
    ]

    averaged = aggregator.aggregate(results, global_params=None)

    assert torch.allclose(averaged["w"], torch.tensor([7.5]))


def test_pure_plexus_threshold_uses_paper_success_fraction():
    aggregator = PurePlexusAggregator(sample_size=2, success_fraction=0.8)

    assert aggregator.get_threshold() == 1


def test_decentralized_plexus_nodes_do_not_share_model_instance():
    template = CNN_GRU_Model((16, 1), num_classes=2)
    node_a = NodeWrapper(
        node_id=0,
        X_train=torch.randn(4, 16, 1),
        y_train=torch.tensor([0, 1, 0, 1]),
        bandwidth=1.0,
        model_template=template,
        device="cpu",
        batch_size=4,
    )
    node_b = NodeWrapper(
        node_id=1,
        X_train=torch.randn(4, 16, 1),
        y_train=torch.tensor([0, 1, 0, 1]),
        bandwidth=1.0,
        model_template=template,
        device="cpu",
        batch_size=4,
    )

    assert node_a.model is not node_b.model
    assert node_a.model is not template
    assert node_b.model is not template


def test_plexus_server_train_round_smoke():
    torch.manual_seed(0)
    clients = [
        FederatedClient(
            client_id=i,
            X_train=torch.randn(4, 16, 1),
            y_train=torch.tensor([0, 1, 0, 1]),
        )
        for i in range(4)
    ]
    test_data = {
        "X_test": torch.randn(4, 16, 1),
        "y_test": torch.tensor([0, 1, 0, 1]),
    }
    config = {
        "algorithm": "plexus",
        "input_shape": (16, 1),
        "num_classes": 2,
        "num_gpus": 0,
        "local_epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.001,
        "plexus_sample_size": 10,
        "plexus_num_aggregators": 1,
        "plexus_success_fraction": 0.5,
        "plexus_scale_clients": False,
        "seed": 123,
    }

    server = PlexusServer(clients, test_data, config)
    expected_sample = SampleManager(10, 1).get_sample(
        1, [client.client_id for client in clients], server.client_bandwidths
    )
    metrics = server.train_round(verbose=False)
    updated_clients = [
        client_id
        for client_id, (round_id, _) in server.population_view.view.items()
        if round_id == 1
    ]

    assert server._round == 1
    assert sorted(updated_clients) == sorted(expected_sample[:2])
    assert metrics["train_loss"] >= 0.0

def test_decentralized_plexus_il_writes_fed_il_output_contract(tmp_path, monkeypatch):
    class FakeIncrementalDataLoader:
        input_shape = (16, 1)

        def __init__(self, data_dir):
            self.data_dir = data_dir

        def get_num_tasks(self):
            return 2

        def get_task_classes(self, task_id):
            return [task_id]

        def get_all_client_ids(self):
            return [0, 1]

        def get_client_data(self, client_id, task_id):
            torch.manual_seed(10 + client_id + task_id)
            return torch.randn(4, 16, 1), torch.full((4,), task_id, dtype=torch.long)

        def get_test_data(self, task_id, cumulative=True):
            X_parts = []
            y_parts = []
            for tid in range(task_id + 1):
                torch.manual_seed(100 + tid)
                X_parts.append(torch.randn(4, 16, 1))
                y_parts.append(torch.full((4,), tid, dtype=torch.long))
            return torch.cat(X_parts), torch.cat(y_parts)

    monkeypatch.setattr(decentralized_il, "IncrementalDataLoader", FakeIncrementalDataLoader)
    monkeypatch.setattr(
        "fed_learning.training.task_loop._evaluate_and_visualize",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "fed_learning.training.task_loop._generate_fcil_report",
        lambda *args, **kwargs: None,
    )

    config = {
        "mode": "decentralized",
        "algorithm": "plexus",
        "data_dir": "unused",
        "output_dir": str(tmp_path / "run"),
        "total_classes": 2,
        "task_start": 0,
        "task_end": 1,
        "rounds_per_task": 1,
        "local_epochs": 1,
        "batch_size": 4,
        "learning_rate": 0.001,
        "plexus_sample_size": 2,
        "plexus_success_fraction": 0.5,
        "eval_every": 1,
        "round_checkpoint_every": 1,
        "seed": 123,
        "random_seed": 123,
    }

    history = decentralized_il.run_decentralized_plexus_il(config)
    output_dirs = list(tmp_path.glob("run_plexus_*"))
    assert len(output_dirs) == 1
    output_dir = output_dirs[0]

    for filename in (
        "config.json",
        "results.json",
        "round_metrics.json",
        "task_metrics.json",
        "phase_summary.json",
        "checkpoint_task_0.pt",
        "checkpoint_task_1.pt",
        "checkpoint_task_0_round_0.pt",
        "checkpoint_task_1_round_1.pt",
    ):
        assert (output_dir / filename).exists()

    assert len(history["round_metrics"]) == 2
    assert len(history["task_accuracies"]) == 2
    assert set(history) == {"task_accuracies", "task_forgetting", "round_metrics"}
    assert set(history["round_metrics"][0]) == {
        "task",
        "round",
        "train_loss",
        "round_time",
        "test_loss",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "f1_weighted",
        "avg_forgetting",
        "evaluated",
    }

def test_decentralized_plexus_il_resumes_second_phase(tmp_path, monkeypatch):
    class FakeIncrementalDataLoader:
        input_shape = (16, 1)

        def __init__(self, data_dir):
            self.data_dir = data_dir

        def get_num_tasks(self):
            return 2

        def get_task_classes(self, task_id):
            return [task_id]

        def get_all_client_ids(self):
            return [0, 1]

        def get_client_data(self, client_id, task_id):
            torch.manual_seed(20 + client_id + task_id)
            return torch.randn(4, 16, 1), torch.full((4,), task_id, dtype=torch.long)

        def get_test_data(self, task_id, cumulative=True):
            X_parts = []
            y_parts = []
            for tid in range(task_id + 1):
                torch.manual_seed(200 + tid)
                X_parts.append(torch.randn(4, 16, 1))
                y_parts.append(torch.full((4,), tid, dtype=torch.long))
            return torch.cat(X_parts), torch.cat(y_parts)

    monkeypatch.setattr(decentralized_il, "IncrementalDataLoader", FakeIncrementalDataLoader)
    monkeypatch.setattr(
        "fed_learning.training.task_loop._evaluate_and_visualize",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "fed_learning.training.task_loop._generate_fcil_report",
        lambda *args, **kwargs: None,
    )

    base_config = {
        "mode": "decentralized",
        "algorithm": "plexus",
        "data_dir": "unused",
        "output_dir": str(tmp_path / "phase"),
        "total_classes": 2,
        "rounds_per_task": 1,
        "local_epochs": 1,
        "batch_size": 4,
        "eval_batch_size": 4,
        "learning_rate": 0.001,
        "plexus_sample_size": 2,
        "plexus_success_fraction": 0.5,
        "eval_every": 1,
        "round_checkpoint_every": 1,
        "seed": 321,
        "random_seed": 321,
    }

    phase1_config = {
        **base_config,
        "task_start": 0,
        "task_end": 0,
        "save_resume_after_task": 0,
    }
    phase1_history = decentralized_il.run_decentralized_plexus_il(phase1_config)
    output_dir = next(tmp_path.glob("phase_plexus_*"))
    resume_path = output_dir / "continuation_state_task_0.pt"

    assert resume_path.exists()
    assert len(phase1_history["task_accuracies"]) == 1

    phase2_config = {
        **base_config,
        "task_start": 1,
        "task_end": 1,
        "save_resume_after_task": None,
        "resume_state_path": str(resume_path),
        "resume_output_dir": str(output_dir),
    }
    phase2_history = decentralized_il.run_decentralized_plexus_il(phase2_config)

    assert [entry["task"] for entry in phase2_history["task_accuracies"]] == [0, 1]
    assert len(phase2_history["round_metrics"]) == 2
    assert (output_dir / "config_phase_resume.json").exists()
    assert os.path.exists(output_dir / "checkpoint_task_1.pt")
