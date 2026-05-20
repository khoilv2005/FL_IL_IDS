from collections import OrderedDict

import torch

from fed_learning.clients.client import FederatedClient
from fed_learning.servers.plexus_server import PlexusServer
from fed_learning.strategies import get_strategy, list_strategies
from fed_learning.strategies.federated.plexus import SampleManager, PlexusAggregator
from fed_learning.plexus.aggregator import PlexusAggregator as PurePlexusAggregator


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
