"""
Tests for incremental learning strategies: EWC, FedLwF, FedCBDR.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn

from fed_learning.clients.cgofed_client import CGoFedClient
from fed_learning.clients.client import FederatedClient
from fed_learning.clients.fedlwf_client import FedLwFClient
from fed_learning.factories.client_factory import create_client
from fed_learning.servers.fedcbdr_server import FedCBDRServer
from fed_learning.servers.refed_server import ReFedServer
from fed_learning.strategies.fed_incremental.ewc import (
    EWCMixin,
    FedAvgEWCTrainer,
    FedProxEWCTrainer,
)
from fed_learning.strategies.fed_incremental.fedlwf import FedLwFTrainer, FedLwFAggregator
from fed_learning.strategies.fed_incremental.fedcbdr import (
    FedCBDRTrainer,
    FedCBDRAggregator,
)
from fed_learning.training.task_loop import _refresh_server_clients
from helpers import make_simple_model, make_client_results


class TestEWC:
    """Test Elastic Weight Consolidation."""

    def test_ewc_mixin_init(self):
        """EWCMixin should initialize with correct parameters."""
        trainer = FedAvgEWCTrainer(ewc_lambda=500.0, fisher_samples=100)
        assert trainer.ewc_lambda == 500.0
        assert trainer.fisher_samples == 100

    def test_ewc_set_task(self):
        """EWC set_task should update current_task and seen_classes."""
        trainer = FedAvgEWCTrainer()
        trainer.set_task(0, [0, 1, 2])
        assert trainer.current_task == 0
        assert trainer.seen_classes == {0, 1, 2}

    def test_fedprox_ewc_has_mu(self):
        """FedProxEWCTrainer should have mu for proximal term."""
        trainer = FedProxEWCTrainer(mu=0.5)
        assert trainer.mu == 0.5

    def test_ewc_compute_loss_without_fisher(self):
        """Without stored Fisher, EWC loss should equal base loss."""
        trainer = FedAvgEWCTrainer(ewc_lambda=1000.0)
        trainer.set_task(0, [0, 1, 2])

        model = make_simple_model()
        x = torch.randn(4, 10)
        output = model(x)
        target = torch.randint(0, 5, (4,))

        loss = trainer.compute_loss(model, output, target)
        loss_ce = trainer._seen_class_cross_entropy(output, target)
        assert abs(loss.item() - loss_ce.item()) < 1e-5

    def test_ewc_online_ewc_gamma(self):
        """Online EWC should use gamma parameter."""
        trainer = FedAvgEWCTrainer(online_ewc=True, gamma=0.8)
        assert trainer.online_ewc is True
        assert trainer.gamma == 0.8

    def test_ewc_uses_plain_federated_client(self):
        """EWC should not route through the CGoFed client implementation."""
        client = create_client(
            0,
            torch.randn(8, 32),
            torch.randint(0, 3, (8,)),
            {"algorithm": "ewc"},
        )
        assert isinstance(client, FederatedClient)
        assert not isinstance(client, CGoFedClient)

    def test_federated_ewc_uses_plain_federated_client(self):
        """FedAvg/FedProx EWC should avoid CGoFed-only representation logic."""
        for algorithm in ("fedavg_ewc", "fedprox_ewc"):
            client = create_client(
                0,
                torch.randn(8, 32),
                torch.randint(0, 3, (8,)),
                {"algorithm": algorithm},
            )
            assert isinstance(client, FederatedClient)
            assert not isinstance(client, CGoFedClient)


class TestFedLwF:
    """Test Federated Learning without Forgetting."""

    def test_fedlwf_trainer_init(self):
        """FedLwFTrainer should initialize with correct parameters."""
        trainer = FedLwFTrainer(lwf_alpha=2.0, temperature=3.0)
        assert trainer.lwf_alpha == 2.0
        assert trainer.temperature == 3.0

    def test_fedlwf_set_task(self):
        """FedLwF set_task should track classes."""
        trainer = FedLwFTrainer()
        trainer.set_task(0, [0, 1, 2])
        assert trainer.current_task == 0
        assert 0 in trainer.seen_classes

    def test_fedlwf_first_task_no_distillation(self):
        """On first task, loss should just be CE."""
        trainer = FedLwFTrainer()
        trainer.set_task(0, [0, 1, 2])

        model = make_simple_model()
        x = torch.randn(4, 10)
        output = model(x)
        target = torch.randint(0, 5, (4,))

        loss = trainer.compute_loss(model, output, target)
        loss_ce = trainer._seen_class_cross_entropy(output, target)
        assert abs(loss.item() - loss_ce.item()) < 1e-5

    def test_fedlwf_aggregator(self):
        """FedLwFAggregator should perform weighted average."""
        aggregator = FedLwFAggregator()
        results = make_client_results(3)
        agg = aggregator.aggregate(results)
        assert isinstance(agg, OrderedDict)

    def test_lwf_uses_fedlwf_client(self):
        """Local LwF should use the dedicated distillation-aware client."""
        client = create_client(
            0,
            torch.randn(8, 32),
            torch.randint(0, 3, (8,)),
            {"algorithm": "lwf"},
        )
        assert isinstance(client, FedLwFClient)


class TestFedCBDR:
    """Test Class-wise Balancing Data Replay."""

    def test_fedcbdr_trainer_init(self):
        """FedCBDRTrainer should initialize with temperature params."""
        trainer = FedCBDRTrainer(
            tau_old=0.9,
            tau_new=1.1,
            omega_old=1.1,
            omega_new=0.9,
        )
        assert trainer.tau_old == 0.9
        assert trainer.tau_new == 1.1
        assert trainer.omega_old == 1.1
        assert trainer.omega_new == 0.9

    def test_fedcbdr_aggregator(self):
        """FedCBDRAggregator should perform weighted average."""
        aggregator = FedCBDRAggregator()
        results = make_client_results(3)
        agg = aggregator.aggregate(results)
        assert isinstance(agg, OrderedDict)

    def test_fedcbdr_server_update_clients(self):
        """FedCBDRServer should support task-loop client refresh across tasks."""
        clients_a = [FederatedClient(0, torch.randn(4, 32), torch.randint(0, 2, (4,)))]
        clients_b = [FederatedClient(1, torch.randn(5, 32), torch.randint(0, 2, (5,)))]
        server = FedCBDRServer(
            clients=clients_a,
            test_data={"X_test": torch.randn(8, 32), "y_test": torch.randint(0, 2, (8,))},
            config={"input_shape": (32,), "num_classes": 34, "num_gpus": 0},
        )

        server.update_clients(clients_b)
        assert server.clients is clients_b


class TestReFedServer:
    """Test Re-Fed server task transition helpers."""

    def test_refed_server_update_clients(self):
        """ReFedServer should support task-loop client refresh across tasks."""
        clients_a = [FederatedClient(0, torch.randn(4, 32), torch.randint(0, 2, (4,)))]
        clients_b = [FederatedClient(1, torch.randn(5, 32), torch.randint(0, 2, (5,)))]
        server = ReFedServer(
            clients=clients_a,
            test_data={"X_test": torch.randn(8, 32), "y_test": torch.randint(0, 2, (8,))},
            config={"input_shape": (32,), "num_classes": 34, "num_gpus": 0},
        )

        server.update_clients(clients_b)
        assert server.clients is clients_b


class TestTaskLoopServerRefresh:
    """Test task-loop fallback when a custom server lacks update_clients()."""

    def test_refresh_server_clients_uses_clients_attribute_fallback(self):
        class DummyServer:
            def __init__(self, clients):
                self.clients = clients

        clients_a = [object()]
        clients_b = [object(), object()]
        server = DummyServer(clients_a)

        refreshed = _refresh_server_clients(
            server=server,
            clients=clients_b,
            config={},
            test_data={},
            task_config={},
        )

        assert refreshed is server
        assert server.clients is clients_b
