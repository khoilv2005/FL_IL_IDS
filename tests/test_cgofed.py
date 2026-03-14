"""
Tests for CGoFed strategy and CGoFedServer.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn

from fed_learning.strategies.fed_incremental.cgofed import CGoFedTrainer, CGoFedAggregator
from fed_learning.clients.cgofed_client import CGoFedClient
from fed_learning.servers.cgofed_server import CGoFedServer
from helpers import make_simple_model, make_client_results


class TestCGoFed:
    """Test CGoFed constrained gradient optimization."""

    def test_cgofed_separate_mu_params(self):
        """CGoFed should maintain separate mu (proximal) and mu_projection."""
        trainer = CGoFedTrainer(mu=1.5, mu_projection=5.0)
        assert trainer.mu == 1.5
        assert trainer.mu_projection == 5.0

    def test_cgofed_mu_projection_defaults_to_mu(self):
        """mu_projection should default to mu if not specified."""
        trainer = CGoFedTrainer(mu=2.0)
        assert trainer.mu_projection == 2.0

    def test_cgofed_local_regularization(self):
        """Paper Eq. 14: CGoFed has local cross-task regularization A(Θ)."""
        trainer = CGoFedTrainer(mu=0.0, lambda_cross_task=0.1)
        trainer.set_task(1, [0, 1, 2, 3, 4])
        model = make_simple_model()

        hist_model = make_simple_model()
        historical_models = {0: copy.deepcopy(hist_model.state_dict())}
        similarity_weights = {0: 1.0}

        x = torch.randn(4, 10)
        output = model(x)
        target = torch.randint(0, 5, (4,))

        loss = trainer.compute_loss(
            model,
            output,
            target,
            historical_models=historical_models,
            similarity_weights=similarity_weights,
        )
        loss_ce = nn.CrossEntropyLoss()(output, target)
        assert loss.item() > loss_ce.item(), (
            f"CGoFed Eq.14 should have A(Θ) regularization"
        )

    def test_cgofed_no_proximal_term(self):
        """Paper Eq. 14: CGoFed does NOT have proximal term."""
        trainer = CGoFedTrainer(mu=0.0)
        model = make_simple_model()

        x = torch.randn(4, 10)
        output = model(x)
        target = torch.randint(0, 5, (4,))

        loss = trainer.compute_loss(model, output, target)
        loss_ce = nn.CrossEntropyLoss()(output, target)
        assert abs(loss.item() - loss_ce.item()) < 1e-6

    def test_cgofed_set_task(self):
        """set_task should track task and seen classes."""
        trainer = CGoFedTrainer()
        trainer.set_task(0, [0, 1, 2])
        assert trainer.current_task == 0
        assert trainer.seen_classes == {0, 1, 2}

        trainer.set_task(1, [3, 4])
        assert trainer.current_task == 1
        assert trainer.seen_classes == {0, 1, 2, 3, 4}

    def test_cgofed_mu_coefficient_decay(self):
        """mu_coefficient should decay with lambda_decay^(task - t_reset)."""
        trainer = CGoFedTrainer(lambda_decay=0.5)
        trainer.set_task(0, [0, 1, 2])
        assert trainer.mu_coefficient == 1.0

        trainer.set_task(1, [3, 4])
        assert abs(trainer.mu_coefficient - 0.5) < 1e-6

    def test_cgofed_forgetting_reset(self):
        """When AF > theta, mu_coefficient should reset to 1.0."""
        trainer = CGoFedTrainer(theta_threshold=0.1, lambda_decay=0.5)
        trainer.set_task(0, [0, 1])
        trainer.best_acc_per_task[0] = 0.9

        trainer.set_task(1, [2, 3])
        trainer.update_forgetting({0: 0.5, 1: 0.8})

        assert trainer.mu_coefficient == 1.0
        assert trainer.t_reset == 1

    def test_cgofed_pre_step_skip_first_task(self):
        """pre_step should do nothing on the first task."""
        trainer = CGoFedTrainer()
        model = make_simple_model()
        trainer.set_task(0, [0, 1, 2])
        trainer.pre_step(model)  # Should not raise

    def test_cgofed_aggregator(self):
        """CGoFedAggregator should handle basic aggregation."""
        aggregator = CGoFedAggregator(cross_task_weight=0.1, top_k=2)
        results = make_client_results(3)
        agg = aggregator.aggregate(results)
        assert isinstance(agg, OrderedDict)


class TestCGoFedServer:
    """Test CGoFedServer integration logic for Eq.12/Eq.14 paths."""

    @staticmethod
    def _make_server(clients, test_data=None):
        if test_data is None:
            test_data = {
                "X_test": torch.randn(8, 32),
                "y_test": torch.randint(0, 4, (8,)),
            }
        config = {
            "algorithm": "cgofed",
            "input_shape": (32,),
            "num_classes": 4,
            "num_gpus": 0,
            "top_k": 1,
            "rounds_per_task": 1,
        }
        return CGoFedServer(clients, test_data, config)

    def test_set_task_eq14_per_client_reg_info(self, monkeypatch):
        """Eq.14 should prepare different historical selections per client."""
        clients = [
            CGoFedClient(0, torch.randn(8, 32), torch.randint(0, 2, (8,))),
            CGoFedClient(1, torch.randn(8, 32), torch.randint(0, 2, (8,))),
        ]
        server = self._make_server(clients)
        agg = CGoFedAggregator(top_k=1, rounds_per_task=1)
        server.aggregator = agg

        base = server.get_global_params()
        model_a = OrderedDict(
            (k, (v + 0.1) if v.dtype.is_floating_point else v.clone())
            for k, v in base.items()
        )
        model_b = OrderedDict(
            (k, (v + 0.2) if v.dtype.is_floating_point else v.clone())
            for k, v in base.items()
        )

        agg.client_historical_models = {
            10: {0: model_a},
            11: {0: model_b},
        }
        agg.client_representations = {
            10: {0: torch.tensor([[1.0, 0.0], [0.0, 1.0]])},
            11: {0: torch.tensor([[0.0, 1.0], [1.0, 0.0]])},
        }

        def fake_rep(client, num_samples=100):
            if client.client_id == 0:
                return torch.tensor([[1.0, 0.0], [0.0, 1.0]])
            return torch.tensor([[0.0, 1.0], [1.0, 0.0]])

        monkeypatch.setattr(server, "_compute_client_task_representation", fake_rep)
        server.set_task(1, [2, 3], [0, 1, 2, 3])

        assert 0 in server._client_reg_info
        assert 1 in server._client_reg_info

        keys_client0 = list(server._client_reg_info[0]["historical_models"].keys())
        keys_client1 = list(server._client_reg_info[1]["historical_models"].keys())
        assert len(keys_client0) == 1
        assert len(keys_client1) == 1
        assert keys_client0[0] != keys_client1[0]

    def test_set_task_does_not_depend_on_test_data_for_eq14(self):
        """Eq.14 setup should rely on client train data, not test_data."""
        clients = [CGoFedClient(0, torch.randn(8, 32), torch.randint(0, 2, (8,)))]
        server = self._make_server(clients, test_data={"y_test": torch.tensor([0, 1])})
        server.aggregator = CGoFedAggregator(top_k=1, rounds_per_task=1)

        server.set_task(1, [2, 3], [0, 1, 2, 3])
        assert isinstance(server._client_reg_info, dict)

    def test_eq12_personalization_uses_matrix_similarity(self):
        """Eq.12 personalization should be driven by matrix distance."""
        clients = [CGoFedClient(0, torch.randn(4, 32), torch.randint(0, 2, (4,)))]
        server = self._make_server(clients)
        server.eq12_self_weight = 0.0

        results = [
            {
                "client_id": 0,
                "params": OrderedDict({"w": torch.tensor([0.0])}),
                "representation": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            },
            {
                "client_id": 1,
                "params": OrderedDict({"w": torch.tensor([2.0])}),
                "representation": torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
            },
            {
                "client_id": 2,
                "params": OrderedDict({"w": torch.tensor([10.0])}),
                "representation": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            },
        ]
        personalized = server._compute_personalized_models(results)
        assert personalized[0]["w"].item() < 6.0
