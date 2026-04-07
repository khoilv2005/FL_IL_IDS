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

    def test_set_task_resets_eq14_state_for_first_round(self):
        """Eq.14 state should be empty at task start and filled after a round."""
        clients = [
            CGoFedClient(0, torch.randn(8, 32), torch.randint(0, 2, (8,))),
            CGoFedClient(1, torch.randn(8, 32), torch.randint(0, 2, (8,))),
        ]
        server = self._make_server(clients)
        agg = CGoFedAggregator(top_k=1, rounds_per_task=1)
        server.aggregator = agg
        server.set_task(1, [2, 3], [0, 1, 2, 3])
        assert server._client_reg_info == {}
        assert server._personalized_round_models == {}

    def test_set_task_does_not_depend_on_test_data_for_eq14(self):
        """Task setup should not depend on test_data for Eq.14 state."""
        clients = [CGoFedClient(0, torch.randn(8, 32), torch.randint(0, 2, (8,)))]
        server = self._make_server(clients, test_data={"y_test": torch.tensor([0, 1])})
        server.aggregator = CGoFedAggregator(top_k=1, rounds_per_task=1)

        server.set_task(1, [2, 3], [0, 1, 2, 3])
        assert server._client_reg_info == {}

    def test_prepare_next_round_reg_info_uses_selected_peer_client_history(self):
        """Eq.14 should use current-round peer selection, then all history of that peer."""
        clients = [
            CGoFedClient(0, torch.randn(8, 32), torch.randint(0, 2, (8,))),
            CGoFedClient(1, torch.randn(8, 32), torch.randint(0, 2, (8,))),
            CGoFedClient(2, torch.randn(8, 32), torch.randint(0, 2, (8,))),
        ]
        server = self._make_server(clients)
        server.aggregator = CGoFedAggregator(top_k=1, rounds_per_task=2)
        server.set_task(2, [4, 5], [0, 1, 2, 3, 4, 5])

        base = server.get_global_params()
        model_a0 = OrderedDict(
            (k, (v + 0.1) if v.dtype.is_floating_point else v.clone())
            for k, v in base.items()
        )
        model_a1 = OrderedDict(
            (k, (v + 0.2) if v.dtype.is_floating_point else v.clone())
            for k, v in base.items()
        )
        model_b0 = OrderedDict(
            (k, (v + 0.3) if v.dtype.is_floating_point else v.clone())
            for k, v in base.items()
        )

        server._client_task_models = {
            1: {0: model_a0, 1: model_a1},
            2: {0: model_b0},
        }
        server._client_task_representations = {
            1: {
                0: torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                1: torch.tensor([[1.0, 0.2], [0.2, 1.0]]),
            },
            2: {
                0: torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
            },
        }

        results = [
            {
                "client_id": 0,
                "params": OrderedDict({"w": torch.tensor([0.0])}),
                "representation": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            },
            {
                "client_id": 1,
                "params": OrderedDict({"w": torch.tensor([2.0])}),
                "representation": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            },
            {
                "client_id": 2,
                "params": OrderedDict({"w": torch.tensor([10.0])}),
                "representation": torch.tensor([[1.0, 1.0], [0.0, 0.0]]),
            },
        ]

        reg_info = server._prepare_next_round_reg_info(results)

        assert 0 in reg_info
        keys_client0 = sorted(reg_info[0]["historical_models"].keys())
        assert keys_client0 == ["c1_t0", "c1_t1"]

    def test_similarity_is_order_invariant(self):
        """Similarity should not change when sample rows are permuted."""
        clients = [CGoFedClient(0, torch.randn(4, 32), torch.randint(0, 2, (4,)))]
        server = self._make_server(clients)

        rep = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        permuted = rep[torch.tensor([2, 0, 1])]

        sim = server._compute_similarity(rep, permuted)
        assert abs(sim) < 1e-6

    def test_eq12_personalization_uses_matrix_similarity(self):
        """Eq.12 personalization should respect permutation-invariant similarity."""
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
        assert personalized[0]["w"].item() > 6.0
