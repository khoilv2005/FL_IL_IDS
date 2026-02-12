"""
Tests for Federated Learning algorithm implementations.

Covers:
- Strategy factory (get_strategy)
- FedAvg aggregation
- FedProx proximal term
- FedAvgM momentum aggregation
- Fed+ dynamic regularization
- CGoFed gradient projection (mu_projection separation)
- EWC Fisher information & penalty
- FedLwF knowledge distillation
- FedCBDR temperature scaling
"""

import sys
import os
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fed_learning.strategies import get_strategy, STRATEGIES, list_strategies
from fed_learning.strategies.federated.fedavg import FedAvgTrainer, FedAvgAggregator
from fed_learning.strategies.federated.fedprox import FedProxTrainer, FedProxAggregator
from fed_learning.strategies.federated.fedavgm import FedAvgMTrainer, FedAvgMAggregator
from fed_learning.strategies.federated.fedplus import FedPlusTrainer, FedPlusAggregator
from fed_learning.strategies.incremental.cgofed import CGoFedTrainer, CGoFedAggregator
from fed_learning.strategies.incremental.ewc import (
    EWCMixin,
    FedAvgEWCTrainer,
    FedProxEWCTrainer,
)
from fed_learning.strategies.incremental.fedlwf import FedLwFTrainer, FedLwFAggregator
from fed_learning.strategies.incremental.fedcbdr import (
    FedCBDRTrainer,
    FedCBDRAggregator,
)
from fed_learning.core.trainer import BaseTrainer
from fed_learning.core.aggregator import BaseAggregator
from fed_learning.models import CNN_GRU_Model
from fed_learning.clients.cgofed_client import CGoFedClient
from fed_learning.servers.cgofed_server import CGoFedServer


# =============================================================================
# TEST HELPERS
# =============================================================================


def make_simple_model(input_dim=10, num_classes=5):
    """Create a simple linear model for testing."""
    model = nn.Sequential(
        nn.Linear(input_dim, 20),
        nn.ReLU(),
        nn.Linear(20, num_classes),
    )
    return model


def make_cnn_gru_model(seq_length=40, num_classes=10):
    """Create a CNN-GRU model for testing."""
    return CNN_GRU_Model(input_shape=(seq_length,), num_classes=num_classes)


def make_client_results(num_clients=3, input_dim=10, num_classes=5):
    """Create fake client results for aggregation tests."""
    results = []
    for i in range(num_clients):
        model = make_simple_model(input_dim, num_classes)
        # Add some variation to parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1 * (i + 1))
        results.append(
            {
                "params": copy.deepcopy(model.state_dict()),
                "num_samples": 100 * (i + 1),  # Different sample sizes
                "loss": 0.5 - 0.1 * i,
            }
        )
    return results


def make_dataloader(num_samples=32, input_dim=10, num_classes=5, batch_size=8):
    """Create a simple DataLoader for testing."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


# =============================================================================
# TEST: Strategy Factory
# =============================================================================


class TestStrategyFactory:
    """Test get_strategy factory function."""

    def test_all_strategies_registered(self):
        """All expected strategies should be in STRATEGIES dict."""
        expected = [
            "fedavg",
            "fedavgm",
            "fedprox",
            "fedplus",
            "cgofed",
            "fedcbdr",
            "fedavg_ewc",
            "fedprox_ewc",
            "fedavg_lwf",
            "fedprox_lwf",
        ]
        for name in expected:
            assert name in STRATEGIES, f"Strategy '{name}' not registered"

    def test_get_strategy_returns_trainer_aggregator(self):
        """get_strategy should return (trainer, aggregator) tuple."""
        for name in STRATEGIES:
            trainer, aggregator = get_strategy(name)
            assert isinstance(trainer, BaseTrainer), f"{name} trainer not BaseTrainer"
            assert isinstance(aggregator, BaseAggregator), (
                f"{name} aggregator not BaseAggregator"
            )

    def test_unknown_algorithm_raises(self):
        """Unknown algorithm name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            get_strategy("nonexistent_algo")

    def test_case_insensitive(self):
        """Algorithm names should be case-insensitive."""
        trainer1, _ = get_strategy("FedAvg")
        trainer2, _ = get_strategy("fedavg")
        assert type(trainer1) == type(trainer2)

    def test_mu_fedprox_config_key(self):
        """get_strategy should use 'mu_fedprox' config key for proximal term."""
        trainer, _ = get_strategy("fedprox", mu_fedprox=0.5)
        assert trainer.mu == 0.5

    def test_mu_fedprox_fallback_to_mu(self):
        """get_strategy should fall back to 'mu' if 'mu_fedprox' not provided."""
        trainer, _ = get_strategy("fedprox", mu=0.3)
        assert trainer.mu == 0.3

    def test_cgofed_separate_mu(self):
        """CGoFed should have mu=0 (no proximal) and separate mu_projection for gradient projection."""
        trainer, _ = get_strategy(
            "cgofed",
            mu_fedprox=1.5,  # Ignored for CGoFed - paper has no proximal term
            mu_cgofed=5.0,
        )
        assert trainer.mu == 0.0, "CGoFed paper has no proximal term (mu=0)"
        assert trainer.mu_projection == 5.0, "Projection mu should be 5.0"

    def test_cgofed_mu_projection_defaults_to_mu(self):
        """If mu_projection not given, CGoFedTrainer defaults to mu."""
        trainer = CGoFedTrainer(mu=0.5)
        assert trainer.mu_projection == 0.5

    def test_cgofed_separate_cross_task_hyperparams(self):
        """Eq.11 cross_task_weight and Eq.14 lambda_cross_task must be independently configurable."""
        trainer, aggregator = get_strategy(
            "cgofed",
            cross_task_weight=0.03,   # Eq.11 global blend
            lambda_cross_task=0.17,   # Eq.14 local regularization
        )
        assert abs(aggregator.cross_task_weight - 0.03) < 1e-8
        assert abs(trainer.lambda_cross_task - 0.17) < 1e-8

    def test_list_strategies(self):
        """list_strategies should return all 10 registered strategies."""
        strategies = list_strategies()
        assert isinstance(strategies, dict)
        assert len(strategies) == 10, (
            f"Expected 10 strategies, got {len(strategies)}: {list(strategies.keys())}"
        )
        # Verify all registered strategies are listed
        for name in STRATEGIES:
            assert name in strategies, (
                f"Strategy '{name}' missing from list_strategies()"
            )


# =============================================================================
# TEST: FedAvg
# =============================================================================


class TestFedAvg:
    """Test FedAvg trainer and aggregator."""

    def test_fedavg_trainer_is_base(self):
        """FedAvgTrainer should use simple cross-entropy loss."""
        trainer = FedAvgTrainer()
        model = make_simple_model()
        x = torch.randn(4, 10)
        output = model(x)
        target = torch.randint(0, 5, (4,))
        loss = trainer.compute_loss(model, output, target)
        assert loss.ndim == 0  # scalar
        assert loss.item() > 0

    def test_fedavg_aggregation(self):
        """FedAvg aggregation should be weighted average by sample count."""
        aggregator = FedAvgAggregator()
        results = make_client_results(3)
        agg_params = aggregator.aggregate(results)

        assert isinstance(agg_params, OrderedDict)
        # Check same keys as original params
        assert set(agg_params.keys()) == set(results[0]["params"].keys())

    def test_fedavg_weighted_average_correctness(self):
        """Verify weighted average is computed correctly."""
        aggregator = FedAvgAggregator()

        # Two clients with known params and sample counts
        p1 = OrderedDict({"w": torch.tensor([1.0, 2.0])})
        p2 = OrderedDict({"w": torch.tensor([3.0, 4.0])})
        results = [
            {"params": p1, "num_samples": 100},
            {"params": p2, "num_samples": 300},
        ]

        agg = aggregator.aggregate(results)
        # Expected: (100/400)*[1,2] + (300/400)*[3,4] = [0.25+2.25, 0.5+3.0] = [2.5, 3.5]
        expected = torch.tensor([2.5, 3.5])
        assert torch.allclose(agg["w"], expected, atol=1e-5)


# =============================================================================
# TEST: FedProx
# =============================================================================


class TestFedProx:
    """Test FedProx proximal regularization."""

    def test_proximal_term_increases_loss(self):
        """FedProx loss should be >= CE loss due to proximal term."""
        trainer = FedProxTrainer(mu=1.0)
        model = make_simple_model()
        global_params = copy.deepcopy(model.state_dict())

        # Perturb model parameters
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        x = torch.randn(4, 10)
        output = model(x)
        target = torch.randint(0, 5, (4,))

        loss_prox = trainer.compute_loss(
            model, output, target, global_params=global_params
        )
        loss_ce = nn.CrossEntropyLoss()(output, target)

        assert loss_prox.item() >= loss_ce.item(), "Proximal term should increase loss"

    def test_proximal_term_zero_when_params_equal(self):
        """Proximal term should be ~0 when model params == global params."""
        trainer = FedProxTrainer(mu=1.0)
        model = make_simple_model()
        global_params = copy.deepcopy(model.state_dict())

        x = torch.randn(4, 10)
        output = model(x)
        target = torch.randint(0, 5, (4,))

        loss_prox = trainer.compute_loss(
            model, output, target, global_params=global_params
        )
        loss_ce = nn.CrossEntropyLoss()(output, target)

        assert abs(loss_prox.item() - loss_ce.item()) < 1e-5

    def test_proximal_without_global_params(self):
        """Without global_params, FedProx should degrade to CE loss."""
        trainer = FedProxTrainer(mu=1.0)
        model = make_simple_model()
        x = torch.randn(4, 10)
        output = model(x)
        target = torch.randint(0, 5, (4,))

        loss = trainer.compute_loss(model, output, target, global_params=None)
        loss_ce = nn.CrossEntropyLoss()(output, target)

        assert abs(loss.item() - loss_ce.item()) < 1e-5

    def test_mu_affects_proximal_strength(self):
        """Higher mu should increase proximal penalty."""
        model = make_simple_model()
        global_params = copy.deepcopy(model.state_dict())

        # Perturb model
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        x = torch.randn(4, 10)
        output = model(x)
        target = torch.randint(0, 5, (4,))

        trainer_low = FedProxTrainer(mu=0.01)
        trainer_high = FedProxTrainer(mu=10.0)

        loss_low = trainer_low.compute_loss(
            model, output, target, global_params=global_params
        )
        loss_high = trainer_high.compute_loss(
            model, output, target, global_params=global_params
        )

        assert loss_high.item() > loss_low.item()


# =============================================================================
# TEST: FedAvgM
# =============================================================================


class TestFedAvgM:
    """Test FedAvg with server momentum."""

    def test_fedavgm_aggregator_creation(self):
        """FedAvgM aggregator should accept momentum and server_lr."""
        aggregator = FedAvgMAggregator(momentum=0.9, server_lr=1.0)
        assert aggregator.momentum == 0.9
        assert aggregator.server_lr == 1.0

    def test_fedavgm_aggregation_runs(self):
        """FedAvgM aggregation should run without errors."""
        aggregator = FedAvgMAggregator(momentum=0.9, server_lr=1.0)
        results = make_client_results(3)

        # FedAvgM requires global_params (for momentum computation)
        model = make_simple_model()
        global_params = copy.deepcopy(model.state_dict())

        # First round with global_params
        agg1 = aggregator.aggregate(results, global_params=global_params)
        assert isinstance(agg1, OrderedDict)

        # Second round
        agg2 = aggregator.aggregate(results, global_params=agg1)
        assert isinstance(agg2, OrderedDict)


# =============================================================================
# TEST: Fed+
# =============================================================================


class TestFedPlus:
    """Test Fed+ dynamic regularization."""

    def test_fedplus_trainer_has_mu(self):
        """FedPlusTrainer should have mu parameter."""
        trainer = FedPlusTrainer(mu=0.5)
        assert trainer.mu == 0.5


# =============================================================================
# TEST: CGoFed
# =============================================================================


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
        trainer = CGoFedTrainer(
            mu=0.0, lambda_cross_task=0.1
        )  # No proximal, has cross-task reg
        trainer.set_task(1, [0, 1, 2, 3, 4])  # Task 1 so historical models apply
        model = make_simple_model()

        # Create historical model (different from current)
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

        # Paper Eq. 14: Loss = CE + A(Θ), should be greater than CE alone
        assert loss.item() > loss_ce.item(), (
            f"CGoFed Eq.14 should have A(Θ) regularization, got loss={loss.item()}, CE={loss_ce.item()}"
        )

    def test_cgofed_no_proximal_term(self):
        """Paper Eq. 14: CGoFed does NOT have proximal term (unlike FedProx)."""
        trainer = CGoFedTrainer(mu=0.0)  # Paper: no proximal term
        model = make_simple_model()

        x = torch.randn(4, 10)
        output = model(x)
        target = torch.randint(0, 5, (4,))

        # Without historical models, should be pure CE (no proximal, no reg)
        loss = trainer.compute_loss(model, output, target)
        loss_ce = nn.CrossEntropyLoss()(output, target)

        assert abs(loss.item() - loss_ce.item()) < 1e-6, (
            f"CGoFed should NOT have proximal term when no historical models, got loss={loss.item()}, expected CE={loss_ce.item()}"
        )

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
        assert trainer.mu_coefficient == 1.0  # No decay for task 0

        trainer.set_task(1, [3, 4])
        # mu_coefficient = 0.5^(1-0) = 0.5
        assert abs(trainer.mu_coefficient - 0.5) < 1e-6

    def test_cgofed_forgetting_reset(self):
        """When AF > theta, mu_coefficient should reset to 1.0."""
        trainer = CGoFedTrainer(theta_threshold=0.1, lambda_decay=0.5)
        trainer.set_task(0, [0, 1])

        # Set best accuracy for task 0
        trainer.best_acc_per_task[0] = 0.9

        # Move to task 1 and report low accuracy for task 0 (high forgetting)
        trainer.set_task(1, [2, 3])
        trainer.update_forgetting({0: 0.5, 1: 0.8})  # AF = 0.4, > theta=0.1

        assert trainer.mu_coefficient == 1.0, (
            "mu_coefficient should reset when AF > theta"
        )
        assert trainer.t_reset == 1

    def test_cgofed_pre_step_skip_first_task(self):
        """pre_step should do nothing on the first task (no old space)."""
        trainer = CGoFedTrainer()
        model = make_simple_model()
        trainer.set_task(0, [0, 1, 2])

        # Should not raise
        trainer.pre_step(model)

    def test_cgofed_aggregator(self):
        """CGoFedAggregator should handle basic aggregation."""
        aggregator = CGoFedAggregator(cross_task_weight=0.1, top_k=2)
        results = make_client_results(3)
        agg = aggregator.aggregate(results)
        assert isinstance(agg, OrderedDict)


# =============================================================================
# TEST: CGoFed Server
# =============================================================================


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
        """Eq.14 should prepare different historical selections per current client when similarities differ."""
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
        """Eq.14 setup should rely on client train data, not test_data tensors."""
        clients = [CGoFedClient(0, torch.randn(8, 32), torch.randint(0, 2, (8,)))]
        server = self._make_server(clients, test_data={"y_test": torch.tensor([0, 1])})
        server.aggregator = CGoFedAggregator(top_k=1, rounds_per_task=1)

        # Should not raise KeyError for missing X_test in set_task.
        server.set_task(1, [2, 3], [0, 1, 2, 3])
        assert isinstance(server._client_reg_info, dict)

    def test_eq12_personalization_uses_matrix_similarity(self):
        """Eq.12 personalization should be driven by matrix distance, not mean-vector collapse."""
        clients = [CGoFedClient(0, torch.randn(4, 32), torch.randint(0, 2, (4,)))]
        server = self._make_server(clients)
        server.eq12_self_weight = 0.0  # isolate weighted sum of other-client models

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
        # If means were used, both distances become 0 and this tends to equal weighting (value ~6).
        assert personalized[0]["w"].item() < 6.0


# =============================================================================
# TEST: EWC
# =============================================================================


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
        loss_ce = nn.CrossEntropyLoss()(output, target)

        # No Fisher stored yet, so EWC penalty should be 0
        assert abs(loss.item() - loss_ce.item()) < 1e-5

    def test_ewc_online_ewc_gamma(self):
        """Online EWC should use gamma parameter."""
        trainer = FedAvgEWCTrainer(online_ewc=True, gamma=0.8)
        assert trainer.online_ewc is True
        assert trainer.gamma == 0.8


# =============================================================================
# TEST: FedLwF
# =============================================================================


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
        """On first task, loss should just be CE (no old model to distill from)."""
        trainer = FedLwFTrainer()
        trainer.set_task(0, [0, 1, 2])

        model = make_simple_model()
        x = torch.randn(4, 10)
        output = model(x)
        target = torch.randint(0, 5, (4,))

        loss = trainer.compute_loss(model, output, target)
        loss_ce = nn.CrossEntropyLoss()(output, target)

        assert abs(loss.item() - loss_ce.item()) < 1e-5

    def test_fedlwf_aggregator(self):
        """FedLwFAggregator should perform weighted average."""
        aggregator = FedLwFAggregator()
        results = make_client_results(3)
        agg = aggregator.aggregate(results)
        assert isinstance(agg, OrderedDict)


# =============================================================================
# TEST: FedCBDR
# =============================================================================


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


# =============================================================================
# TEST: Config Integration
# =============================================================================


class TestConfigIntegration:
    """Test that the CONFIG from train_incremental_kaggle.py works correctly."""

    def test_no_duplicate_mu_key(self):
        """CONFIG should not have duplicate 'mu' key (use mu_fedprox and mu_cgofed)."""
        # Read and parse the CONFIG dict from the main script
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "train_incremental_kaggle.py",
        )
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Count occurrences of '"mu"' as a dict key (not "mu_fedprox" or "mu_cgofed")
        import re

        # Match "mu" followed by : but NOT "mu_fedprox" or "mu_cgofed" or "mu_coefficient"
        mu_keys = re.findall(r'"mu"\s*:', content)
        assert len(mu_keys) == 0, (
            f"Found {len(mu_keys)} duplicate 'mu' keys in CONFIG. Use 'mu_fedprox' and 'mu_cgofed' instead."
        )

    def test_cgofed_gets_separate_mu_values(self):
        """CGoFed strategy should receive different mu values for proximal and projection."""
        trainer, _ = get_strategy(
            algorithm="cgofed",
            mu_fedprox=1.5,
            mu_cgofed=5.0,
        )
        # Paper: CGoFed has no proximal term (mu=0), only projection mu
        assert trainer.mu == 0.0, "CGoFed paper has no proximal term"
        assert trainer.mu_projection == 5.0, "Projection mu should be 5.0"


# =============================================================================
# TEST: CNN-GRU Model
# =============================================================================


class TestCNNGRUModel:
    """Test CNN-GRU model architecture."""

    def test_model_creation(self):
        """CNN_GRU_Model should create without errors."""
        model = CNN_GRU_Model(input_shape=(40,), num_classes=10)
        assert model.num_classes == 10

    def test_model_forward(self):
        """Forward pass should produce correct output shape."""
        model = CNN_GRU_Model(input_shape=(40,), num_classes=10)
        x = torch.randn(4, 40)  # batch of 4, seq_len=40
        output = model(x)
        assert output.shape == (4, 10)

    def test_model_with_features(self):
        """Model should handle multi-feature input."""
        model = CNN_GRU_Model(input_shape=(40, 3), num_classes=10)
        x = torch.randn(4, 40, 3)
        output = model(x)
        assert output.shape == (4, 10)


# =============================================================================
# TEST: BaseTrainer hooks
# =============================================================================


class TestBaseTrainerHooks:
    """Test that BaseTrainer hooks work correctly."""

    def test_default_hooks_are_noop(self):
        """Default hook implementations should be no-ops."""
        trainer = FedAvgTrainer()
        model = make_simple_model()

        # These should not raise
        trainer.pre_train(model)
        trainer.post_train(model)
        trainer.pre_step(model)
        trainer.post_step(model)

    def test_default_optimizer_is_adam(self):
        """Default optimizer should be Adam."""
        trainer = FedAvgTrainer()
        assert trainer.get_optimizer_class() == torch.optim.Adam


# =============================================================================
# TEST: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_aggregation_single_client(self):
        """Aggregation with single client should return that client's params."""
        aggregator = FedAvgAggregator()
        model = make_simple_model()
        results = [
            {
                "params": copy.deepcopy(model.state_dict()),
                "num_samples": 100,
            }
        ]
        agg = aggregator.aggregate(results)

        for key in model.state_dict():
            assert torch.allclose(agg[key], model.state_dict()[key].float(), atol=1e-5)

    def test_aggregation_equal_weights(self):
        """Equal sample counts should give simple average."""
        aggregator = FedAvgAggregator()
        p1 = OrderedDict({"w": torch.tensor([0.0])})
        p2 = OrderedDict({"w": torch.tensor([2.0])})
        results = [
            {"params": p1, "num_samples": 100},
            {"params": p2, "num_samples": 100},
        ]
        agg = aggregator.aggregate(results)
        assert torch.allclose(agg["w"], torch.tensor([1.0]), atol=1e-5)
