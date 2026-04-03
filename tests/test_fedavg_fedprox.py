"""
Tests for FedAvg and FedProx strategies.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn

from fed_learning.strategies.federated.fedavg import FedAvgTrainer, FedAvgAggregator
from fed_learning.strategies.federated.fedprox import FedProxTrainer
from helpers import make_simple_model, make_client_results


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
        assert loss.ndim == 0
        assert loss.item() > 0

    def test_fedavg_aggregation(self):
        """FedAvg aggregation should be weighted average by sample count."""
        aggregator = FedAvgAggregator()
        results = make_client_results(3)
        agg_params = aggregator.aggregate(results)
        assert isinstance(agg_params, OrderedDict)
        assert set(agg_params.keys()) == set(results[0]["params"].keys())

    def test_fedavg_weighted_average_correctness(self):
        """Verify weighted average is computed correctly."""
        aggregator = FedAvgAggregator()
        p1 = OrderedDict({"w": torch.tensor([1.0, 2.0])})
        p2 = OrderedDict({"w": torch.tensor([3.0, 4.0])})
        results = [
            {"params": p1, "num_samples": 100},
            {"params": p2, "num_samples": 300},
        ]
        agg = aggregator.aggregate(results)
        expected = torch.tensor([2.5, 3.5])
        assert torch.allclose(agg["w"], expected, atol=1e-5)


class TestFedProx:
    """Test FedProx proximal regularization."""

    def test_proximal_term_increases_loss(self):
        """FedProx loss should be >= CE loss due to proximal term."""
        trainer = FedProxTrainer(mu=1.0)
        model = make_simple_model()
        global_params = copy.deepcopy(model.state_dict())

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
