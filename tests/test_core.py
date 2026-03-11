"""
Tests for FedAvgM, Fed+, CNN-GRU model, BaseTrainer hooks, and edge cases.
"""

import copy
import os
from collections import OrderedDict

import torch
import torch.nn as nn

from fed_learning.strategies import get_strategy
from fed_learning.strategies.federated.fedavg import FedAvgTrainer, FedAvgAggregator
from fed_learning.strategies.federated.fedavgm import FedAvgMTrainer, FedAvgMAggregator
from fed_learning.strategies.federated.fedplus import FedPlusTrainer
from fed_learning.models import CNN_GRU_Model
from helpers import make_simple_model, make_client_results


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

        model = make_simple_model()
        global_params = copy.deepcopy(model.state_dict())

        agg1 = aggregator.aggregate(results, global_params=global_params)
        assert isinstance(agg1, OrderedDict)

        agg2 = aggregator.aggregate(results, global_params=agg1)
        assert isinstance(agg2, OrderedDict)


class TestFedPlus:
    """Test Fed+ dynamic regularization."""

    def test_fedplus_trainer_has_mu(self):
        """FedPlusTrainer should have mu parameter."""
        trainer = FedPlusTrainer(mu=0.5)
        assert trainer.mu == 0.5


class TestCNNGRUModel:
    """Test CNN-GRU model architecture."""

    def test_model_creation(self):
        """CNN_GRU_Model should create without errors."""
        model = CNN_GRU_Model(input_shape=(40,), num_classes=10)
        assert model.num_classes == 10

    def test_model_forward(self):
        """Forward pass should produce correct output shape."""
        model = CNN_GRU_Model(input_shape=(40,), num_classes=10)
        x = torch.randn(4, 40)
        output = model(x)
        assert output.shape == (4, 10)

    def test_model_with_features(self):
        """Model should handle multi-feature input."""
        model = CNN_GRU_Model(input_shape=(40, 3), num_classes=10)
        x = torch.randn(4, 40, 3)
        output = model(x)
        assert output.shape == (4, 10)


class TestBaseTrainerHooks:
    """Test that BaseTrainer hooks work correctly."""

    def test_default_hooks_are_noop(self):
        """Default hook implementations should be no-ops."""
        trainer = FedAvgTrainer()
        model = make_simple_model()

        trainer.pre_train(model)
        trainer.post_train(model)
        trainer.pre_step(model)
        trainer.post_step(model)

    def test_default_optimizer_is_adam(self):
        """Default optimizer should be Adam."""
        trainer = FedAvgTrainer()
        assert trainer.get_optimizer_class() == torch.optim.Adam


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


class TestConfigIntegration:
    """Test that CONFIG from train_incremental_kaggle.py works correctly."""

    def test_no_duplicate_mu_key(self):
        """CONFIG should not have duplicate 'mu' key."""
        import re

        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "train_incremental_kaggle.py",
        )
        with open(config_path, "r", encoding="utf-8") as f:
            content = f.read()

        mu_keys = re.findall(r'"mu"\s*:', content)
        assert len(mu_keys) == 0, (
            f"Found {len(mu_keys)} duplicate 'mu' keys in CONFIG."
        )

    def test_cgofed_gets_separate_mu_values(self):
        """CGoFed strategy should receive different mu values."""
        trainer, _ = get_strategy(
            algorithm="cgofed",
            mu_fedprox=1.5,
            mu_cgofed=5.0,
        )
        assert trainer.mu == 0.0
        assert trainer.mu_projection == 5.0
