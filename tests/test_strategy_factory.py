"""
Tests for the strategy factory (get_strategy) and strategy registration.
"""

import pytest

from fed_learning.strategies import get_strategy, STRATEGIES, list_strategies
from fed_learning.strategies.incremental.cgofed import CGoFedTrainer
from fed_learning.core.trainer import BaseTrainer
from fed_learning.core.aggregator import BaseAggregator


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
        """CGoFed should have mu=0 (no proximal) and separate mu_projection."""
        trainer, _ = get_strategy(
            "cgofed",
            mu_fedprox=1.5,
            mu_cgofed=5.0,
        )
        assert trainer.mu == 0.0, "CGoFed paper has no proximal term (mu=0)"
        assert trainer.mu_projection == 5.0, "Projection mu should be 5.0"

    def test_cgofed_mu_projection_defaults_to_mu(self):
        """If mu_projection not given, CGoFedTrainer defaults to mu."""
        trainer = CGoFedTrainer(mu=0.5)
        assert trainer.mu_projection == 0.5

    def test_cgofed_separate_cross_task_hyperparams(self):
        """Eq.11 cross_task_weight and Eq.14 lambda_cross_task must be independent."""
        trainer, aggregator = get_strategy(
            "cgofed",
            cross_task_weight=0.03,
            lambda_cross_task=0.17,
        )
        assert abs(aggregator.cross_task_weight - 0.03) < 1e-8
        assert abs(trainer.lambda_cross_task - 0.17) < 1e-8

    def test_list_strategies(self):
        """list_strategies should return all registered strategies."""
        strategies = list_strategies()
        assert isinstance(strategies, dict)
        assert len(strategies) >= 10, (
            f"Expected >=10 strategies, got {len(strategies)}: {list(strategies.keys())}"
        )
        for name in STRATEGIES:
            assert name in strategies, (
                f"Strategy '{name}' missing from list_strategies()"
            )
