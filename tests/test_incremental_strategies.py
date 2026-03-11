"""
Tests for incremental learning strategies: EWC, FedLwF, FedCBDR.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn

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
        loss_ce = nn.CrossEntropyLoss()(output, target)
        assert abs(loss.item() - loss_ce.item()) < 1e-5

    def test_ewc_online_ewc_gamma(self):
        """Online EWC should use gamma parameter."""
        trainer = FedAvgEWCTrainer(online_ewc=True, gamma=0.8)
        assert trainer.online_ewc is True
        assert trainer.gamma == 0.8


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
        loss_ce = nn.CrossEntropyLoss()(output, target)
        assert abs(loss.item() - loss_ce.item()) < 1e-5

    def test_fedlwf_aggregator(self):
        """FedLwFAggregator should perform weighted average."""
        aggregator = FedLwFAggregator()
        results = make_client_results(3)
        agg = aggregator.aggregate(results)
        assert isinstance(agg, OrderedDict)


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
