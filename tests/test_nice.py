"""
Tests for NICE (Neurogenesis Inspired Contextual Encoding) strategy.
"""

from collections import OrderedDict

import torch
import numpy as np

from fed_learning.strategies import STRATEGIES, get_strategy
from fed_learning.strategies.fed_incremental.nice import NICETrainer, NICEAggregator
from fed_learning.models.nice_model import NICEModel
from fed_learning.clients.nice_client import NICEClient


class TestNICE:
    """Test NICE algorithm implementation."""

    def test_nice_strategy_registration(self):
        """NICE should be registered in STRATEGIES."""
        assert "nice" in STRATEGIES
        assert "trainer" in STRATEGIES["nice"]
        assert "aggregator" in STRATEGIES["nice"]

    def test_nice_get_strategy(self):
        """get_strategy('nice') should return NICETrainer and NICEAggregator."""
        trainer, aggregator = get_strategy("nice")
        assert type(trainer).__name__ == "NICETrainer"
        assert type(aggregator).__name__ == "NICEAggregator"

    def test_nice_trainer_init(self):
        """NICETrainer should initialize with default params."""
        trainer = NICETrainer()
        assert trainer.tau == 0.95
        assert trainer.max_phases == 5
        assert trainer.phase_epochs == 5
        assert trainer.memo_per_class == 50
        assert trainer.current_task == 0

    def test_nice_trainer_custom_params(self):
        """NICETrainer should accept custom params."""
        trainer = NICETrainer(tau=0.9, max_phases=3, phase_epochs=10, memo_per_class=100)
        assert trainer.tau == 0.9
        assert trainer.max_phases == 3
        assert trainer.phase_epochs == 10
        assert trainer.memo_per_class == 100

    def test_nice_set_task(self):
        """NICETrainer.set_task should update task tracking."""
        trainer = NICETrainer()
        trainer.set_task(task_id=0, new_classes=[0, 1, 2])

        assert trainer.current_task == 0
        assert trainer.seen_classes == {0, 1, 2}
        assert trainer.new_classes == [0, 1, 2]

    def test_nice_aggregator_creation(self):
        """NICEAggregator should be created without errors."""
        aggregator = NICEAggregator()
        assert aggregator is not None

    def test_nice_aggregator_fedavg(self):
        """NICEAggregator should use FedAvg (same as base)."""
        aggregator = NICEAggregator()

        p1 = OrderedDict({"w": torch.tensor([0.0])})
        p2 = OrderedDict({"w": torch.tensor([2.0])})
        results = [
            {"params": p1, "num_samples": 100},
            {"params": p2, "num_samples": 100},
        ]
        agg = aggregator.aggregate(results)
        assert torch.allclose(agg["w"], torch.tensor([1.0]), atol=1e-5)

    def test_nice_model_creation(self):
        """NICEModel should be created without errors."""
        model = NICEModel(input_shape=(100, 1), num_classes=34)
        assert model is not None
        assert model.num_classes == 34

    def test_nice_model_forward(self):
        """NICEModel forward pass should work."""
        model = NICEModel(input_shape=(100, 1), num_classes=34)
        x = torch.randn(4, 100)
        output = model(x)
        assert output.shape == (4, 34)

    def test_nice_model_unit_ranks(self):
        """NICEModel should track unit ranks (neuron ages)."""
        model = NICEModel(input_shape=(100, 1), num_classes=34)

        # NICEModel uses unit_ranks dict, not _neuron_ages
        assert "conv1" in model.unit_ranks
        assert "conv2" in model.unit_ranks
        assert "fc2" in model.unit_ranks
        assert np.all(model.unit_ranks["conv1"] == 0)

    def test_nice_client_no_buffer(self):
        """NICE client should not require buffer (replay-free)."""
        X = torch.randn(100, 50)
        y = torch.randint(0, 10, (100,))
        client = NICEClient(0, X, y)

        stats = client.get_buffer_stats()
        assert stats["buffer_type"] == "none"
        assert stats["has_replay"] == False
