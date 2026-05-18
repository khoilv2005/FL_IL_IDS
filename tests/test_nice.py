"""
Tests for NICE (Neurogenesis Inspired Contextual Encoding) strategy.
"""

from collections import OrderedDict

import torch
import numpy as np

from fed_learning.strategies import STRATEGIES, get_strategy
from fed_learning.strategies.fed_incremental.nice import NICETrainer, NICEAggregator
from fed_learning.strategies.incremental.nice import select_learner_units
from fed_learning.models.nice_model import NICEModel
from fed_learning.clients.nice_client import NICEClient
from fed_learning.servers.nice_server import NICEServer


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

    def test_nice_loss_keeps_gradient_for_single_class_batch(self):
        """NICE CE must not collapse to zero on one-class non-IID batches."""
        trainer = NICETrainer()
        output = torch.randn(8, 6, requires_grad=True)
        target = torch.full((8,), 2, dtype=torch.long)

        loss = trainer.compute_loss(None, output, target)
        loss.backward()

        assert loss.item() > 0.0
        assert output.grad is not None
        assert output.grad.abs().sum().item() > 0.0

    def test_select_learner_units_prunes_current_learners_only(self):
        """Later NICE phases should not reintroduce already-pruned young units."""

        class FakeModel:
            LAYER_NAMES = ["fc1", "fc2"]

            def __init__(self):
                self.unit_ranks = {
                    "fc1": np.array([1, 1, 0, 2], dtype=np.int32),
                    "fc2": np.array([1, 0], dtype=np.int32),
                }

            def get_activations(self, _data):
                return {
                    "fc1": torch.tensor([0.1, 0.9, 100.0, 0.0]),
                }

        model = FakeModel()
        select_learner_units(model, tau=0.5, data=torch.empty(1))

        assert model.unit_ranks["fc1"].tolist() == [0, 1, 0, 2]

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

    def test_nice_aggregator_restores_mature_bn_and_gru_channels(self):
        """Federated averaging must not drift frozen BN stats or GRU gate rows."""
        model = NICEModel(input_shape=(32,), num_classes=4)
        global_params = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
        )

        client_params = OrderedDict()
        for key, value in global_params.items():
            if value.dtype.is_floating_point:
                client_params[key] = value + 10.0
            else:
                client_params[key] = value.clone()

        conv1_freeze = np.zeros(64, dtype=bool)
        conv1_freeze[0] = True
        gru_freeze = np.zeros(100, dtype=bool)
        gru_freeze[0] = True

        aggregator = NICEAggregator()
        aggregator.set_freeze_masks({"conv1": conv1_freeze, "gru": gru_freeze})
        averaged = aggregator.aggregate(
            [{"params": client_params, "num_samples": 1}],
            global_params=global_params,
        )

        assert torch.allclose(averaged["conv1.weight"][0], global_params["conv1.weight"][0])
        assert torch.allclose(averaged["bn1.weight"][0], global_params["bn1.weight"][0])
        assert torch.allclose(
            averaged["bn1.running_mean"][0], global_params["bn1.running_mean"][0]
        )
        assert torch.allclose(
            averaged["bn1.running_var"][0], global_params["bn1.running_var"][0]
        )
        assert torch.allclose(
            averaged["gru.weight_ih_l0"][0], global_params["gru.weight_ih_l0"][0]
        )
        assert torch.allclose(
            averaged["gru.weight_ih_l0"][100], global_params["gru.weight_ih_l0"][100]
        )
        assert torch.allclose(
            averaged["gru.weight_ih_l0"][200], global_params["gru.weight_ih_l0"][200]
        )
        assert not torch.allclose(
            averaged["bn1.weight"][1], global_params["bn1.weight"][1]
        )

    def test_nice_reset_frozen_gradients_freezes_bn_and_gru_rows(self):
        """Local optimizer should not update mature BN channels or GRU gate rows."""
        model = NICEModel(input_shape=(32,), num_classes=4)
        conv1_freeze = np.zeros(64, dtype=bool)
        conv1_freeze[0] = True
        gru_freeze = np.zeros(100, dtype=bool)
        gru_freeze[0] = True
        model.freeze_masks = {"conv1": conv1_freeze, "gru": gru_freeze}

        for param in model.parameters():
            param.grad = torch.ones_like(param)

        model.reset_frozen_gradients()

        assert model.conv1.weight.grad[0].abs().sum().item() == 0.0
        assert model.bn1.weight.grad[0].item() == 0.0
        assert model.bn1.bias.grad[0].item() == 0.0
        assert model.gru.weight_ih_l0.grad[0].abs().sum().item() == 0.0
        assert model.gru.weight_ih_l0.grad[100].abs().sum().item() == 0.0
        assert model.gru.weight_ih_l0.grad[200].abs().sum().item() == 0.0
        assert model.bn1.weight.grad[1].item() == 1.0

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

    def test_nice_context_mask_boosts_predicted_episode_classes(self, monkeypatch):
        """Evaluation should follow official NICE context-logit boost."""
        clients = [NICEClient(0, torch.randn(4, 32), torch.randint(0, 2, (4,)))]
        test_data = {
            "X_test": torch.randn(4, 32),
            "y_test": torch.tensor([0, 2, 4, 5]),
        }
        server = NICEServer(
            clients,
            test_data,
            {
                "algorithm": "nice",
                "input_shape": (32,),
                "num_classes": 6,
                "num_gpus": 0,
            },
        )
        server.seen_classes = [0, 1, 2, 3, 4, 5]
        server.context_detector.episode_classes = {
            0: [0, 1],
            1: [2, 3],
            2: [4, 5],
        }

        monkeypatch.setattr(
            server.context_detector,
            "_binarize_per_sample",
            lambda _model, data: np.zeros((len(data), 3), dtype=np.float32),
        )
        monkeypatch.setattr(
            server.context_detector,
            "predict_episodes_batch",
            lambda _binary: np.array([0, 1]),
        )

        logits = torch.tensor(
            [
                [0.0, 1.0, 9.0, 8.0, 7.0, 6.0],
                [0.0, 1.0, 2.0, 3.0, 9.0, 8.0],
            ],
            dtype=torch.float32,
        )
        masked = server._apply_context_mask(logits, torch.randn(2, 32))

        assert masked[0, :2].min() > 99998.0
        assert masked[0, 2:].max() < 99998.0
        assert masked[1, 2:4].min() > 99998.0
        assert masked[1, :2].max() < 99998.0
        assert masked[1, 4:].max() < 99998.0
        assert masked.argmax(dim=1).tolist() == [1, 3]

    def test_nice_global_eval_context_boost_is_opt_in(self, monkeypatch):
        """Default NICE evaluation should only mask unseen classes."""
        clients = [NICEClient(0, torch.randn(4, 32), torch.randint(0, 2, (4,)))]
        test_data = {
            "X_test": torch.randn(2, 32),
            "y_test": torch.tensor([0, 2]),
        }
        server = NICEServer(
            clients,
            test_data,
            {
                "algorithm": "nice",
                "input_shape": (32,),
                "num_classes": 4,
                "num_gpus": 0,
            },
        )
        server.seen_classes = [0, 1, 2, 3]

        logits = torch.tensor(
            [
                [9.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 9.0, 0.0],
            ],
            dtype=torch.float32,
        )

        monkeypatch.setattr(
            server.global_model,
            "get_output_and_context_activations",
            lambda data: (logits[: len(data)].to(server.primary_device), None),
        )
        monkeypatch.setattr(
            server,
            "_apply_context_mask",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("context eval should be opt-in")
            ),
        )

        metrics = server.evaluate_global(batch_size=2)

        assert metrics["accuracy"] == 1.0
