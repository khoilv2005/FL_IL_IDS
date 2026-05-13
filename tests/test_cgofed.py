"""
Tests for CGoFed strategy and CGoFedServer.
"""

import copy
from collections import OrderedDict

import torch
import torch.nn as nn

from fed_learning.strategies.fed_incremental.cgofed import (
    CGoFedTrainer,
    CGoFedAggregator,
    build_representation_artifact,
    is_representation_artifact,
)
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

    def test_cgofed_task_ce_updates_full_head(self):
        """CGoFed should preserve its stable full-head CE training objective."""
        trainer = CGoFedTrainer(mu=0.0)
        trainer.set_task(0, [0, 1])
        output = torch.tensor(
            [[3.0, 0.5, 8.0, 7.0], [0.2, 2.0, 6.0, 5.0]],
            requires_grad=True,
        )
        target = torch.tensor([0, 1])

        loss = trainer.compute_loss(nn.Linear(1, 4), output, target)
        loss.backward()

        assert output.grad is not None
        assert output.grad[:, 2:].abs().sum() > 0
        assert output.grad[:, :2].abs().sum() > 0

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

    def test_client_train_uses_num_samples_rep(self, monkeypatch):
        """Client-side representation should use the trainer's n_s setting."""
        client = CGoFedClient(0, torch.randn(16, 32), torch.randint(0, 2, (16,)))
        client.setup_for_gpu(nn.Linear(32, 4), "cpu")

        observed = {}

        def fake_representation(model, num_samples=None):
            observed["num_samples"] = num_samples
            return torch.zeros(3, 2)

        monkeypatch.setattr(client, "compute_activation_representation", fake_representation)

        trainer = CGoFedTrainer(num_samples_rep=13)
        result = client.train(trainer, epochs=1, batch_size=4, lr=0.01)

        assert observed["num_samples"] == 13
        assert is_representation_artifact(result["representation"])
        assert result["representation"]["shape"] == (3, 2)

    def test_client_representation_prefers_fused_representation(self):
        """Task-level R^t should use get_fused_representation when available."""

        class ToyFusedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(3, 5)
                self.fc2 = nn.Linear(5, 2)

            def get_fused_representation(self, x):
                return x[:, :3] + 10.0

            def forward(self, x):
                z = self.get_fused_representation(x)
                return self.fc2(torch.relu(self.fc1(z)))

        X = torch.arange(24, dtype=torch.float32).view(6, 4)
        y = torch.randint(0, 2, (6,))
        client = CGoFedClient(0, X, y)
        model = ToyFusedModel()

        rep = client.compute_activation_representation(model, num_samples=6)

        assert rep.shape == (6, 3)
        expected_rows = {tuple(row.tolist()) for row in (X[:, :3] + 10.0)}
        actual_rows = {tuple(row.tolist()) for row in rep}
        assert actual_rows == expected_rows

    def test_client_builds_local_projection_space(self, tmp_path):
        """Eq.3-5 projection bases should be built and stored on the client."""
        client = CGoFedClient(0, torch.randn(12, 4), torch.randint(0, 2, (12,)))
        model = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(4, 3)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(3, 2)),
        ]))
        trainer = CGoFedTrainer(temp_dir=str(tmp_path), num_samples_rep=8, mu_projection=0.5)
        trainer.set_task(0, [0, 1])

        client.build_projection_space(model=model, trainer=trainer, num_samples=8)

        assert "task_0" in client.projection_layer_bases
        assert "fc1" in client.projection_layer_bases["task_0"]
        info = client.projection_layer_bases["task_0"]["fc1"]
        assert torch.load(info["basis"], map_location="cpu").shape[0] == 4
        assert torch.load(info["importance"], map_location="cpu").ndim == 1

    def test_projection_targets_exclude_classifier_head(self):
        """CGoFed should mirror upstream by excluding classifier heads from projection."""
        model = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv1d(1, 2, 3)),
            ("fc1", nn.Linear(4, 3)),
            ("fc2", nn.Linear(3, 2)),
        ]))
        client = CGoFedClient(0, torch.randn(4, 4), torch.randint(0, 2, (4,)))
        trainer = CGoFedTrainer()

        client_targets = [name for name, _ in client._get_projection_target_modules(model)]
        trainer_targets = [name for name, _ in trainer._get_projection_target_modules(model)]

        assert "fc2" not in client_targets
        assert "fc2" not in trainer_targets
        assert "fc1" in client_targets
        assert "fc1" in trainer_targets

    def test_trainer_pre_step_uses_client_projection_bases(self, tmp_path):
        """Shared trainer should project from kwargs client bases, not global bases."""
        client = CGoFedClient(0, torch.randn(12, 4), torch.randint(0, 2, (12,)))
        model = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(4, 3)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(3, 2)),
        ]))
        trainer = CGoFedTrainer(temp_dir=str(tmp_path), num_samples_rep=8, mu_projection=0.5)
        trainer.set_task(0, [0, 1])
        client.build_projection_space(model=model, trainer=trainer, num_samples=8)

        trainer.set_task(1, [2, 3])
        trainer.layer_bases = {}
        output = model(torch.randn(4, 4))
        loss = output.sum()
        loss.backward()

        grad_before = model.fc1.weight.grad.detach().clone()
        trainer.pre_step(
            model,
            projection_layer_bases=client.projection_layer_bases,
            projection_cache_key="client_0_test",
        )
        grad_after = model.fc1.weight.grad.detach().clone()

        assert not torch.equal(grad_before, grad_after)
        trainer.clear_projection_cache("client_0_test")

    def test_client_train_applies_local_projection_without_trainer_bases(self, tmp_path):
        """Paper Eq.6/8/9 should execute inside CGoFedClient local training."""
        X = torch.randn(16, 4)
        y = torch.randint(0, 2, (16,))
        client = CGoFedClient(0, X, y)
        model = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(4, 3)),
            ("relu", nn.ReLU()),
            ("fc2", nn.Linear(3, 2)),
        ]))
        client.setup_for_gpu(model, "cpu")

        trainer = CGoFedTrainer(
            temp_dir=str(tmp_path),
            num_samples_rep=8,
            mu_projection=0.5,
        )
        trainer.set_task(0, [0, 1])
        client.build_projection_space(model=model, trainer=trainer, num_samples=8)

        trainer.set_task(1, [2, 3])
        trainer.layer_bases = {}
        stats_before = trainer.get_projection_stats(reset=True)
        assert stats_before["projected"] == 0

        result = client.train(trainer, epochs=1, batch_size=4, lr=0.01)
        stats_after = trainer.get_projection_stats(reset=True)

        assert result["client_id"] == 0
        assert stats_after["projected"] > 0

    def test_build_representation_artifact_passes_through_existing_artifact(self):
        """Artifact reuse should avoid rebuilding the heavy signature twice."""
        artifact = {
            "_artifact_type": "cgofed_representation_artifact",
            "signature": torch.tensor([1.0, 2.0]),
            "shape": (3, 2),
            "mean_vector": torch.tensor([0.5, 0.5]),
            "mean_norm": 0.7071,
        }

        output = build_representation_artifact(artifact)

        assert output is not artifact
        assert is_representation_artifact(output)
        assert output["shape"] == (3, 2)

    def test_store_client_representations_keeps_all_samples(self):
        """Aggregator should retain full sample metadata without a 50k cap."""
        aggregator = CGoFedAggregator(cross_task_weight=0.1, top_k=2)
        aggregator.set_task(0)

        rep_a = torch.randn(30000, 2)
        rep_b = torch.randn(30000, 2)
        aggregator._store_client_representations(
            [
                {"client_id": 0, "representation": rep_a},
                {"client_id": 1, "representation": rep_b},
            ]
        )

        stored = aggregator.task_representation_matrices[0]
        assert stored["shape"] == (60000, 2)


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

    def test_initial_round_state_prepares_eq14_for_one_round_tasks(self):
        """Round 0 of task>0 should receive Eq.14/Eq.12 state before training."""
        clients = [
            CGoFedClient(0, torch.randn(8, 32), torch.randint(0, 2, (8,))),
            CGoFedClient(1, torch.randn(8, 32), torch.randint(0, 2, (8,))),
        ]
        server = self._make_server(clients)
        server.aggregator = CGoFedAggregator(top_k=1, rounds_per_task=1)
        server.set_task(1, [2, 3], [0, 1, 2, 3])

        base = server.get_global_params()
        hist_model = OrderedDict(
            (k, (v + 0.2) if v.dtype.is_floating_point else v.clone())
            for k, v in base.items()
        )
        server._client_task_models = {1: {0: hist_model}}
        server._client_task_representations = {
            1: {0: torch.tensor([[1.0, 0.0], [0.0, 1.0]])}
        }
        server._compute_pre_round_representations = lambda: {
            0: torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            1: torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        }

        server._prepare_initial_round_state(base, verbose=False)

        assert 0 in server._client_reg_info
        assert server._client_reg_info[0]["historical_models"]
        assert 0 in server._personalized_round_models

    def test_train_round_prepares_initial_state_by_default(self, monkeypatch):
        """Eq.12/Eq.14 pre-round state should be on unless explicitly disabled."""
        clients = [CGoFedClient(0, torch.randn(4, 32), torch.randint(0, 2, (4,)))]
        server = self._make_server(clients)
        server.set_task(1, [2, 3], [0, 1, 2, 3])

        called = {"value": False}

        def fake_prepare(global_params, verbose=True):
            called["value"] = True

        monkeypatch.setattr(server, "_prepare_initial_round_state", fake_prepare)
        monkeypatch.setattr(
            "fed_learning.training.cgofed_worker.train_cgofed_clients_on_gpu",
            lambda *args, **kwargs: args[4].update(
                {
                    0: {
                        "client_id": 0,
                        "num_samples": 1,
                        "loss": 0.0,
                        "params": server.get_global_params(),
                        "representation": torch.zeros(1, 2),
                    }
                }
            ),
        )

        server.train_round(verbose=False)

        assert called["value"] is True

    def test_similarity_uses_paper_l2_matrix_distance(self):
        """Similarity should be negative L2 distance between representation matrices."""
        clients = [CGoFedClient(0, torch.randn(4, 32), torch.randint(0, 2, (4,)))]
        server = self._make_server(clients)

        rep = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        permuted = rep[torch.tensor([2, 0, 1])]

        sim = server._compute_similarity(rep, permuted)
        expected = -torch.norm(rep - permuted, p="fro").item()
        assert abs(sim - expected) < 1e-6

    def test_eq12_personalization_uses_matrix_similarity(self):
        """Eq.12 personalization should use paper matrix-distance similarity."""
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
        assert 2.0 < personalized[0]["w"].item() < 6.0
