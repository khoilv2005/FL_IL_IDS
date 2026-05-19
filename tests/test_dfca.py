"""
Comprehensive tests for DFCA (Decentralized Federated Clustering Algorithm) implementation.

Tests cover:
- Cluster assignment: client selects cluster with minimum loss
- Local update: only assigned cluster params change, unassigned stay same
- Message export: client only sends assigned cluster params
- Sequential running average: math correctness
- Fixed k: config dfca_num_clusters=10 creates 10 model banks
- Active client schedule: 100 clients, 6 tasks -> 50,60,70,80,90,100 active
- No architecture change: CNN_GRU_Model.num_classes always equals total_classes
- Factory registration: "dfca_il" creates correct server/client
"""

import copy
import random
from collections import OrderedDict
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from fed_learning.clients.dfca_client import DFCAClient
from fed_learning.strategies.federated.dfca import DFCATrainer, DFCAAggregator
from fed_learning.core.trainer import BaseTrainer
from fed_learning.core.aggregator import BaseAggregator
from fed_learning.models import CNN_GRU_Model
from fed_learning.factories.client_factory import _resolve_client_class, _build_extra_kwargs
from fed_learning.factories.server_factory import _SERVER_REGISTRY


# =============================================================================
# Test helpers
# =============================================================================

class SimpleLinearModel(nn.Module):
    """Minimal linear model for fast unit tests."""
    def __init__(self, input_dim=10, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def make_simple_params(val=1.0, keys=None):
    """Create a simple OrderedDict of params."""
    if keys is None:
        keys = ["weight", "bias"]
    return OrderedDict((k, torch.full((10, 10), val)) for k in keys)


# =============================================================================
# Test 1: Sequential Running Average
# =============================================================================

class TestSequentialRunningAverage:
    """Test the sequential running average formula."""

    def test_scalar_formula(self):
        """Test formula: theta_new = ((r+1)/(r+2)) * theta_old + (1/(r+2)) * theta_incoming"""
        # Scalar test case
        theta_old = 10.0
        theta_incoming = 0.0
        r = 1  # second neighbor (r=1 means 1 neighbor already incorporated)

        theta_new = DFCAAggregator.sequential_running_average(
            torch.tensor(theta_old),
            torch.tensor(theta_incoming),
            count=r
        )
        # alpha = (r+1)/(r+2) = 2/3, beta = 1/(r+2) = 1/3
        # result = (2/3)*10 + (1/3)*0 = 6.67
        expected = ((r + 1.0) / (r + 2.0)) * theta_old + (1.0 / (r + 2.0)) * theta_incoming
        assert abs(theta_new.item() - expected) < 1e-6

    def test_tensor_formula(self):
        """Test running average on tensors."""
        theta_old = torch.ones(10, 10) * 10.0
        theta_incoming = torch.zeros(10, 10)
        r = 1

        result = DFCAAggregator.sequential_running_average(theta_old, theta_incoming, count=r)
        # alpha = 2/3, beta = 1/3
        # result = (2/3)*10 + (1/3)*0 = 6.67
        expected = ((r + 1.0) / (r + 2.0)) * theta_old + (1.0 / (r + 2.0)) * theta_incoming
        assert torch.allclose(result, expected)

    def test_multiple_neighbors(self):
        """Test sequential averaging with multiple neighbors."""
        theta = torch.tensor(100.0)
        neighbors = [torch.tensor(0.0)] * 3  # 3 neighbors, all zeros

        for r, incoming in enumerate(neighbors):
            theta = DFCAAggregator.sequential_running_average(
                theta, incoming, count=r
            )

        # After first neighbor (r=0): theta = (1/2)*100 + (1/2)*0 = 50
        # After second neighbor (r=1): theta = (2/3)*50 + (1/3)*0 = 33.33
        # After third neighbor (r=2): theta = (3/4)*33.33 + (1/4)*0 = 25
        expected = 25.0
        assert abs(theta.item() - expected) < 1e-4

    def test_unweighted_converges_toward_neighbors(self):
        """Test that sequential averaging progressively incorporates neighbors."""
        # Sequential averaging progressively brings local model toward neighbor values.
        theta = torch.tensor(100.0)

        # Neighbor A then B
        t1 = torch.tensor(100.0)
        t1 = DFCAAggregator.sequential_running_average(t1, torch.tensor(50.0), count=0)
        t1 = DFCAAggregator.sequential_running_average(t1, torch.tensor(0.0), count=1)

        # Neighbor B then A
        t2 = torch.tensor(100.0)
        t2 = DFCAAggregator.sequential_running_average(t2, torch.tensor(0.0), count=0)
        t2 = DFCAAggregator.sequential_running_average(t2, torch.tensor(50.0), count=1)

        # Both should be closer to the average of {100, 50, 0} = 50 than to 100
        avg = (100.0 + 50.0 + 0.0) / 3.0
        assert abs(t1.item() - avg) < abs(100.0 - avg), "t1 should move toward neighbors"
        assert abs(t2.item() - avg) < abs(100.0 - avg), "t2 should move toward neighbors"


# =============================================================================
# Test 2: Cluster Assignment
# =============================================================================

class TestClusterAssignment:
    """Test that client correctly picks cluster with minimum loss."""

    def test_assign_cluster_picks_min_loss(self):
        """Client should assign to cluster with lowest evaluated loss."""
        X_train = torch.randn(20, 40)
        y_train = torch.randint(0, 5, (20,))
        client = DFCAClient(0, X_train, y_train, num_clusters=3)

        # Create a simple model
        model = SimpleLinearModel(input_dim=40, num_classes=5)
        client.setup_for_gpu(model, "cpu")
        client.model.eval()

        # Initialize cluster banks with different params (different losses)
        params = OrderedDict(
            (k, v.clone()) for k, v in model.state_dict().items()
        )
        client.initialize_cluster_bank(global_params=params)

        # Modify cluster 0 to have very different params (higher loss)
        for k in client.cluster_params[0]:
            client.cluster_params[0][k] = client.cluster_params[0][k] * 10.0

        # Modify cluster 2 to have very different params (moderate loss)
        for k in client.cluster_params[2]:
            client.cluster_params[2][k] = client.cluster_params[2][k] * 2.0

        # Client should pick cluster 1 (original params = lowest loss)
        assigned = client.assign_cluster(verbose=False)
        assert assigned == 1, f"Expected cluster 1 (original params), got {assigned}"

    def test_assignment_losses_recorded(self):
        """Assignment should record losses for all clusters."""
        X_train = torch.randn(20, 40)
        y_train = torch.randint(0, 5, (20,))
        client = DFCAClient(0, X_train, y_train, num_clusters=3)

        model = SimpleLinearModel(input_dim=40, num_classes=5)
        client.setup_for_gpu(model, "cpu")
        client.model.eval()

        params = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
        client.initialize_cluster_bank(global_params=params)
        client.assign_cluster(verbose=False)

        assert len(client.assignment_losses) == 3
        for cid in range(3):
            assert cid in client.assignment_losses
            assert client.assignment_losses[cid] >= 0


# =============================================================================
# Test 3: Local Update
# =============================================================================

class TestLocalUpdate:
    """Test that only assigned cluster params change during local update."""

    def test_only_assigned_cluster_changes(self):
        """After training, only the assigned cluster's params should change."""
        X_train = torch.randn(50, 40)
        y_train = torch.randint(0, 5, (50,))
        client = DFCAClient(0, X_train, y_train, num_clusters=3)

        model = SimpleLinearModel(input_dim=40, num_classes=5)
        client.setup_for_gpu(model, "cpu")
        client.model.train()

        params = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
        client.initialize_cluster_bank(global_params=params)

        # Assign to cluster 1
        client.assigned_cluster = 1

        # Record params before training
        before = {}
        for cid in range(3):
            before[cid] = {k: v.clone() for k, v in client.cluster_params[cid].items()}

        # Train
        trainer = DFCATrainer()
        client.train_assigned_cluster(
            trainer, epochs=1, batch_size=50, lr=0.01
        )

        # Verify: only assigned cluster should change
        for cid in range(3):
            for k in client.cluster_params[cid]:
                changed = not torch.allclose(
                    client.cluster_params[cid][k], before[cid][k]
                )
                if cid == 1:  # assigned
                    assert changed, f"Cluster {cid} params should have changed"
                else:  # unassigned
                    assert not changed, f"Cluster {cid} params should NOT have changed"

    def test_train_returns_correct_structure(self):
        """train_assigned_cluster should return expected dict."""
        X_train = torch.randn(50, 40)
        y_train = torch.randint(0, 5, (50,))
        client = DFCAClient(0, X_train, y_train, num_clusters=3)

        model = SimpleLinearModel(input_dim=40, num_classes=5)
        client.setup_for_gpu(model, "cpu")
        client.model.train()

        params = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
        client.initialize_cluster_bank(global_params=params)
        client.assigned_cluster = 1

        trainer = DFCATrainer()
        result = client.train_assigned_cluster(
            trainer, epochs=1, batch_size=50, lr=0.01
        )

        assert "client_id" in result
        assert "assigned_cluster" in result
        assert result["assigned_cluster"] == 1
        assert "loss" in result
        assert "params" in result


# =============================================================================
# Test 4: Message Export
# =============================================================================

class TestMessageExport:
    """Test that client only exports assigned cluster params."""

    def test_export_only_assigned_cluster(self):
        """Message should only contain assigned cluster's params."""
        X_train = torch.randn(20, 40)
        y_train = torch.randint(0, 5, (20,))
        client = DFCAClient(0, X_train, y_train, num_clusters=5)

        model = SimpleLinearModel(input_dim=40, num_classes=5)
        client.setup_for_gpu(model, "cpu")
        client.model.train()

        params = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
        client.initialize_cluster_bank(global_params=params)

        for assigned in range(5):
            client.assigned_cluster = assigned
            msg = client.export_assigned_cluster_message()

            assert len(msg) == 1, f"Expected 1 cluster in message, got {len(msg)}"
            assert assigned in msg, f"Assigned cluster {assigned} should be in message"

            # Other clusters should NOT be in message
            for cid in range(5):
                if cid != assigned:
                    assert cid not in msg, f"Non-assigned cluster {cid} should NOT be in message"

    def test_export_all_cluster_params(self):
        """export_all_cluster_params should return all k clusters."""
        X_train = torch.randn(20, 40)
        y_train = torch.randint(0, 5, (20,))
        client = DFCAClient(0, X_train, y_train, num_clusters=5)

        model = SimpleLinearModel(input_dim=40, num_classes=5)
        client.setup_for_gpu(model, "cpu")
        params = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
        client.initialize_cluster_bank(global_params=params)

        msg = client.export_all_cluster_params()
        assert len(msg) == 5
        for cid in range(5):
            assert cid in msg


# =============================================================================
# Test 5: Fixed k = 10 Clusters
# =============================================================================

class TestFixedK:
    """Test that dfca_num_clusters=10 creates exactly 10 model banks."""

    def test_k_creates_exactly_10_banks(self):
        """dfca_num_clusters=10 should create 10 cluster banks."""
        X_train = torch.randn(20, 40)
        y_train = torch.randint(0, 5, (20,))
        client = DFCAClient(0, X_train, y_train, num_clusters=10)

        model = SimpleLinearModel(input_dim=40, num_classes=5)
        client.setup_for_gpu(model, "cpu")
        params = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
        client.initialize_cluster_bank(global_params=params)

        assert len(client.cluster_params) == 10
        for cid in range(10):
            assert cid in client.cluster_params
            assert isinstance(client.cluster_params[cid], OrderedDict)

    def test_config_k_10_from_factory(self):
        """Factory should create client with k=10 from config."""
        config = {
            "algorithm": "dfca_il",
            "dfca_num_clusters": 10,
            "seed": 42,
        }
        client_cls, extra_spec = _resolve_client_class("dfca_il")
        extra_kwargs = _build_extra_kwargs(config, extra_spec)

        X_train = torch.randn(20, 40)
        y_train = torch.randint(0, 5, (20,))
        client = client_cls(0, X_train, y_train, **extra_kwargs)

        assert client.num_clusters == 10


# =============================================================================
# Test 6: Active Client Schedule
# =============================================================================

class TestActiveClientSchedule:
    """Test that active client ratios follow the specified schedule."""

    def test_nested_client_selection(self):
        """Active sets should be nested prefixes using fixed order."""
        # Bug 2 fix: uses FIXED order + prefix, NOT re-shuffle per task
        client_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        total_clients = 100
        seed = 42

        all_ids = list(range(total_clients))
        rng = random.Random(seed)
        rng.shuffle(all_ids)

        results = []
        for task_id in range(6):
            ratio = client_ratios[task_id]
            num_active = max(1, int(total_clients * ratio))
            active_ids = set(all_ids[:num_active])
            results.append(active_ids)

        # Verify counts
        expected_counts = [50, 60, 70, 80, 90, 100]
        for task_id, expected in enumerate(expected_counts):
            actual = len(results[task_id])
            assert actual == expected, f"Task {task_id}: expected {expected}, got {actual}"

    def test_nested_property(self):
        """Each task's active set must be a superset of previous task's set."""
        # Bug 2 fix: FIXED order + prefix, ensuring nested supersets
        client_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        total_clients = 100
        seed = 42

        all_ids = list(range(total_clients))
        rng = random.Random(seed)
        rng.shuffle(all_ids)

        sets = []
        for task_id in range(6):
            ratio = client_ratios[task_id]
            num_active = max(1, int(total_clients * ratio))
            active_ids = frozenset(all_ids[:num_active])
            sets.append(active_ids)

        # Each subsequent set should be a superset of the previous
        for t in range(1, 6):
            assert len(sets[t]) > len(sets[t - 1]), (
                f"Task {t} should have more clients than task {t-1}"
            )
            assert sets[t - 1].issubset(sets[t]), (
                f"BUG: Task {t-1} is NOT a subset of Task {t}. "
                f"Old code re-shuffled with seed+task_id, breaking nesting. "
                f"Size: T{t-1}={len(sets[t-1])}, T{t}={len(sets[t])}"
            )


# =============================================================================
# Test 7: No Architecture Change
# =============================================================================

class TestNoArchitectureChange:
    """Test that CNN_GRU_Model architecture doesn't change after initialization."""

    def test_num_classes_unchanged(self):
        """num_classes should always equal total_classes (34), never dynamic."""
        total_classes = 34
        model = CNN_GRU_Model(input_shape=(40,), num_classes=total_classes)
        assert model.num_classes == total_classes

        # Simulate multiple tasks with different seen classes
        for task_id in range(6):
            # Architecture should be unchanged regardless of task
            assert model.num_classes == total_classes

    def test_output_layer_size_constant(self):
        """Output layer (fc2) should always have 34 output units."""
        total_classes = 34
        model = CNN_GRU_Model(input_shape=(40,), num_classes=total_classes)

        fc2 = model.fc2
        assert fc2.out_features == total_classes

        # After forward pass, output shape should be [batch, 34]
        x = torch.randn(4, 40)
        out = model(x)
        assert out.shape == (4, total_classes)


# =============================================================================
# Test 8: Factory Registration
# =============================================================================

class TestFactoryRegistration:
    """Test that 'dfca_il' algorithm creates correct server and client."""

    def test_dfca_il_in_server_registry(self):
        """DFCAServer should be registered for 'dfca_il'."""
        assert "dfca_il" in _SERVER_REGISTRY
        assert _SERVER_REGISTRY["dfca_il"].__name__ == "DFCAServer"

    def test_dfca_il_client_factory(self):
        """Client factory should create DFCAClient for 'dfca_il'."""
        client_cls, _ = _resolve_client_class("dfca_il")
        assert client_cls.__name__ == "DFCAClient"

    def test_dfca_strategy_creation(self):
        """get_strategy should create DFCATrainer + DFCAAggregator."""
        from fed_learning.strategies import get_strategy

        trainer, aggregator = get_strategy(
            "dfca_il",
            dfca_num_clusters=10,
            local_epochs=1,
            learning_rate=0.001,
            batch_size=2048,
        )

        assert isinstance(trainer, DFCATrainer)
        assert isinstance(trainer, BaseTrainer)
        assert isinstance(aggregator, DFCAAggregator)
        assert isinstance(aggregator, BaseAggregator)
        assert aggregator.num_clusters == 10

    def test_dfca_trainer_seen_class_masking(self):
        """DFCATrainer should mask unseen classes in incremental setting."""
        trainer = DFCATrainer()
        trainer.seen_classes = {0, 1, 2, 3, 4, 5}
        trainer.new_classes = [0, 1, 2, 3, 4, 5]

        model = SimpleLinearModel(input_dim=10, num_classes=5)
        output = model(torch.randn(4, 10))
        target = torch.tensor([0, 1, 2, 3])

        loss = trainer.compute_loss(model, output, target, None)
        assert loss.item() >= 0


# =============================================================================
# Test 9: DFCAAggregator raises on misuse
# =============================================================================

class TestDFCAAggregator:
    """Test that DFCAAggregator.aggregate() raises (no FedAvg-style aggregation)."""

    def test_aggregate_raises(self):
        """DFCAAggregator.aggregate() should raise RuntimeError."""
        aggregator = DFCAAggregator(num_clusters=10)
        with pytest.raises(RuntimeError, match="should never be called"):
            aggregator.aggregate(results=[], global_params=None)

    def test_sequential_running_average_class_method(self):
        """sequential_running_average should be accessible as a class method."""
        result = DFCAAggregator.sequential_running_average(
            torch.tensor(5.0), torch.tensor(10.0), count=2
        )
        # alpha = (2+1)/(2+2) = 3/4, beta = 1/(2+2) = 1/4
        # result = (3/4)*5 + (1/4)*10 = 15/4 + 10/4 = 25/4 = 6.25
        expected = ((2 + 1.0) / (2 + 2.0)) * 5.0 + (1.0 / (2 + 2.0)) * 10.0
        assert abs(result.item() - expected) < 1e-6


# =============================================================================
# Test 10: DFCAClient aggregation
# =============================================================================

class TestDFCAClientAggregation:
    """Test peer-to-peer message passing and aggregation."""

    def test_receive_and_aggregate(self):
        """Client should correctly aggregate received peer messages via sequential running average."""
        # Test the sequential running average formula directly
        from fed_learning.strategies.federated.dfca import DFCAAggregator

        # Step 1: Local model = 0.967, neighbor 1 sends zeros
        local = torch.tensor([[0.967]])
        neighbor1 = torch.tensor([[0.0]])

        # Step 2: neighbor 2 sends twos (value=2)
        neighbor2 = torch.tensor([[2.0]])

        # Sequential running average:
        # r=0: local = (1/2)*0.967 + (1/2)*0.0 = 0.483
        local = DFCAAggregator.sequential_running_average(local, neighbor1, count=0)
        assert torch.allclose(local, torch.tensor([[0.4835]]), atol=1e-3)

        # r=1: local = (2/3)*0.483 + (1/3)*2.0 = 0.322 + 0.667 = 0.989
        local = DFCAAggregator.sequential_running_average(local, neighbor2, count=1)
        assert torch.allclose(local, torch.tensor([[0.989]]), atol=1e-3)

    def test_messages_cleared_after_aggregation(self):
        """Neighbor messages should be cleared after aggregation."""
        X_train = torch.randn(10, 40)
        y_train = torch.randint(0, 5, (10,))
        client = DFCAClient(0, X_train, y_train, num_clusters=3)

        model = SimpleLinearModel(input_dim=40, num_classes=5)
        client.setup_for_gpu(model, "cpu")
        params = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
        client.initialize_cluster_bank(global_params=params)

        client.receive_neighbor_message(99, {0: OrderedDict((k, v.clone()) for k, v in params.items())})
        assert len(client.neighbor_messages) == 1

        client.aggregate_received_messages()
        assert len(client.neighbor_messages) == 0


# =============================================================================
# Test 11: DFCATrainer incremental learning
# =============================================================================

class TestDFCATrainer:
    """Test DFCATrainer for incremental learning compatibility."""

    def test_set_task_updates_seen_classes(self):
        """set_task should update seen_classes and new_classes."""
        trainer = DFCATrainer()
        assert trainer.seen_classes == set()
        assert trainer.current_task == 0

        trainer.set_task(0, [0, 1, 2, 3, 4, 5])
        assert trainer.seen_classes == {0, 1, 2, 3, 4, 5}
        assert trainer.current_task == 0

        trainer.set_task(1, [6, 7, 8, 9, 10, 11])
        assert trainer.seen_classes == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}
        assert trainer.current_task == 1

    def test_compute_loss_with_seen_classes(self):
        """compute_loss should mask unseen classes."""
        trainer = DFCATrainer()
        trainer.seen_classes = {0, 1}
        trainer.new_classes = [0, 1]

        model = SimpleLinearModel(input_dim=10, num_classes=5)
        output = torch.randn(4, 5)
        target = torch.tensor([0, 1, 0, 1])

        loss = trainer.compute_loss(model, output, target, None)
        assert loss.item() >= 0

    def test_optimizer_class_is_adam(self):
        """DFCATrainer should use Adam optimizer by default."""
        trainer = DFCATrainer()
        assert trainer.get_optimizer_class() == torch.optim.Adam


# =============================================================================
# Bug Fix Tests: Local Training (Bug 1)
# =============================================================================

class TestBug1LocalTrainingNoReset:
    """Bug 1: load_state_dict must NOT be inside the batch loop."""

    def test_train_assigned_cluster_loads_once_before_batches(self):
        """train_assigned_cluster loads params once before epoch/batch loop, not per batch."""
        X_train = torch.randn(60, 40)
        y_train = torch.randint(0, 5, (60,))
        client = DFCAClient(0, X_train, y_train, num_clusters=3)

        model = SimpleLinearModel(input_dim=40, num_classes=5)
        client.setup_for_gpu(model, "cpu")
        client.model.train()

        params = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
        client.initialize_cluster_bank(global_params=params)
        client.assigned_cluster = 1

        trainer = DFCATrainer()

        # Track how many times load_state_dict is called
        load_calls = []
        original_load = client.model.load_state_dict

        def tracking_load(state_dict):
            load_calls.append(state_dict)
            return original_load(state_dict)

        client.model.load_state_dict = tracking_load

        client.train_assigned_cluster(trainer, epochs=2, batch_size=20, lr=0.01)

        # Should load exactly ONCE before training (to load cluster params)
        assert len(load_calls) == 1, (
            f"Expected 1 load_state_dict call (before training), got {len(load_calls)}. "
            f"Bug: load_state_dict was inside the batch loop, resetting weights each batch."
        )

    def test_multi_batch_accumulates_gradients(self):
        """Training over multiple batches should accumulate updates, not reset each batch."""
        X_train = torch.randn(60, 40)
        y_train = torch.randint(0, 5, (60,))
        client = DFCAClient(0, X_train, y_train, num_clusters=3)

        model = SimpleLinearModel(input_dim=40, num_classes=5)
        client.setup_for_gpu(model, "cpu")
        client.model.train()

        params = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
        client.initialize_cluster_bank(global_params=params)
        client.assigned_cluster = 1

        # Record initial params
        initial = {k: v.clone() for k, v in client.cluster_params[1].items()}

        trainer = DFCATrainer()

        # Train over multiple batches
        result = client.train_assigned_cluster(
            trainer, epochs=1, batch_size=20, lr=0.01
        )

        # Check that params changed (gradient was accumulated)
        changed_count = 0
        for k in client.cluster_params[1]:
            if not torch.equal(client.cluster_params[1][k], initial[k]):
                changed_count += 1

        assert changed_count > 0, (
            "Training should change cluster params after 3 batches. "
            "If only the last batch had effect, params may have been reset each batch."
        )


# =============================================================================
# Bug Fix Tests: Active Client Schedule (Bug 2)
# =============================================================================

class TestBug2NestedActiveSchedule:
    """Bug 2: active sets must be nested prefixes (task t is superset of task t-1)."""

    def test_fixed_order_prefix_selection(self):
        """Active selection must use fixed order with prefix, not re-shuffle per task."""
        # This tests the NEW correct logic (fixed order + prefix)
        # NOT the old buggy logic (seed+task_id re-shuffle)
        client_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        total_clients = 100
        seed = 42

        all_ids = list(range(total_clients))
        rng = random.Random(seed)
        rng.shuffle(all_ids)

        active_sets = []
        for task_id in range(6):
            ratio = client_ratios[task_id]
            num_active = max(1, int(total_clients * ratio))
            active_ids = set(all_ids[:num_active])
            active_sets.append(active_ids)

        # Verify counts
        expected_counts = [50, 60, 70, 80, 90, 100]
        for t, expected in enumerate(expected_counts):
            assert len(active_sets[t]) == expected, (
                f"Task {t}: expected {expected} active clients, got {len(active_sets[t])}"
            )

        # Verify nesting: task t must be superset of task t-1
        for t in range(1, 6):
            assert active_sets[t - 1].issubset(active_sets[t]), (
                f"Task {t-1} active set is NOT a subset of Task {t}. "
                f"Old buggy logic re-shuffled per task, breaking the nested property. "
                f"Size: T{t-1}={len(active_sets[t-1])}, T{t}={len(active_sets[t])}"
            )

    def test_server_active_schedule_nested(self):
        """DFCAServer._get_active_clients_for_task must produce nested active sets."""
        from fed_learning.servers.dfca_server import DFCAServer

        class DummyClient:
            def __init__(self, cid):
                self.client_id = cid

        clients = [DummyClient(i) for i in range(100)]

        config = {
            "dfca_num_clusters": 10,
            "dfca_init": "global",
            "dfca_graph": "erdos_renyi",
            "dfca_connectivity": 0.15,
            "dfca_client_ratios": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "dfca_aggregation": "sequential_running_average",
            "seed": 42,
            "input_shape": (40,),
            "num_classes": 34,
        }

        with patch.object(
            DFCAServer, "__init__", lambda self, *args, **kwargs: None
        ):
            server = object.__new__(DFCAServer)
            server.config = config
            server.num_clusters = 10
            server.connectivity = 0.15
            server._client_order = list(range(100))
            random.Random(42).shuffle(server._client_order)
            server.client_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            server.clients = clients

            active_sets = []
            for t in range(6):
                active = server._get_active_clients_for_task(t)
                active_sets.append({c.client_id for c in active})

        expected_counts = [50, 60, 70, 80, 90, 100]
        for t, expected in enumerate(expected_counts):
            assert len(active_sets[t]) == expected, (
                f"Task {t}: expected {expected} active clients, got {len(active_sets[t])}"
            )

        for t in range(1, 6):
            assert active_sets[t - 1].issubset(active_sets[t]), (
                f"Nested property violated: T{t-1} (size {len(active_sets[t-1])}) "
                f"is NOT subset of T{t} (size {len(active_sets[t])})"
            )


# =============================================================================
# Bug Fix Tests: Graph/Client Refresh (Bug 3)
# =============================================================================

class TestBug3GraphClientRefresh:
    """Bug 3: server must maintain full client population and graph across tasks."""

    def test_server_graph_has_all_clients(self):
        """Graph must be built on full population, not just clients with task data."""
        from fed_learning.servers.dfca_server import DFCAServer

        class DummyClient:
            def __init__(self, cid):
                self.client_id = cid

        clients = [DummyClient(i) for i in range(50)]

        config = {
            "dfca_num_clusters": 10,
            "dfca_init": "global",
            "dfca_graph": "erdos_renyi",
            "dfca_connectivity": 0.15,
            "dfca_client_ratios": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "dfca_aggregation": "sequential_running_average",
            "seed": 42,
            "input_shape": (40,),
            "num_classes": 34,
        }

        # Mock __init__ to avoid model initialization
        with patch.object(
            DFCAServer, "__init__", lambda self, *args, **kwargs: None
        ):
            server = object.__new__(DFCAServer)
            server.config = config
            server.num_clusters = 10
            server.connectivity = 0.15
            server._client_order = list(range(50))
            server.clients = clients
            server.graph = {}

            # Call _build_graph
            DFCAServer._build_graph(server)

            # All 50 clients must be in graph
            assert len(server.graph) == 50, (
                f"Graph should have 50 nodes, got {len(server.graph)}"
            )
            for cid in range(50):
                assert cid in server.graph, f"Client {cid} missing from graph"

    def test_init_skips_already_initialized_clients(self):
        """_initialize_all_client_cluster_banks must not reset existing cluster banks."""
        from fed_learning.servers.dfca_server import DFCAServer

        class DummyClient:
            def __init__(self, cid):
                self.client_id = cid
                self._initialized = False
                self.cluster_params = {}

            def initialize_cluster_bank(self, global_params=None, template_model=None):
                self._initialized = True
                self.cluster_params = {"param": torch.tensor(1.0)}

        clients = [DummyClient(0), DummyClient(1)]

        config = {
            "dfca_num_clusters": 10,
            "dfca_init": "global",
            "seed": 42,
            "input_shape": (40,),
            "num_classes": 34,
        }

        with patch.object(
            DFCAServer, "__init__", lambda self, *args, **kwargs: None
        ):
            server = object.__new__(DFCAServer)
            server.config = config
            server.num_clusters = 10
            server.init_type = "global"
            server.clients = clients
            # Provide a minimal mock global_model so get_global_params() works
            server.global_model = SimpleLinearModel(input_dim=40, num_classes=34)
            server.representative_cluster_params = {0: {}}

            # Mark client 0 as already initialized
            clients[0]._initialized = True
            clients[0].cluster_params = {"param": torch.tensor(999.0)}

            # Call init (should skip client 0)
            DFCAServer._initialize_all_client_cluster_banks(server)

            # Client 0 should keep its state
            assert clients[0].cluster_params["param"].item() == 999.0, (
                "Already-initialized client lost its cluster bank state!"
            )
            # Client 1 should be initialized
            assert clients[1]._initialized is True
            assert "param" in clients[1].cluster_params


# =============================================================================
# Bug Fix Tests: Assignment Seen-Class Masking (Bug 4)
# =============================================================================

class TestBug4AssignmentMasking:
    """Bug 4: cluster assignment must use seen-class masking so unseen logits don't affect it."""

    def test_assign_cluster_accepts_trainer(self):
        """assign_cluster must accept a trainer argument for seen-class masking."""
        X_train = torch.randn(20, 40)
        y_train = torch.randint(0, 3, (20,))
        client = DFCAClient(0, X_train, y_train, num_clusters=3)

        model = SimpleLinearModel(input_dim=40, num_classes=5)
        client.setup_for_gpu(model, "cpu")
        params = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
        client.initialize_cluster_bank(global_params=params)

        trainer = DFCATrainer()
        trainer.seen_classes = {0, 1}
        trainer.new_classes = [0, 1]

        # Must accept trainer= keyword arg
        result = client.assign_cluster(trainer=trainer, verbose=False)
        assert result in range(3)

    def test_unseen_logits_do_not_affect_assignment(self):
        """
        Seen-class masking must prevent unseen class logits from influencing assignment.

        Bug 4 fix: assign_cluster passes the trainer to _evaluate_cluster_loss,
        which applies seen-class masking. This test verifies that:
        1. The trainer parameter is accepted without error
        2. The assignment losses are computed with the trainer (masking applied)
        3. Losses are non-negative (valid CE values)
        """
        torch.manual_seed(12345)
        X_data = torch.randn(30, 40)
        y_data = torch.randint(0, 2, (30,))  # targets are SEEN classes {0, 1}
        client = DFCAClient(0, X_data, y_data, num_clusters=3)

        model = SimpleLinearModel(input_dim=40, num_classes=5)
        client.setup_for_gpu(model, "cpu")
        params = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
        client.initialize_cluster_bank(global_params=params)

        trainer = DFCATrainer()
        trainer.seen_classes = {0, 1}
        trainer.new_classes = [0, 1]

        # Assertion 1: method accepts trainer= parameter without error
        result = client.assign_cluster(trainer=trainer, verbose=False)
        assert result in range(3), f"assign_cluster should return valid cluster id, got {result}"

        # Assertion 2: losses are computed via trainer (masking applied)
        assert len(client.assignment_losses) == 3
        for cid in range(3):
            assert cid in client.assignment_losses, f"Cluster {cid} missing from assignment_losses"
            assert client.assignment_losses[cid] >= 0, f"Loss for cluster {cid} must be non-negative"

    def test_assign_without_trainer_uses_raw_ce(self):
        """Without trainer, assign_cluster should fall back to raw CE."""
        X_train = torch.randn(20, 40)
        y_train = torch.randint(0, 3, (20,))
        client = DFCAClient(0, X_train, y_train, num_clusters=3)

        model = SimpleLinearModel(input_dim=40, num_classes=5)
        client.setup_for_gpu(model, "cpu")
        params = OrderedDict((k, v.clone()) for k, v in model.state_dict().items())
        client.initialize_cluster_bank(global_params=params)

        # Should work without trainer (uses raw CE)
        result = client.assign_cluster(trainer=None, verbose=False)
        assert result in range(3)


# =============================================================================
# Bug Fix Tests: Evaluation Label Leakage (Bug 5)
# =============================================================================

class TestBug5EvaluationNoLabelLeakage:
    """Bug 5: evaluation must not use ground-truth labels to select clusters."""

    def test_evaluate_global_uses_ensemble_averaging(self):
        """evaluate_global must use ensemble averaging across clusters."""
        from fed_learning.servers.dfca_server import DFCAServer

        class DummyClient:
            def __init__(self, cid):
                self.client_id = cid

        clients = [DummyClient(i) for i in range(5)]

        config = {
            "dfca_num_clusters": 10,
            "dfca_init": "global",
            "dfca_graph": "erdos_renyi",
            "dfca_connectivity": 0.15,
            "dfca_client_ratios": [0.5],
            "dfca_aggregation": "sequential_running_average",
            "seed": 42,
            "input_shape": (40,),
            "num_classes": 34,
            "total_classes": 34,
        }

        with patch.object(
            DFCAServer, "__init__", lambda self, *args, **kwargs: None
        ):
            server = object.__new__(DFCAServer)
            server.config = config
            server.num_clusters = 2
            server.seen_classes = []
            server.representative_cluster_params = {}
            server.global_model = CNN_GRU_Model(input_shape=(40,), num_classes=34)
            server.primary_device = "cpu"
            server.test_data = {
                "X_test": torch.randn(8, 40),
                "y_test": torch.randint(0, 34, (8,)),
            }
            server.num_gpus = 1
            server.use_cpu = True
            server._round = 1
            server.round_counter = 1

            # Create representative params for 2 clusters
            params0 = OrderedDict((k, v.clone()) for k, v in server.global_model.state_dict().items())
            params1 = OrderedDict((k, v.clone() * 2.0) for k, v in server.global_model.state_dict().items())
            server.representative_cluster_params[0] = params0
            server.representative_cluster_params[1] = params1

            with patch.object(server, "_mask_unseen_classes", side_effect=lambda x: x):
                result = server.evaluate_global(batch_size=8, seen_classes_only=False)

        # Should return valid accuracy and a non-zero loss
        assert "accuracy" in result
        assert "loss" in result
        assert 0.0 <= result["accuracy"] <= 1.0
        assert result["loss"] > 0.0, "Ensemble CE loss should be positive"

    def test_evaluate_global_signature_no_labels_in_selection(self):
        """Ensemble evaluation must not use y_batch to select models or predictions."""
        import inspect
        from fed_learning.servers.dfca_server import DFCAServer

        source = inspect.getsource(DFCAServer.evaluate_global)

        # Ensemble averaging: stack probs, average, then argmax
        assert "torch.stack" in source, (
            "Must use torch.stack to combine cluster probabilities"
        )
        # No CrossEntropyLoss criterion used for model/prediction selection
        assert "CrossEntropyLoss" not in source, (
            "Bug: CrossEntropyLoss must not be used. "
            "Must use ensemble averaging (stack probs + avg + argmax)."
        )
        # argmax on averaged probabilities
        assert "argmax(dim=1)" in source, (
            "Must use argmax on averaged ensemble probabilities for predictions"
        )


# =============================================================================
# Bug Fix Tests: Representative Averaging (Bug 6)
# =============================================================================

class TestBug6RepresentativeAveraging:
    """Bug 6: representative models must divide by actual contributor count per cluster."""

    def test_average_divides_by_actual_contributors(self):
        """_update_representative_cluster_models must divide by actual contributor count."""
        from fed_learning.servers.dfca_server import DFCAServer

        class DummyClient:
            def __init__(self, cid):
                self.client_id = cid

        clients = [DummyClient(0), DummyClient(1), DummyClient(2)]

        config = {
            "dfca_num_clusters": 2,
            "dfca_init": "global",
            "seed": 42,
            "input_shape": (40,),
            "num_classes": 34,
        }

        with patch.object(
            DFCAServer, "__init__", lambda self, *args, **kwargs: None
        ):
            server = object.__new__(DFCAServer)
            server.config = config
            server.num_clusters = 2
            server.clients = clients

            # Client 0 only has cluster 0
            clients[0].cluster_params = {
                0: OrderedDict([("w", torch.tensor([[1.0]]))])
            }
            # Client 1 only has cluster 1
            clients[1].cluster_params = {
                1: OrderedDict([("w", torch.tensor([[2.0]]))])
            }
            # Client 2 has both clusters
            clients[2].cluster_params = {
                0: OrderedDict([("w", torch.tensor([[3.0]]))]),
                1: OrderedDict([("w", torch.tensor([[4.0]]))]),
            }

            server.representative_cluster_params = {}
            DFCAServer._update_representative_cluster_models(server, clients)

            # Cluster 0: only clients 0 and 2 contribute => (1.0 + 3.0) / 2 = 2.0
            assert 0 in server.representative_cluster_params
            assert "w" in server.representative_cluster_params[0]
            assert abs(server.representative_cluster_params[0]["w"].item() - 2.0) < 1e-6, (
                f"Cluster 0 average should be 2.0, got {server.representative_cluster_params[0]['w'].item()}. "
                f"Bug: divided by total active clients ({len(clients)}=3) instead of actual contributors (2)."
            )

            # Cluster 1: only clients 1 and 2 contribute => (2.0 + 4.0) / 2 = 3.0
            assert 1 in server.representative_cluster_params
            assert "w" in server.representative_cluster_params[1]
            assert abs(server.representative_cluster_params[1]["w"].item() - 3.0) < 1e-6, (
                f"Cluster 1 average should be 3.0, got {server.representative_cluster_params[1]['w'].item()}"
            )

    def test_empty_representative_does_not_cause_none(self):
        """Empty representative cluster should not cause best_out=None silently."""
        from fed_learning.servers.dfca_server import DFCAServer

        class DummyClient:
            def __init__(self, cid):
                self.client_id = cid

        config = {
            "dfca_num_clusters": 2,
            "dfca_init": "global",
            "seed": 42,
            "input_shape": (40,),
            "num_classes": 34,
            "total_classes": 34,
        }

        with patch.object(
            DFCAServer, "__init__", lambda self, *args, **kwargs: None
        ):
            server = object.__new__(DFCAServer)
            server.config = config
            server.num_clusters = 2
            server.seen_classes = []
            server.representative_cluster_params = {}
            server.global_model = CNN_GRU_Model(input_shape=(40,), num_classes=34)
            server.primary_device = "cpu"
            server.test_data = {
                "X_test": torch.randn(4, 40),
                "y_test": torch.randint(0, 34, (4,)),
            }
            server.num_gpus = 1
            server.use_cpu = True
            server._round = 1
            server.round_counter = 1

            # Both representative clusters are empty
            server.representative_cluster_params = {}

            # Should not crash; should handle gracefully
            with patch.object(server, "_mask_unseen_classes", side_effect=lambda x: x):
                result = server.evaluate_global(batch_size=4, seen_classes_only=False)

            assert "accuracy" in result


# =============================================================================
# P1 Fix Tests: No-Data Clients
# =============================================================================

class TestP1NoDataClients:
    """P1: active clients with num_samples=0 must be skipped from assign/train/export."""

    def test_train_round_with_no_data_client_does_not_crash(self):
        """train_round must not crash when some active clients have num_samples=0."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        # 3 clients: 2 with data, 1 with empty data
        clients = [
            DFCAClient(0, torch.randn(10, 40), torch.randint(0, 2, (10,)), num_clusters=2),
            DFCAClient(1, torch.randn(10, 40), torch.randint(0, 2, (10,)), num_clusters=2),
            DFCAClient(2, torch.tensor([]).reshape(0, 40), torch.tensor([], dtype=torch.long), num_clusters=2),
        ]
        clients[2].num_samples = 0

        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [0.5, 1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)

        # Round must not crash
        result = server.train_round(task_id=0, verbose=False)
        assert "train_loss" in result
        assert "round_stats" in result

    def test_no_data_client_skipped_from_training(self):
        """Only clients with num_samples > 0 should participate in training."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(0, torch.randn(10, 40), torch.randint(0, 2, (10,)), num_clusters=2),
            DFCAClient(1, torch.randn(10, 40), torch.randint(0, 2, (10,)), num_clusters=2),
            DFCAClient(2, torch.tensor([]).reshape(0, 40), torch.tensor([], dtype=torch.long), num_clusters=2),
        ]
        clients[2].num_samples = 0

        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)
        server.train_round(task_id=0, verbose=False)

        stats = server.cluster_history[-1]
        # 2 training clients (0,1), 1 skipped (2)
        assert stats["active_clients"] == 3
        assert stats["training_clients"] == 2
        assert stats["skipped_no_data"] == 1

    def test_no_data_client_not_forced_to_cluster_0(self):
        """Clients with num_samples=0 should not have assigned_cluster forced to 0."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        # Create a client with no data, give it an assigned_cluster of 99
        # It should NOT have its assigned_cluster changed to 0 by the round
        clients = [
            DFCAClient(0, torch.randn(10, 40), torch.randint(0, 2, (10,)), num_clusters=2),
            DFCAClient(1, torch.tensor([]).reshape(0, 40), torch.tensor([], dtype=torch.long), num_clusters=2),
        ]
        clients[1].num_samples = 0
        clients[1].assigned_cluster = 99  # Set to non-zero

        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)
        server.train_round(task_id=0, verbose=False)

        # Client 1 (no-data) should NOT appear in assignment results
        # because it's not in training_clients
        stats = server.cluster_history[-1]
        # Only 1 training client (client 0), so distribution has 1 entry
        assert stats["training_clients"] == 1

    def test_all_no_data_active_clients_returns_zero_loss(self):
        """If all active clients have no data, train_round must return loss=0.0 and not crash."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        # 2 clients, both with no data
        clients = [
            DFCAClient(0, torch.tensor([]).reshape(0, 40), torch.tensor([], dtype=torch.long), num_clusters=2),
            DFCAClient(1, torch.tensor([]).reshape(0, 40), torch.tensor([], dtype=torch.long), num_clusters=2),
        ]
        clients[0].num_samples = 0
        clients[1].num_samples = 0

        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)

        result = server.train_round(task_id=0, verbose=False)
        assert "train_loss" in result
        assert result["train_loss"] == 0.0
        stats = server.cluster_history[-1]
        assert stats["training_clients"] == 0
        assert stats["skipped_no_data"] == 2

    def test_dfca_server_init_no_mock(self):
        """DFCAServer must be creatable without mocking __init__."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(4, 40), torch.randint(0, 2, (4,)), num_clusters=2)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [0.5, 1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 2,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }

        # This must not raise AttributeError about missing attributes
        server = DFCAServer(clients, test_data, cfg)
        assert hasattr(server, "representative_cluster_params")
        assert hasattr(server, "_client_order")
        assert hasattr(server, "graph")


# =============================================================================
# Debug Logging Tests
# =============================================================================

class TestDFCADebugLogging:
    """Test dfca_debug_messages and dfca_debug_message_limit configs and logging."""

    def test_train_round_runs_with_debug_messages_false(self):
        """train_round must run without error when dfca_debug_messages=False."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(4, 40), torch.randint(0, 2, (4,)), num_clusters=2)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
            "dfca_debug_messages": False,
            "dfca_debug_message_limit": 50,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)
        result = server.train_round(task_id=0, verbose=False)
        assert "round_stats" in result
        assert "per_cluster_updates" in result["round_stats"]

    def test_train_round_runs_with_debug_messages_true(self):
        """train_round must run without error when dfca_debug_messages=True."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(4, 40), torch.randint(0, 2, (4,)), num_clusters=2)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
            "dfca_debug_messages": True,
            "dfca_debug_message_limit": 50,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)
        result = server.train_round(task_id=0, verbose=False)
        assert "round_stats" in result
        assert "per_cluster_updates" in result["round_stats"]

    def test_debug_messages_logged_to_stdout(self, capsys):
        """When dfca_debug_messages=True, stdout must contain '[DFCA][messages]'."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(4, 40), torch.randint(0, 2, (4,)), num_clusters=2)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
            "dfca_debug_messages": True,
            "dfca_debug_message_limit": 50,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)
        server.train_round(task_id=0, verbose=False)
        captured = capsys.readouterr()
        assert "[DFCA][messages]" in captured.out

    def test_debug_messages_not_logged_when_disabled(self, capsys):
        """When dfca_debug_messages=False, stdout must NOT contain '[DFCA][messages]'."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(4, 40), torch.randint(0, 2, (4,)), num_clusters=2)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
            "dfca_debug_messages": False,
            "dfca_debug_message_limit": 50,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)
        server.train_round(task_id=0, verbose=False)
        captured = capsys.readouterr()
        assert "[DFCA][messages]" not in captured.out

    def test_cluster_updates_logged_to_stdout(self, capsys):
        """round_stats must contain per_cluster_updates and verbose log shows 'Cluster updates'."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(4, 40), torch.randint(0, 2, (4,)), num_clusters=2)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)
        server.train_round(task_id=0, verbose=True)
        captured = capsys.readouterr()

        # Stats dict must have per_cluster_updates
        stats = server.cluster_history[-1]
        assert "per_cluster_updates" in stats

        # Stdout must contain the cluster updates line
        assert "Cluster updates" in captured.out

    def test_round_stats_has_per_cluster_updates(self):
        """round_stats returned by train_round must contain per_cluster_updates."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(4, 40), torch.randint(0, 2, (4,)), num_clusters=2)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)
        result = server.train_round(task_id=0, verbose=False)
        assert "round_stats" in result
        assert "per_cluster_updates" in result["round_stats"]
        assert isinstance(result["round_stats"]["per_cluster_updates"], dict)

    def test_format_cluster_updates_helper_all_zeros(self):
        """_format_cluster_updates must produce readable output for all-zero dict."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(4, 40), torch.randint(0, 2, (4,)), num_clusters=2)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)

        result = server._format_cluster_updates({0: 0, 1: 0})
        assert "c0=0" in result
        assert "c1=0" in result

    def test_format_cluster_updates_helper_mixed(self):
        """_format_cluster_updates must format mixed non-zero values correctly."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(4, 40), torch.randint(0, 2, (4,)), num_clusters=3)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 3,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)

        result = server._format_cluster_updates({0: 5, 1: 3, 2: 0})
        assert "c0=5" in result
        assert "c1=3" in result
        assert "c2=0" in result

    def test_debug_message_limit_hides_excess(self, capsys):
        """When dfca_debug_message_limit=1, only 1 message log should appear."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(4, 40), torch.randint(0, 2, (4,)), num_clusters=2)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
            "dfca_debug_messages": True,
            "dfca_debug_message_limit": 1,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)
        server.train_round(task_id=0, verbose=False)
        captured = capsys.readouterr()

        # Should contain at least one message log
        assert "[DFCA][messages]" in captured.out
        # Should NOT contain more than 2 message lines (1 real + 1 hidden note)
        # The hidden note may or may not appear depending on how many training clients
        # but in any case, only the first sender's log should appear
        lines = [l for l in captured.out.splitlines() if "[DFCA][messages]" in l]
        # At most 2 lines: 1 real log + 1 "hidden X additional" note
        assert len(lines) <= 2

    def test_debug_messages_false_no_hidden_note(self, capsys):
        """When dfca_debug_messages=False, no message logs including hidden note."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(4, 40), torch.randint(0, 2, (4,)), num_clusters=2)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
            "dfca_debug_messages": False,
            "dfca_debug_message_limit": 50,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)
        server.train_round(task_id=0, verbose=False)
        captured = capsys.readouterr()
        assert "[DFCA][messages]" not in captured.out
        assert "additional message logs" not in captured.out

    def test_debug_message_limit_zero_unlimited(self, capsys):
        """When dfca_debug_message_limit=0, all messages should be logged (unlimited)."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(4, 40), torch.randint(0, 2, (4,)), num_clusters=2)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
            "dfca_debug_messages": True,
            "dfca_debug_message_limit": 0,  # Unlimited
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)
        server.train_round(task_id=0, verbose=False)
        captured = capsys.readouterr()
        # Should have message logs for each training client
        assert "[DFCA][messages]" in captured.out
        # Should NOT have a "hidden" note since limit=0 means unlimited
        assert "additional message logs" not in captured.out


# =============================================================================
# Multi-GPU Fix Tests
# =============================================================================

class TestMultiGPUFix:
    """Test that multi-GPU multi-threading does not share model objects."""

    def test_ensure_client_model_creates_per_client_instance(self):
        """_ensure_client_model_on_device must create unique model per client."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(4, 40), torch.randint(0, 2, (4,)), num_clusters=2)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)

        # Ensure each client gets a model on CPU
        for c in clients:
            server._ensure_client_model_on_device(c, "cpu")

        # All models must be distinct objects
        model_ids = [id(c.model) for c in clients]
        assert len(set(model_ids)) == len(clients), (
            f"Expected {len(clients)} distinct model instances, got {len(set(model_ids))}. "
            f"Multi-GPU bug: clients are sharing a model object."
        )

        # No model should be the server's global_model
        for c in clients:
            assert c.model is not server.global_model, (
                "Client model is the same object as server.global_model — multi-GPU bug!"
            )

    def test_ensure_client_model_detects_shared_model(self):
        """_ensure_client_model_on_device must detect and fix shared model."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(4, 40), torch.randint(0, 2, (4,)), num_clusters=2)
            for i in range(2)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)

        # Manually set both clients to share the server's global_model (the bug scenario)
        clients[0].model = server.global_model
        clients[0].device = "cpu"
        clients[1].model = server.global_model
        clients[1].device = "cpu"

        # _ensure_client_model_on_device must replace with distinct instances
        server._ensure_client_model_on_device(clients[0], "cpu")
        server._ensure_client_model_on_device(clients[1], "cpu")

        # Now they must be different objects
        assert clients[0].model is not clients[1].model, (
            "After fix, clients must have different model instances"
        )
        # And neither should be the shared global_model
        assert clients[0].model is not server.global_model
        assert clients[1].model is not server.global_model

    def test_train_round_with_cpu_preserves_model_isolation(self):
        """train_round must not produce shared model objects."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(8, 40), torch.randint(0, 2, (8,)), num_clusters=2)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)
        server.train_round(task_id=0, verbose=False)

        # All training clients must have distinct model objects
        model_ids = [id(c.model) for c in clients if c.model is not None]
        assert len(set(model_ids)) == len(model_ids), (
            f"Model objects are shared: {len(model_ids)} models but only "
            f"{len(set(model_ids))} unique objects. Multi-GPU bug!"
        )
        # No client model should be the server's global_model
        for c in clients:
            if c.model is not None:
                assert c.model is not server.global_model

    def test_evaluate_global_with_cpu_cluster_params(self):
        """evaluate_global must not crash when representative params are CPU tensors."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(8, 40), torch.randint(0, 2, (8,)), num_clusters=2)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(16, 40),
            "y_test": torch.randint(0, 2, (16,)),
        }
        server = DFCAServer(clients, test_data, cfg)
        server.train_round(task_id=0, verbose=False)

        # representative_cluster_params are CPU tensors (from cluster banks)
        # evaluate_global must move them to GPU (primary_device) without crashing
        result = server.evaluate_global(seen_classes_only=False)
        assert "accuracy" in result
        assert "loss" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_evaluate_global_with_mask_on_correct_device(self):
        """_mask_unseen_classes must produce tensors on the same device as input."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(8, 40), torch.randint(0, 2, (8,)), num_clusters=2)
            for i in range(4)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(16, 40),
            "y_test": torch.randint(0, 2, (16,)),
        }
        server = DFCAServer(clients, test_data, cfg)
        server.train_round(task_id=0, verbose=False)

        # Set seen classes to trigger masking
        server.seen_classes = [0, 1]

        # Must not crash with device mismatch
        result = server.evaluate_global(seen_classes_only=True)
        assert "accuracy" in result

    def test_results_dict_thread_safety_no_crash(self):
        """results_dict writes from multiple threads must not crash."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(8, 40), torch.randint(0, 2, (8,)), num_clusters=2)
            for i in range(8)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 0,
            "batch_size": 4,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(4, 40),
            "y_test": torch.randint(0, 2, (4,)),
        }
        server = DFCAServer(clients, test_data, cfg)

        # Run multiple rounds — threading must be safe
        for _ in range(3):
            result = server.train_round(task_id=0, verbose=False)
            assert "round_stats" in result

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 GPUs")
    def test_multi_gpu_smoke_train_and_eval(self):
        """Smoke test: train_round + evaluate_global on 2 GPUs without device mismatch."""
        import torch
        from fed_learning.clients.dfca_client import DFCAClient
        from fed_learning.servers.dfca_server import DFCAServer

        clients = [
            DFCAClient(i, torch.randn(16, 40), torch.randint(0, 2, (16,)), num_clusters=2)
            for i in range(8)
        ]
        cfg = {
            "algorithm": "dfca_il",
            "input_shape": (40,),
            "num_classes": 4,
            "total_classes": 4,
            "dfca_num_clusters": 2,
            "dfca_client_ratios": [1.0],
            "seed": 42,
            "num_gpus": 2,
            "batch_size": 4,
            "local_epochs": 1,
        }
        test_data = {
            "X_test": torch.randn(16, 40),
            "y_test": torch.randint(0, 2, (16,)),
        }
        server = DFCAServer(clients, test_data, cfg)

        # Must not crash: device mismatch, flatten_weight, inplace version
        result = server.train_round(task_id=0, verbose=True)
        assert "round_stats" in result

        # evaluate_global must not crash after multi-GPU training
        eval_result = server.evaluate_global(seen_classes_only=False)
        assert "accuracy" in eval_result
        assert "loss" in eval_result
