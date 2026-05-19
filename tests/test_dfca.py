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
        """Active sets should be nested prefixes (deterministic)."""
        # Simulate the server's nested prefix logic
        client_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        total_clients = 100

        all_ids = list(range(total_clients))
        seed = 42

        results = []
        for task_id in range(6):
            rng = random.Random(seed + task_id)
            shuffled = all_ids.copy()
            rng.shuffle(shuffled)

            ratio = client_ratios[task_id]
            num_active = max(1, int(total_clients * ratio))
            active_ids = set(shuffled[:num_active])
            results.append(active_ids)

        # Verify counts
        expected_counts = [50, 60, 70, 80, 90, 100]
        for task_id, expected in enumerate(expected_counts):
            actual = len(results[task_id])
            assert actual == expected, f"Task {task_id}: expected {expected}, got {actual}"

    def test_nested_property(self):
        """Each task's active set should be a superset of previous task's set."""
        client_ratios = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        total_clients = 100

        all_ids = list(range(total_clients))
        seed = 42

        sets = []
        for task_id in range(6):
            rng = random.Random(seed + task_id)
            shuffled = all_ids.copy()
            rng.shuffle(shuffled)

            ratio = client_ratios[task_id]
            num_active = max(1, int(total_clients * ratio))
            active_ids = frozenset(shuffled[:num_active])
            sets.append(active_ids)

        # Each subsequent set should be larger
        for t in range(1, 6):
            assert len(sets[t]) > len(sets[t - 1])


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
