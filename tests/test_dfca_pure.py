"""
Tests for Pure DFCA (Decentralized Federated Clustering Algorithm).

Tests cover:
- Initialization (DFCA-GI and DFCA-LI)
- Cluster assignment (Step 1 of Algorithm 1)
- Local update (Step 2 of Algorithm 1)
- Sequential running average aggregation (Step 3 of Algorithm 1)
- Graph building (Erdos-Renyi)
- Pure runner smoke test
- No IL concepts in pure DFCA code
- Evaluation functions
"""

import os
import random
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from fed_learning.dfca import (
    DFCANode,
    DFCAAggregator,
    build_erdos_renyi_graph,
    build_graph_summary,
    run_dfca_training,
    evaluate_ensemble_average,
    evaluate_representative_clusters,
    evaluate_oracle,
)


# =============================================================================
# Helpers
# =============================================================================

class SimpleLinearModel(nn.Module):
    """Minimal linear model for fast unit tests."""
    def __init__(self, input_dim=10, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.input_shape = input_dim
        self.num_classes = num_classes

    def forward(self, x):
        return self.fc(x)


# =============================================================================
# Test A: Initialization
# =============================================================================

class TestDFCAInitialization:
    """Test DFCA-GI and DFCA-LI initialization."""

    def test_dfca_gi_all_clients_identical_params(self):
        """DFCA-GI: all clients have identical params for same cluster at start."""
        model_template = SimpleLinearModel(input_dim=4, num_classes=3)
        global_params = OrderedDict((k, v.clone()) for k, v in model_template.state_dict().items())

        nodes = []
        for i in range(3):
            node = DFCANode(
                client_id=i,
                X_train=torch.randn(10, 4),
                y_train=torch.randint(0, 3, (10,)),
                num_clusters=3,
                num_classes=3,
            )
            node.initialize_cluster_bank(global_params=global_params, init_type="global")
            nodes.append(node)

        # All clients should have identical params for all clusters
        for cid in range(3):
            for i in range(3):
                for j in range(i + 1, 3):
                    for key in nodes[i].cluster_params[cid]:
                        assert nodes[i].cluster_params[cid][key].equal(
                            nodes[j].cluster_params[cid][key]
                        ), f"DFCA-GI: cluster {cid} params differ between clients"

    def test_dfca_li_params_differ_across_clients(self):
        """DFCA-LI: at least some client/cluster params differ."""
        model_template = SimpleLinearModel(input_dim=4, num_classes=3)

        nodes = []
        for i in range(3):
            node = DFCANode(
                client_id=i,
                X_train=torch.randn(10, 4),
                y_train=torch.randint(0, 3, (10,)),
                num_clusters=3,
                num_classes=3,
                init_seed=42,
            )
            node.initialize_cluster_bank(template_model=model_template, init_type="local")
            nodes.append(node)

        # At least one pair should differ
        found_difference = False
        for i in range(3):
            for j in range(i + 1, 3):
                for cid in range(3):
                    for key in nodes[i].cluster_params[cid]:
                        if not nodes[i].cluster_params[cid][key].equal(
                            nodes[j].cluster_params[cid][key]
                        ):
                            found_difference = True
                            break
                    if found_difference:
                        break
                if found_difference:
                    break
            if found_difference:
                break
        assert found_difference, "DFCA-LI should produce different params across clients"

    def test_each_client_has_exactly_k_clusters(self):
        """Each client has exactly k cluster models."""
        for k in [2, 5, 10]:
            model_template = SimpleLinearModel(input_dim=4, num_classes=3)
            global_params = OrderedDict((k, v.clone()) for k, v in model_template.state_dict().items())

            node = DFCANode(
                client_id=0,
                X_train=torch.randn(10, 4),
                y_train=torch.randint(0, 3, (10,)),
                num_clusters=k,
                num_classes=3,
            )
            node.initialize_cluster_bank(global_params=global_params, init_type="global")
            assert len(node.cluster_params) == k
            for cid in range(k):
                assert cid in node.cluster_params


# =============================================================================
# Test B: Cluster Assignment
# =============================================================================

class TestDFCAClusterAssignment:
    """Test Step 1: assign_cluster."""

    def test_assign_cluster_evaluates_all_k_models(self):
        """assign_cluster must evaluate loss for all k cluster models."""
        model_template = SimpleLinearModel(input_dim=4, num_classes=3)
        global_params = OrderedDict((k, v.clone()) for k, v in model_template.state_dict().items())

        node = DFCANode(
            client_id=0,
            X_train=torch.randn(20, 4),
            y_train=torch.randint(0, 3, (20,)),
            num_clusters=3,
            num_classes=3,
        )
        node.initialize_cluster_bank(global_params=global_params, init_type="global")
        # Make cluster 0 much worse (very large weights)
        for key in node.cluster_params[0]:
            node.cluster_params[0][key].fill_(100.0)

        assigned, losses, margin = node.assign_cluster(
            model_template, device="cpu", verbose=False
        )

        assert len(losses) == 3
        for cid in range(3):
            assert cid in losses
            assert losses[cid] >= 0

    def test_assign_cluster_is_argmin_loss(self):
        """assigned_cluster must be argmin of evaluated losses."""
        model_template = SimpleLinearModel(input_dim=4, num_classes=3)
        global_params = OrderedDict((k, v.clone()) for k, v in model_template.state_dict().items())

        node = DFCANode(
            client_id=0,
            X_train=torch.randn(20, 4),
            y_train=torch.randint(0, 3, (20,)),
            num_clusters=3,
            num_classes=3,
        )
        node.initialize_cluster_bank(global_params=global_params, init_type="global")
        # Make cluster 2 have zero weights (best for identity-like data)
        for key in node.cluster_params[2]:
            node.cluster_params[2][key].fill_(0.0)

        assigned, losses, margin = node.assign_cluster(
            model_template, device="cpu", verbose=False
        )

        assert assigned == 2, f"Expected cluster 2 (best), got {assigned}"

    def test_no_il_fields_in_dfca_node(self):
        """DFCANode must not have task_id, seen_classes, new_classes, classes_per_task."""
        node = DFCANode(
            client_id=0,
            X_train=torch.randn(10, 4),
            y_train=torch.randint(0, 3, (10,)),
            num_clusters=3,
            num_classes=3,
        )
        attrs = dir(node)
        il_terms = ["task_id", "seen_classes", "new_classes", "classes_per_task",
                    "current_task", "_seen_classes", "avg_forgetting"]
        found = [t for t in il_terms if t in attrs]
        assert not found, f"DFCANode should not have IL fields: {found}"


# =============================================================================
# Test C: Local Update
# =============================================================================

class TestDFCALocalUpdate:
    """Test Step 2: train_assigned_cluster."""

    def test_only_assigned_cluster_changes(self):
        """After training, only assigned cluster params should change."""
        model_template = SimpleLinearModel(input_dim=4, num_classes=3)
        global_params = OrderedDict((k, v.clone()) for k, v in model_template.state_dict().items())

        node = DFCANode(
            client_id=0,
            X_train=torch.randn(50, 4),
            y_train=torch.randint(0, 3, (50,)),
            num_clusters=3,
            num_classes=3,
        )
        node.initialize_cluster_bank(global_params=global_params, init_type="global")
        node.assigned_cluster = 1

        before = {
            cid: {k: v.clone() for k, v in node.cluster_params[cid].items()}
            for cid in range(3)
        }

        result = node.train_assigned_cluster(
            model_template, device="cpu",
            epochs=1, batch_size=50, lr=0.1, verbose=False
        )

        for cid in range(3):
            for k in node.cluster_params[cid]:
                changed = not torch.allclose(
                    node.cluster_params[cid][k], before[cid][k]
                )
                if cid == 1:
                    assert changed, f"Cluster {cid} params should have changed"
                else:
                    assert not changed, f"Cluster {cid} params should NOT have changed"

    def test_train_returns_correct_structure(self):
        """train_assigned_cluster returns expected dict structure."""
        model_template = SimpleLinearModel(input_dim=4, num_classes=3)
        global_params = OrderedDict((k, v.clone()) for k, v in model_template.state_dict().items())

        node = DFCANode(
            client_id=0,
            X_train=torch.randn(50, 4),
            y_train=torch.randint(0, 3, (50,)),
            num_clusters=3,
            num_classes=3,
        )
        node.initialize_cluster_bank(global_params=global_params, init_type="global")
        node.assigned_cluster = 1

        result = node.train_assigned_cluster(
            model_template, device="cpu",
            epochs=1, batch_size=50, lr=0.1, verbose=False
        )

        assert "client_id" in result
        assert "assigned_cluster" in result
        assert result["assigned_cluster"] == 1
        assert "loss" in result
        assert "params" in result


# =============================================================================
# Test D: Sequential Aggregation
# =============================================================================

class TestDFCASequentialAggregation:
    """Test Step 3: sequential running average aggregation."""

    def test_scalar_running_average_formula(self):
        """Test the sequential running average formula."""
        local = torch.tensor(10.0)
        neighbor1 = torch.tensor(0.0)
        neighbor2 = torch.tensor(0.0)

        result = DFCAAggregator.sequential_running_average(local, neighbor1, count=0)
        assert torch.allclose(result, torch.tensor(5.0))

        result = DFCAAggregator.sequential_running_average(result, neighbor2, count=1)
        assert torch.allclose(result, torch.tensor(10.0 / 3.0), atol=1e-5)

    def test_tensor_running_average(self):
        """Running average on tensor values."""
        local = torch.ones(5, 5) * 10.0
        neighbor = torch.zeros(5, 5)

        result = DFCAAggregator.sequential_running_average(local, neighbor, count=0)
        expected = 0.5 * local + 0.5 * neighbor
        assert torch.allclose(result, expected)

    def test_equal_weight_after_multiple_neighbors(self):
        """After N neighbors, all N+1 models have equal weight."""
        theta = torch.tensor(100.0)
        neighbors = [torch.tensor(0.0) for _ in range(3)]

        for r, nbr in enumerate(neighbors):
            theta = DFCAAggregator.sequential_running_average(theta, nbr, count=r)

        assert abs(theta.item() - 25.0) < 1e-4

    def test_node_aggregate_only_matching_cluster(self):
        """Only cluster matching sender's assignment is updated during aggregation."""
        model_template = SimpleLinearModel(input_dim=4, num_classes=3)
        global_params = OrderedDict((k, v.clone()) for k, v in model_template.state_dict().items())

        node = DFCANode(
            client_id=0,
            X_train=torch.randn(10, 4),
            y_train=torch.randint(0, 3, (10,)),
            num_clusters=3,
            num_classes=3,
        )
        node.initialize_cluster_bank(global_params=global_params, init_type="global")

        # Sender says it's assigned to cluster 2, sends cluster 2 params
        sender_state = OrderedDict((k, v.clone()) for k, v in global_params.items())
        for k in sender_state:
            sender_state[k].fill_(5.0)

        sender_msg = {2: sender_state}
        node.receive_message(sender_id=1, message=sender_msg)

        before_c0 = node.cluster_params[0]["fc.weight"].clone()
        before_c1 = node.cluster_params[1]["fc.weight"].clone()

        update_counts = node.aggregate_received_messages()

        # Cluster 0 and 1 should be unchanged
        assert node.cluster_params[0]["fc.weight"].equal(before_c0)
        assert node.cluster_params[1]["fc.weight"].equal(before_c1)
        # Cluster 2 should be updated
        assert update_counts[2] == 1

    def test_messages_cleared_after_aggregation(self):
        """Neighbor messages should be cleared after aggregation."""
        model_template = SimpleLinearModel(input_dim=4, num_classes=3)
        global_params = OrderedDict((k, v.clone()) for k, v in model_template.state_dict().items())

        node = DFCANode(
            client_id=0,
            X_train=torch.randn(10, 4),
            y_train=torch.randint(0, 3, (10,)),
            num_clusters=3,
            num_classes=3,
        )
        node.initialize_cluster_bank(global_params=global_params, init_type="global")

        msg = {1: OrderedDict((k, v.clone()) for k, v in global_params.items())}
        node.receive_message(sender_id=1, message=msg)
        assert len(node.received_messages) == 1

        node.aggregate_received_messages()
        assert len(node.received_messages) == 0


# =============================================================================
# Test E: Graph
# =============================================================================

class TestDFCAGraph:
    """Test Erdos-Renyi graph building."""

    def test_graph_undirected(self):
        """Graph should be undirected."""
        node_ids = list(range(20))
        neighbors = build_erdos_renyi_graph(node_ids, connectivity=0.3, seed=42)

        for nid in node_ids:
            for nbr in neighbors.get(nid, []):
                assert nid in neighbors.get(nbr, []), (
                    f"Graph is not undirected: {nid} -> {nbr} but {nbr} -> {nid}"
                )

    def test_graph_neighbor_symmetry(self):
        """neighbors[A] contains B iff neighbors[B] contains A."""
        node_ids = list(range(15))
        neighbors = build_erdos_renyi_graph(node_ids, connectivity=0.2, seed=99)

        for nid in node_ids:
            for nbr in neighbors.get(nid, []):
                assert nbr in neighbors.get(nid, [])
                assert nid in neighbors.get(nbr, [])

    def test_no_isolated_nodes(self):
        """With ensure_connectivity=True, no node should be isolated."""
        node_ids = list(range(10))
        neighbors = build_erdos_renyi_graph(
            node_ids, connectivity=0.05, seed=123,
            ensure_connectivity=True
        )

        for nid in node_ids:
            assert len(neighbors.get(nid, [])) > 0, f"Node {nid} is isolated"

    def test_graph_summary(self):
        """build_graph_summary returns expected fields."""
        node_ids = list(range(20))
        neighbors = build_erdos_renyi_graph(node_ids, connectivity=0.2, seed=42)
        summary = build_graph_summary(neighbors, node_ids)

        assert summary["num_nodes"] == 20
        assert summary["num_edges"] >= 0
        assert summary["isolated_count"] == 0
        assert summary["min_degree"] >= 0
        assert summary["avg_degree"] >= 0

    def test_graph_deterministic_with_seed(self):
        """Same seed should produce same graph."""
        def make_graph(seed):
            return build_erdos_renyi_graph(list(range(15)), connectivity=0.2, seed=seed)

        g1 = make_graph(42)
        g2 = make_graph(42)
        g3 = make_graph(99)

        for nid in g1:
            assert g1[nid] == g2[nid], "Same seed should produce same graph"
        assert g1 != g3, "Different seed should produce different graph"


# =============================================================================
# Test F: Pure Runner Smoke
# =============================================================================

class TestDFCAPureRunner:
    """Smoke test for run_dfca_training."""

    def test_runner_small_cpu_smoke(self):
        """run_dfca_training smoke test with 4 clients, k=2, 2 rounds."""
        torch.manual_seed(42)

        node_ids = [0, 1, 2, 3]
        node_data = {}
        for cid in node_ids:
            X = torch.randn(20, 4)
            y = torch.randint(0, 3, (20,))
            node_data[cid] = (X, y)

        model_template = SimpleLinearModel(input_dim=4, num_classes=3)

        config = {
            "num_rounds": 2,
            "dfca_num_clusters": 2,
            "dfca_init": "global",
            "dfca_graph": "erdos_renyi",
            "dfca_connectivity": 0.3,
            "dfca_participation_rate": 1.0,
            "local_epochs": 1,
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "batch_size": 20,
            "seed": 42,
            "dfca_debug_assignments": False,
            "dfca_debug_messages": False,
            "dfca_debug_cluster_models": False,
            "num_gpus": 0,
            "eval_every": 10,
        }

        result = run_dfca_training(
            node_ids=node_ids,
            node_data=node_data,
            model_template=model_template,
            config=config,
            verbose=False,
        )

        assert "history" in result
        assert "final_assignments" in result
        assert "cluster_history" in result
        assert "graph_summary" in result
        assert "representative_params" in result

        assert len(result["history"]["round"]) == 2
        assert len(result["final_assignments"]) == 4

    def test_runner_with_test_data_evaluation(self):
        """run_dfca_training with test_data evaluates ensemble."""
        torch.manual_seed(42)

        node_ids = [0, 1, 2]
        node_data = {}
        for cid in node_ids:
            X = torch.randn(30, 4)
            y = torch.randint(0, 3, (30,))
            node_data[cid] = (X, y)

        X_test = torch.randn(20, 4)
        y_test = torch.randint(0, 3, (20,))
        test_data = {"X_test": X_test, "y_test": y_test}

        model_template = SimpleLinearModel(input_dim=4, num_classes=3)

        config = {
            "num_rounds": 2,
            "dfca_num_clusters": 2,
            "dfca_init": "global",
            "dfca_graph": "erdos_renyi",
            "dfca_connectivity": 0.5,
            "dfca_participation_rate": 1.0,
            "local_epochs": 1,
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "batch_size": 20,
            "seed": 42,
            "dfca_debug_assignments": False,
            "dfca_debug_messages": False,
            "dfca_debug_cluster_models": False,
            "num_gpus": 0,
            "eval_every": 1,
        }

        result = run_dfca_training(
            node_ids=node_ids,
            node_data=node_data,
            model_template=model_template,
            config=config,
            test_data=test_data,
            verbose=False,
        )

        assert "history" in result
        assert result["history"]["test_accuracy"][0] is not None
        assert result["history"]["test_accuracy"][1] is not None

    def test_runner_participation_rate(self):
        """With participation_rate < 1.0, not all nodes participate every round."""
        torch.manual_seed(42)

        node_ids = list(range(10))
        node_data = {}
        for cid in node_ids:
            X = torch.randn(10, 4)
            y = torch.randint(0, 2, (10,))
            node_data[cid] = (X, y)

        model_template = SimpleLinearModel(input_dim=4, num_classes=2)

        config = {
            "num_rounds": 5,
            "dfca_num_clusters": 2,
            "dfca_init": "global",
            "dfca_graph": "erdos_renyi",
            "dfca_connectivity": 0.3,
            "dfca_participation_rate": 0.5,
            "local_epochs": 1,
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "batch_size": 10,
            "seed": 42,
            "dfca_debug_assignments": False,
            "dfca_debug_messages": False,
            "dfca_debug_cluster_models": False,
            "num_gpus": 0,
            "eval_every": 10,
        }

        result = run_dfca_training(
            node_ids=node_ids,
            node_data=node_data,
            model_template=model_template,
            config=config,
            verbose=False,
        )

        participating = result["history"]["participating_nodes"]
        assert not all(p == 10 for p in participating)


# =============================================================================
# Test G: No IL Leakage
# =============================================================================

class TestDFCANoILLeakage:
    """Verify pure DFCA code does not use IL concepts."""

    def test_dfca_aggregator_has_raise_aggregate(self):
        """DFCAAggregator.aggregate() must raise — no central aggregation."""
        agg = DFCAAggregator(num_clusters=3)
        with pytest.raises(RuntimeError, match="should never be called"):
            agg.aggregate(results=[], global_params=None)

    def test_runner_signature_no_task_fields(self):
        """run_dfca_training must not have task_id, seen_classes, etc. in its signature."""
        import inspect
        from fed_learning.dfca.runner import run_dfca_training

        sig = inspect.signature(run_dfca_training)
        params = set(sig.parameters.keys())

        il_params = {"task_id", "task_start", "task_end", "seen_classes",
                     "new_classes", "classes_per_task", "avg_forgetting"}
        found = params & il_params
        assert not found, f"run_dfca_training should not have IL params: {found}"

    def test_runner_config_no_il_keys(self):
        """run_dfca_training code must not reference IL config keys."""
        import inspect
        from fed_learning.dfca import runner

        source = inspect.getsource(runner)
        il_terms = ["task_id", "seen_classes", "new_classes", "classes_per_task",
                     "avg_forgetting", "post_task_processing"]
        for term in il_terms:
            # Count occurrences that are NOT in comments or string literals
            in_code = 0
            for line in source.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                    continue
                if term in line and f'"{term}"' not in line and f"'{term}'" not in line:
                    in_code += 1
            assert in_code == 0, f"IL term '{term}' appears {in_code} times in runner code"

    def test_evaluation_no_global_model(self):
        """evaluate_ensemble_average must not use server.global_model."""
        import inspect
        from fed_learning.dfca.evaluation import evaluate_ensemble_average

        source = inspect.getsource(evaluate_ensemble_average)
        assert "global_model" not in source


# =============================================================================
# Test H: Evaluation Functions
# =============================================================================

class TestDFCAEvaluation:
    """Test evaluation functions."""

    def test_evaluate_ensemble_average_returns_metrics(self):
        """evaluate_ensemble_average returns accuracy/loss/f1."""
        model_template = SimpleLinearModel(input_dim=4, num_classes=3)
        rep_params = {
            0: OrderedDict((k, v.clone()) for k, v in model_template.state_dict().items()),
            1: OrderedDict((k, v.clone()) for k, v in model_template.state_dict().items()),
        }

        X_test = torch.randn(20, 4)
        y_test = torch.randint(0, 3, (20,))

        result = evaluate_ensemble_average(
            nodes={},
            representative_params=rep_params,
            X_test=X_test,
            y_test=y_test,
            model_class=SimpleLinearModel,
            input_shape=4,
            num_classes=3,
            device="cpu",
            batch_size=20,
        )

        assert "accuracy" in result
        assert "loss" in result
        assert "f1_macro" in result
        assert "f1_weighted" in result
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_evaluate_representative_clusters_returns_per_cluster(self):
        """evaluate_representative_clusters returns per-cluster metrics."""
        model_template = SimpleLinearModel(input_dim=4, num_classes=3)
        rep_params = {
            0: OrderedDict((k, v.clone()) for k, v in model_template.state_dict().items()),
        }
        X_test = torch.randn(10, 4)
        y_test = torch.randint(0, 3, (10,))

        result = evaluate_representative_clusters(
            representative_params=rep_params,
            X_test=X_test,
            y_test=y_test,
            model_class=SimpleLinearModel,
            input_shape=4,
            num_classes=3,
            device="cpu",
            batch_size=10,
        )

        assert "per_cluster_metrics" in result
        assert "best_cluster_by_loss" in result
        assert "best_cluster_by_accuracy" in result

    def test_evaluate_oracle_returns_diagnostic(self):
        """evaluate_oracle returns oracle metrics (diagnostic only)."""
        model_template = SimpleLinearModel(input_dim=4, num_classes=3)
        rep_params = {
            0: OrderedDict((k, v.clone()) for k, v in model_template.state_dict().items()),
        }
        X_test = torch.randn(10, 4)
        y_test = torch.randint(0, 3, (10,))

        result = evaluate_oracle(
            representative_params=rep_params,
            X_test=X_test,
            y_test=y_test,
            model_class=SimpleLinearModel,
            input_shape=4,
            num_classes=3,
            device="cpu",
            batch_size=10,
        )

        assert "oracle_accuracy" in result
        assert "oracle_loss" in result


# =============================================================================
# Test I: New tests for fixes
# =============================================================================

class TestDFCANewFixes:
    """Tests for bugs fixed in this session."""

    def test_dfca_init_local_creates_k_clusters(self):
        """dfca_init='local' must create k cluster models without crashing."""
        model_template = SimpleLinearModel(input_dim=4, num_classes=3)

        node = DFCANode(
            client_id=0,
            X_train=torch.randn(10, 4),
            y_train=torch.randint(0, 3, (10,)),
            num_clusters=5,
            num_classes=3,
            init_seed=42,
        )
        # This must NOT raise KeyError or any other error
        node.initialize_cluster_bank(template_model=model_template, init_type="local")
        assert len(node.cluster_params) == 5
        for cid in range(5):
            assert cid in node.cluster_params

    def test_dfca_init_global_creates_k_clusters(self):
        """dfca_init='global' must create k identical cluster models."""
        model_template = SimpleLinearModel(input_dim=4, num_classes=3)
        global_params = OrderedDict((k, v.clone()) for k, v in model_template.state_dict().items())

        node = DFCANode(
            client_id=0,
            X_train=torch.randn(10, 4),
            y_train=torch.randint(0, 3, (10,)),
            num_clusters=5,
            num_classes=3,
        )
        node.initialize_cluster_bank(global_params=global_params, init_type="global")
        assert len(node.cluster_params) == 5

    def test_num_messages_equals_total_deliveries(self):
        """num_messages must equal total deliveries (senders × their participating neighbors)."""
        from fed_learning.dfca.runner import _run_message_passing

        torch.manual_seed(0)

        # 3 nodes: 0->[1], 1->[0,2], 2->[1]
        # When all participate:
        #   node 0 sends to node 1 (1 delivery)
        #   node 1 sends to nodes 0 and 2 (2 deliveries)
        #   node 2 sends to node 1 (1 delivery)
        #   total = 4
        nodes = {}
        for nid in [0, 1, 2]:
            node = DFCANode(
                client_id=nid,
                X_train=torch.randn(10, 4),
                y_train=torch.randint(0, 3, (10,)),
                num_clusters=2,
                num_classes=3,
                init_seed=42,
            )
            global_params = OrderedDict(
                (k, v.clone()) for k, v in SimpleLinearModel(4, 3).state_dict().items()
            )
            node.initialize_cluster_bank(global_params=global_params, init_type="global")
            node.assigned_cluster = nid % 2  # distribute across clusters
            nodes[nid] = node

        neighbors = {0: [1], 1: [0, 2], 2: [1]}

        num_messages, per_cluster_updates, _, _ = _run_message_passing(
            nodes, participating_ids=[0, 1, 2], neighbors=neighbors,
            num_clusters=2, debug_messages=False, debug_message_limit=0,
        )

        # Total deliveries
        expected = sum(len([m for m in [0, 1, 2] if nid in neighbors.get(m, [])]) for nid in [0, 1, 2])
        assert num_messages == expected, f"Expected {expected}, got {num_messages}"

    def test_num_messages_equals_sum_per_cluster_updates(self):
        """num_messages must equal sum of all per_cluster_updates values."""
        from fed_learning.dfca.runner import _run_message_passing

        torch.manual_seed(0)

        nodes = {}
        for nid in range(3):
            node = DFCANode(
                client_id=nid,
                X_train=torch.randn(10, 4),
                y_train=torch.randint(0, 3, (10,)),
                num_clusters=2,
                num_classes=3,
                init_seed=42,
            )
            global_params = OrderedDict(
                (k, v.clone()) for k, v in SimpleLinearModel(4, 3).state_dict().items()
            )
            node.initialize_cluster_bank(global_params=global_params, init_type="global")
            node.assigned_cluster = 0
            nodes[nid] = node

        neighbors = {0: [1, 2], 1: [0], 2: [0]}

        num_messages, per_cluster_updates, _, _ = _run_message_passing(
            nodes, participating_ids=[0, 1, 2], neighbors=neighbors,
            num_clusters=2, debug_messages=False, debug_message_limit=0,
        )

        assert num_messages == sum(per_cluster_updates.values()), (
            f"num_messages={num_messages} != sum(per_cluster_updates)={sum(per_cluster_updates.values())}"
        )

    def test_eval_every_1_all_rounds_have_metrics(self):
        """With eval_every=1, every round must have test metrics."""
        config = {
            "num_rounds": 5,
            "dfca_num_clusters": 2,
            "dfca_init": "global",
            "dfca_graph": "erdos_renyi",
            "dfca_connectivity": 0.3,
            "dfca_participation_rate": 1.0,
            "local_epochs": 1,
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "batch_size": 10,
            "seed": 42,
            "dfca_debug_assignments": False,
            "dfca_debug_messages": False,
            "dfca_debug_cluster_models": False,
            "num_gpus": 0,
            "eval_every": 1,
        }

        node_ids = [0, 1]
        node_data = {}
        for cid in node_ids:
            X = torch.randn(10, 4)
            y = torch.randint(0, 3, (10,))
            node_data[cid] = (X, y)

        model_template = SimpleLinearModel(input_dim=4, num_classes=3)
        X_test = torch.randn(8, 4)
        y_test = torch.randint(0, 3, (8,))
        test_data = {"X_test": X_test, "y_test": y_test}

        result = run_dfca_training(
            node_ids=node_ids,
            node_data=node_data,
            model_template=model_template,
            config=config,
            test_data=test_data,
            verbose=False,
        )

        assert len(result["history"]["test_accuracy"]) == 5
        for acc in result["history"]["test_accuracy"]:
            assert acc is not None

    def test_eval_every_2_metrics_aligned_with_rounds(self):
        """With eval_every=2, test metric arrays have same length as rounds with None for non-eval rounds."""
        config = {
            "num_rounds": 6,
            "dfca_num_clusters": 2,
            "dfca_init": "global",
            "dfca_graph": "erdos_renyi",
            "dfca_connectivity": 0.3,
            "dfca_participation_rate": 1.0,
            "local_epochs": 1,
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "batch_size": 10,
            "seed": 42,
            "dfca_debug_assignments": False,
            "dfca_debug_messages": False,
            "dfca_debug_cluster_models": False,
            "num_gpus": 0,
            "eval_every": 2,
        }

        node_ids = [0, 1]
        node_data = {}
        for cid in node_ids:
            X = torch.randn(10, 4)
            y = torch.randint(0, 3, (10,))
            node_data[cid] = (X, y)

        model_template = SimpleLinearModel(input_dim=4, num_classes=3)
        X_test = torch.randn(8, 4)
        y_test = torch.randint(0, 3, (8,))
        test_data = {"X_test": X_test, "y_test": y_test}

        result = run_dfca_training(
            node_ids=node_ids,
            node_data=node_data,
            model_template=model_template,
            config=config,
            test_data=test_data,
            verbose=False,
        )

        # All arrays must have same length as rounds
        for key in ["test_loss", "test_accuracy", "test_f1_macro"]:
            arr = result["history"][key]
            assert len(arr) == len(result["history"]["round"]), (
                f"{key} length {len(arr)} != rounds length {len(result['history']['round'])}"
            )

        # Rounds 1, 3, 5 have metrics (eval_every=2, first eval at round 1)
        assert result["history"]["test_accuracy"][1] is not None
        assert result["history"]["test_accuracy"][0] is None
        assert result["history"]["test_accuracy"][3] is not None

    def test_runner_start_round_offset(self):
        """start_round must offset the round indices correctly."""
        config = {
            "num_rounds": 3,
            "dfca_num_clusters": 2,
            "dfca_init": "global",
            "dfca_graph": "erdos_renyi",
            "dfca_connectivity": 0.3,
            "dfca_participation_rate": 1.0,
            "local_epochs": 1,
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "batch_size": 10,
            "seed": 42,
            "dfca_debug_assignments": False,
            "dfca_debug_messages": False,
            "dfca_debug_cluster_models": False,
            "num_gpus": 0,
            "eval_every": 10,
        }

        node_ids = [0, 1]
        node_data = {}
        for cid in node_ids:
            X = torch.randn(10, 4)
            y = torch.randint(0, 3, (10,))
            node_data[cid] = (X, y)

        model_template = SimpleLinearModel(input_dim=4, num_classes=3)

        result = run_dfca_training(
            node_ids=node_ids,
            node_data=node_data,
            model_template=model_template,
            config=config,
            verbose=False,
            start_round=5,
            num_rounds_override=3,
        )

        assert result["history"]["round"] == [5, 6, 7]
        assert len(result["history"]["round"]) == 3

    def test_runner_with_test_data_no_task_fields(self):
        """run_dfca_training must not accept task_id, task_start, task_end kwargs."""
        import inspect
        from fed_learning.dfca.runner import run_dfca_training

        sig = inspect.signature(run_dfca_training)
        params = set(sig.parameters.keys())
        il_params = {"task_id", "task_start", "task_end", "seen_classes",
                     "new_classes", "classes_per_task", "avg_forgetting"}
        found = params & il_params
        assert not found, f"run_dfca_training should not have IL params: {found}"


class TestIncrementalDataLoaderFullData:
    """Test pure FL full-data loading methods."""

    def test_get_client_full_data_returns_all_samples(self):
        """get_client_full_data must return all samples without task filtering."""
        # This test verifies the method signature exists and returns correct type
        from fed_learning.data.incremental_loader import IncrementalDataLoader

        # Method must exist
        assert hasattr(IncrementalDataLoader, "get_client_full_data")
        assert hasattr(IncrementalDataLoader, "get_full_test_data")

        # get_client_full_data must have the right signature
        import inspect
        sig = inspect.signature(IncrementalDataLoader.get_client_full_data)
        params = list(sig.parameters.keys())
        # self, client_id
        assert params == ["self", "client_id"]

        # get_full_test_data must have the right signature
        sig2 = inspect.signature(IncrementalDataLoader.get_full_test_data)
        params2 = list(sig2.parameters.keys())
        assert params2 == ["self"]

    def test_get_client_full_data_no_task_arg(self):
        """get_client_full_data must NOT have a task_id parameter."""
        import inspect
        from fed_learning.data.incremental_loader import IncrementalDataLoader

        sig = inspect.signature(IncrementalDataLoader.get_client_full_data)
        params = set(sig.parameters.keys())
        task_params = {"task_id", "task_start", "task_end", "cumulative"}
        found = params & task_params
        assert not found, f"get_client_full_data should not have task params: {found}"


# =============================================================================
# Test J: Checkpoint / Resume
# =============================================================================

class TestDFCACheckpointResume:
    """Test checkpoint save/load and resume functionality."""

    def test_checkpoint_state_contains_cluster_banks(self, tmp_path):
        """build_dfca_checkpoint must include nodes_state with cluster params."""
        from fed_learning.dfca import build_dfca_checkpoint

        torch.manual_seed(42)
        nodes = {}
        for nid in [0, 1]:
            node = DFCANode(
                client_id=nid,
                X_train=torch.randn(10, 4),
                y_train=torch.randint(0, 3, (10,)),
                num_clusters=2,
                num_classes=3,
                init_seed=42 + nid,
            )
            global_params = OrderedDict(
                (k, v.clone()) for k, v in SimpleLinearModel(4, 3).state_dict().items()
            )
            node.initialize_cluster_bank(global_params=global_params, init_type="global")
            node.assigned_cluster = nid % 2
            nodes[nid] = node

        graph_neighbors = {0: [1], 1: [0]}
        prev_assign = {0: 0, 1: 1}
        history = _make_hist()
        cluster_history = []
        rep_params = {}
        config = {"dfca_num_clusters": 2, "seed": 42}

        ckpt = build_dfca_checkpoint(
            nodes=nodes,
            graph_neighbors=graph_neighbors,
            prev_assignments=prev_assign,
            history=history,
            cluster_history=cluster_history,
            representative_params=rep_params,
            config=config,
            current_round=5,
            num_rounds=10,
        )

        assert "nodes_state" in ckpt
        assert "graph_neighbors" in ckpt
        assert "prev_assignments" in ckpt
        assert "rng_state" in ckpt
        assert "current_round" in ckpt
        assert ckpt["current_round"] == 5
        # nodes_state keyed by int (not str)
        assert 0 in ckpt["nodes_state"]
        assert 1 in ckpt["nodes_state"]
        assert 0 in ckpt["nodes_state"][0]   # cluster 0
        assert 1 in ckpt["nodes_state"][0]   # cluster 1 (k=2)

    def test_resume_continues_from_saved_cluster_banks(self):
        """After resuming, cluster banks must continue from saved state (not re-initialized)."""
        torch.manual_seed(99)
        config = {
            "num_rounds": 1,
            "dfca_num_clusters": 2,
            "dfca_init": "global",
            "dfca_graph": "erdos_renyi",
            "dfca_connectivity": 0.5,
            "dfca_participation_rate": 1.0,
            "local_epochs": 1,
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "batch_size": 10,
            "seed": 99,
            "dfca_debug_assignments": False,
            "dfca_debug_messages": False,
            "dfca_debug_cluster_models": False,
            "num_gpus": 0,
            "eval_every": 10,
        }

        node_ids = [0, 1]
        node_data = {}
        for cid in node_ids:
            X = torch.randn(10, 4)
            y = torch.randint(0, 3, (10,))
            node_data[cid] = (X, y)

        model_template = SimpleLinearModel(input_dim=4, num_classes=3)

        # Run 1 round
        result1 = run_dfca_training(
            node_ids=node_ids,
            node_data=node_data,
            model_template=model_template,
            config=config,
            verbose=False,
        )

        ckpt = result1["checkpoint_state"]
        assert ckpt["nodes_state"], "checkpoint_state must contain cluster banks"

        # Resume and run 1 more round
        resume_config = dict(config)
        resume_config["num_rounds"] = 2
        result2 = run_dfca_training(
            node_ids=node_ids,
            node_data=node_data,
            model_template=model_template,
            config=resume_config,
            verbose=False,
            resume_state=ckpt,
        )

        # History from round 0 is preserved and new training continues at round 1.
        assert result2["history"]["round"] == [0, 1]
        assert result2["checkpoint_state"]["nodes_state"], "resumed checkpoint must remain resumable"

    def test_find_latest_checkpoint_helper(self, tmp_path):
        """find_latest_checkpoint finds checkpoint in base_dir and subdirs."""
        from fed_learning.dfca import find_latest_checkpoint

        # Create fake checkpoint files
        base = str(tmp_path)
        open(os.path.join(base, "checkpoint_round_2.pt"), "w").close()
        os.makedirs(os.path.join(base, "results_dfca_20250101"))
        open(os.path.join(base, "results_dfca_20250101", "checkpoint_round_5.pt"), "w").close()
        sibling = f"{base}_20250102"
        os.makedirs(sibling)
        open(os.path.join(sibling, "checkpoint_round_8.pt"), "w").close()

        path = find_latest_checkpoint(base)
        assert path is not None
        assert "checkpoint_round_8.pt" in path

    def test_message_log_records_recipients_and_received_clusters(self):
        """With a known graph, msg_log must show recipients and received_from."""
        from fed_learning.dfca.runner import _run_message_passing

        nodes = {}
        for nid in [0, 1, 2]:
            node = DFCANode(
                client_id=nid,
                X_train=torch.randn(10, 4),
                y_train=torch.randint(0, 3, (10,)),
                num_clusters=2,
                num_classes=3,
                init_seed=42,
            )
            global_params = OrderedDict(
                (k, v.clone()) for k, v in SimpleLinearModel(4, 3).state_dict().items()
            )
            node.initialize_cluster_bank(global_params=global_params, init_type="global")
            node.assigned_cluster = nid % 2
            nodes[nid] = node

        # Fully connected graph among 3 nodes
        neighbors = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

        num_messages, _, delivery_log, received_from = _run_message_passing(
            nodes, participating_ids=[0, 1, 2], neighbors=neighbors,
            num_clusters=2, debug_messages=False, debug_message_limit=0,
        )

        # delivery_log must have entries for all nodes
        assert len(delivery_log) == 3
        for nid in [0, 1, 2]:
            key = f"node_{nid}"
            assert key in delivery_log
            assert delivery_log[key]["delivery_count"] == 2  # each node sends to 2 others

        # received_from must be populated (each node receives from 2 neighbors)
        assert len(received_from) == 3
        for nid in [0, 1, 2]:
            assert nid in received_from
            assert len(received_from[nid]) == 2  # received from 2 neighbors

    def test_messages_sent_is_recipient_count(self):
        """messages_sent (delivery_count) must be number of recipients, not number of cluster keys."""
        from fed_learning.dfca.runner import _run_message_passing

        nodes = {}
        for nid in [0, 1]:
            node = DFCANode(
                client_id=nid,
                X_train=torch.randn(10, 4),
                y_train=torch.randint(0, 3, (10,)),
                num_clusters=2,
                num_classes=3,
                init_seed=42,
            )
            global_params = OrderedDict(
                (k, v.clone()) for k, v in SimpleLinearModel(4, 3).state_dict().items()
            )
            node.initialize_cluster_bank(global_params=global_params, init_type="global")
            node.assigned_cluster = 0
            nodes[nid] = node

        # 0 connects to 1; 1 connects to 0
        neighbors = {0: [1], 1: [0]}
        num_messages, _, delivery_log, _ = _run_message_passing(
            nodes, participating_ids=[0, 1], neighbors=neighbors,
            num_clusters=2, debug_messages=False, debug_message_limit=0,
        )

        # Both nodes have 1 recipient each, so delivery_count = 1 for each
        for key, entry in delivery_log.items():
            assert entry["delivery_count"] == len(entry["recipients"]), (
                f"delivery_count ({entry['delivery_count']}) != len(recipients) ({len(entry['recipients'])})"
            )
        # num_messages = total deliveries
        assert num_messages == sum(e["delivery_count"] for e in delivery_log.values())

    def test_completed_checkpoint_does_not_train_extra_round(self):
        """When resume_state current_round >= num_rounds, no training loop runs."""
        torch.manual_seed(55)
        config = {
            "num_rounds": 3,
            "dfca_num_clusters": 2,
            "dfca_init": "global",
            "dfca_graph": "erdos_renyi",
            "dfca_connectivity": 0.5,
            "dfca_participation_rate": 1.0,
            "local_epochs": 1,
            "optimizer": "sgd",
            "learning_rate": 0.01,
            "batch_size": 10,
            "seed": 55,
            "dfca_debug_assignments": False,
            "dfca_debug_messages": False,
            "dfca_debug_cluster_models": False,
            "num_gpus": 0,
            "eval_every": 10,
        }

        node_ids = [0, 1]
        node_data = {}
        for cid in node_ids:
            X = torch.randn(10, 4)
            y = torch.randint(0, 3, (10,))
            node_data[cid] = (X, y)

        model_template = SimpleLinearModel(input_dim=4, num_classes=3)

        # Run 3 rounds
        result1 = run_dfca_training(
            node_ids=node_ids,
            node_data=node_data,
            model_template=model_template,
            config=config,
            verbose=False,
        )
        assert len(result1["history"]["round"]) == 3

        ckpt = result1["checkpoint_state"]

        # Resume with same num_rounds=3: remaining = 3 - (2+1) = 0
        result2 = run_dfca_training(
            node_ids=node_ids,
            node_data=node_data,
            model_template=model_template,
            config=config,
            verbose=False,
            resume_state=ckpt,
        )

        # No new round is appended; the existing history is preserved.
        assert result2["history"]["round"] == [0, 1, 2]


# =============================================================================
# Helpers for tests
# =============================================================================

def _make_hist():
    from fed_learning.dfca.runner import _make_empty_history
    return _make_empty_history()

