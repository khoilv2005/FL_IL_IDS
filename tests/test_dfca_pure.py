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
