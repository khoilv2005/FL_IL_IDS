"""
run_dfca_training - Pure DFCA decentralized training runner.

Based on: Dhasade et al., "DFCA: Decentralized Federated Clustering Algorithm"

Algorithm 1 per-round steps:
    Step 1: AssignCluster — each node picks cluster with minimum local loss
    Step 2: LocalUpdate — each node trains only its assigned cluster model
    Step 3: Decentralized Aggregation — peer-to-peer sequential running average

NO incremental learning. NO task loop. NO central server aggregation.
"""

import os
import json
import time
import random
from collections import Counter, OrderedDict
from threading import Thread, Lock
from typing import Dict, List, Any, Optional, Callable

import numpy as np
import torch
import torch.nn as nn

from .client import DFCANode
from .graph import build_erdos_renyi_graph, build_graph_summary
from .evaluation import (
    evaluate_ensemble_average,
    evaluate_representative_clusters,
    evaluate_oracle,
)


# =============================================================================
# Representative model building
# =============================================================================

def _build_representative_models(
    nodes: Dict[int, DFCANode],
    participating_ids: List[int],
    num_clusters: int,
) -> Dict[int, OrderedDict]:
    """
    Build representative cluster models as the average of theta_i,j
    across participating nodes that have cluster j.

    Args:
        nodes: Dict of DFCANode instances.
        participating_ids: List of participating node IDs this round.
        num_clusters: Number of clusters k.

    Returns:
        Dict[cluster_id -> OrderedDict of averaged params].
    """
    contributor_count: Dict[int, int] = {c: 0 for c in range(num_clusters)}
    representative: Dict[int, OrderedDict] = {c: OrderedDict() for c in range(num_clusters)}

    for nid in participating_ids:
        node = nodes[nid]
        if not hasattr(node, "cluster_params") or not node.cluster_params:
            continue
        for cluster_id in range(num_clusters):
            if cluster_id not in node.cluster_params:
                continue
            params = node.cluster_params[cluster_id]
            contributor_count[cluster_id] += 1

            if not representative[cluster_id]:
                representative[cluster_id] = OrderedDict(
                    (k, v.clone().float()) for k, v in params.items()
                )
            else:
                for k in representative[cluster_id]:
                    if k in params:
                        representative[cluster_id][k] += params[k].float()

    for cluster_id in range(num_clusters):
        count = contributor_count[cluster_id]
        if count == 0:
            representative[cluster_id] = OrderedDict()
            continue
        for k in representative[cluster_id]:
            if representative[cluster_id][k].dtype.is_floating_point:
                representative[cluster_id][k] /= count

    return representative


# =============================================================================
# Format helpers
# =============================================================================

def _format_cluster_updates(per_cluster_updates: Dict[int, int]) -> str:
    """Format per-cluster updates dict into compact readable string."""
    if not per_cluster_updates:
        return "none"
    total = sum(per_cluster_updates.values())
    if total == 0:
        parts = [f"c{cid}=0" for cid in sorted(per_cluster_updates.keys())]
        return ", ".join(parts)
    parts = [f"c{cid}={per_cluster_updates.get(cid, 0)}" for cid in sorted(per_cluster_updates.keys())]
    return ", ".join(parts)


def _format_assignment_log(
    assignment_losses: Dict[int, Dict[int, float]]
) -> Dict[str, Any]:
    """Summarize assignment losses across all nodes."""
    if not assignment_losses:
        return {}
    all_losses = list(assignment_losses.values())
    if not all_losses:
        return {}
    flat_losses = []
    for losses in all_losses:
        flat_losses.extend(losses.values())
    return {
        "avg_best_loss": float(np.mean([min(l.values()) for l in all_losses])),
        "avg_margin": float(np.mean([
            sorted(l.values())[1] - sorted(l.values())[0]
            if len(l) > 1 else 0.0
            for l in all_losses
        ])),
    }


def _rep_params_summary(rep_params: Dict[int, OrderedDict]) -> Dict[str, Any]:
    """Summarize representative cluster params for logging (not full tensors)."""
    summary = {}
    for cluster_id, params in rep_params.items():
        if not params:
            summary[f"cluster_{cluster_id}"] = "empty"
            continue
        layer_shapes = {k: list(v.shape) for k, v in params.items()}
        norm = float(np.sqrt(sum((v.float() ** 2).sum().item() for v in params.values())))
        summary[f"cluster_{cluster_id}"] = {
            "num_layers": len(params),
            "layer_shapes": layer_shapes,
            "param_norm": round(norm, 6),
        }
    return summary


# =============================================================================
# Per-GPU worker threads
# =============================================================================

def _worker_assign_and_train(
    node_ids_batch: List[int],
    nodes: Dict[int, DFCANode],
    model_class,
    input_shape: int,
    num_classes: int,
    device: str,
    optimizer: str,
    local_epochs: int,
    lr: float,
    batch_size: int,
    debug_assignments: bool,
    results_assign: Dict[int, Any],
    results_train: Dict[int, Any],
    results_lock: Lock,
):
    """
    Worker function for a GPU batch: assign + train all nodes in this batch.

    Each node gets its own model instance on the correct device.
    """
    model_template = model_class(input_shape, num_classes)

    for nid in node_ids_batch:
        node = nodes[nid]

        # Step 1: Cluster Assignment (calls _ensure_model_on_device internally)
        try:
            assigned, losses, margin = node.assign_cluster(
                model_template=model_template,
                device=device,
                verbose=debug_assignments,
            )
            with results_lock:
                results_assign[nid] = {
                    "assigned": assigned,
                    "losses": losses,
                    "margin": margin,
                }
        except Exception as e:
            with results_lock:
                results_assign[nid] = {
                    "assigned": getattr(node, "assigned_cluster", 0),
                    "losses": {},
                    "margin": 0.0,
                    "error": str(e),
                }

        # Step 2: Local Training (calls _ensure_model_on_device internally)
        try:
            train_result = node.train_assigned_cluster(
                model_template=model_template,
                device=device,
                epochs=local_epochs,
                batch_size=batch_size,
                lr=lr,
                verbose=False,
            )
            with results_lock:
                results_train[nid] = train_result
        except Exception as e:
            with results_lock:
                results_train[nid] = {
                    "client_id": nid,
                    "assigned_cluster": node.assigned_cluster,
                    "loss": 0.0,
                    "error": str(e),
                }


# =============================================================================
# Message passing simulation
# =============================================================================

def _run_message_passing(
    nodes: Dict[int, DFCANode],
    participating_ids: List[int],
    neighbors: Dict[int, List[int]],
    num_clusters: int,
    debug_messages: bool,
    debug_message_limit: int,
) -> tuple[int, Dict[int, int], Dict[int, List[int]]]:
    """
    Simulate peer-to-peer message passing.

    Returns:
        (num_messages, per_cluster_updates, messages dict for logging)
    """
    # Each participating node exports its assigned cluster message
    messages: Dict[int, Dict[int, OrderedDict]] = {}
    for nid in participating_ids:
        node = nodes[nid]
        msg = node.export_assigned_cluster_message()
        messages[nid] = msg
        node.received_messages = {}

    # Debug logging (per-sender, one line per sender)
    logged_count = 0
    limit = debug_message_limit
    unlimited = (limit <= 0)

    if debug_messages:
        for nid in participating_ids:
            if not unlimited and logged_count >= limit:
                remaining = len(participating_ids) - logged_count
                if remaining > 0:
                    print(f"[DFCA][messages] hidden {remaining} additional message logs")
                break

            msg = messages.get(nid, {})
            if not msg:
                print(f"[DFCA][messages] sender=node_{nid} skipped reason=no_assigned_cluster")
                logged_count += 1
                continue

            cluster_ids = list(msg.keys())
            if not cluster_ids:
                print(f"[DFCA][messages] sender=node_{nid} skipped reason=no_assigned_cluster")
                logged_count += 1
                continue

            sender_cluster = cluster_ids[0]

            recipients = [
                m for m in participating_ids
                if nid in neighbors.get(m, [])
            ]

            if recipients:
                rec_str = ", ".join(f"node_{r}" for r in sorted(recipients))
                print(
                    f"[DFCA][messages] sender=node_{nid} cluster={sender_cluster} "
                    f"recipients=[{rec_str}] count={len(recipients)}"
                )
            else:
                print(
                    f"[DFCA][messages] sender=node_{nid} cluster={sender_cluster} "
                    f"recipients=[] count=0"
                )
            logged_count += 1

    # Deliver messages and aggregate
    per_cluster_updates: Dict[int, int] = {c: 0 for c in range(num_clusters)}

    for nid in participating_ids:
        node = nodes[nid]
        node_neighbors = neighbors.get(nid, [])

        for sender_id in node_neighbors:
            if sender_id not in messages:
                continue
            node.receive_message(sender_id, messages[sender_id])

        update_counts = node.aggregate_received_messages()
        for cid, cnt in update_counts.items():
            per_cluster_updates[cid] = per_cluster_updates.get(cid, 0) + cnt

    num_messages = sum(len(messages.get(nid, {})) for nid in participating_ids)
    return num_messages, per_cluster_updates, messages


# =============================================================================
# Main runner
# =============================================================================

def run_dfca_training(
    node_ids: List[int],
    node_data: Dict[int, tuple],
    model_template,
    config: Dict[str, Any],
    test_data: Optional[Dict] = None,
    verbose: bool = True,
    round_callback: Optional[Callable] = None,
) -> Dict:
    """
    Run pure DFCA decentralized training.

    Implements Algorithm 1 from the DFCA paper:
        Step 1: AssignCluster — each node picks cluster with min local loss
        Step 2: LocalUpdate  — each node trains only its assigned cluster model
        Step 3: Decentralized Aggregation — peer-to-peer sequential running average

    Args:
        node_ids: List of client/node IDs.
        node_data: Dict[node_id -> (X_train, y_train)].
        model_template: Pre-created model instance (CNN_GRU_Model).
        config: Training configuration dict.
        test_data: Optional Dict["X_test", "y_test"] for evaluation.
        verbose: Print round-by-round logs.
        round_callback: Optional callback(round_id, history_dict, metrics_dict) -> None.

    Returns:
        Dict with:
            history: round_metrics list
            final_assignments: Dict[node_id -> cluster_id]
            cluster_history: List of per-round statistics
            graph_summary: Graph topology statistics
            representative_params: Final per-cluster representative params
            config: The config used
    """
    # ---- Config ----
    num_rounds = config.get("num_rounds", 150)
    num_clusters = config.get("dfca_num_clusters", 10)
    dfca_init = config.get("dfca_init", "global")
    dfca_graph = config.get("dfca_graph", "erdos_renyi")
    dfca_connectivity = config.get("dfca_connectivity", 0.15)
    participation_rate = config.get("dfca_participation_rate", 1.0)
    optimizer = config.get("optimizer", "sgd")
    local_epochs = config.get("local_epochs", 5)
    lr = config.get("learning_rate", 0.1)
    batch_size = config.get("batch_size", 2048)
    eval_every = config.get("eval_every", 10)
    seed = config.get("seed", 42)
    debug_assignments = config.get("dfca_debug_assignments", True)
    debug_messages = config.get("dfca_debug_messages", False)
    debug_message_limit = config.get("dfca_debug_message_limit", 50)
    debug_cluster_models = config.get("dfca_debug_cluster_models", True)

    num_classes = model_template.num_classes
    input_shape = model_template.input_shape
    model_class = model_template.__class__

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if config.get("num_gpus", 0) > 0:
        num_gpus = config["num_gpus"]

    device_main = "cpu" if num_gpus == 0 else "cuda:0"
    rng = random.Random(seed)

    # ---- Build Graph ----
    if verbose:
        print(f"\n{'='*60}")
        print(f"DFCA - Pure Decentralized Federated Clustering Algorithm")
        print(f"{'='*60}")
        print(f"Nodes: {len(node_ids)}, Clusters k={num_clusters}")
        print(f"Init: {dfca_init}, Graph: {dfca_graph}(p={dfca_connectivity})")
        print(f"Optimizer: {optimizer}, LR={lr}, LocalEpochs={local_epochs}")
        print(f"Participation: {participation_rate:.0%}, Rounds: {num_rounds}")
        print(f"Device: {'cpu' if num_gpus == 0 else f'{num_gpus} GPUs'}")

    graph_neighbors = build_erdos_renyi_graph(
        node_ids=node_ids,
        connectivity=dfca_connectivity,
        seed=seed,
        ensure_connectivity=True,
    )
    graph_summary = build_graph_summary(graph_neighbors, node_ids)

    if verbose:
        print(
            f"  Graph: {graph_summary['num_nodes']} nodes, "
            f"{graph_summary['num_edges']} edges, "
            f"avg_deg={graph_summary['avg_degree']:.1f}, "
            f"min_deg={graph_summary['min_degree']}, "
            f"max_deg={graph_summary['max_degree']}, "
            f"isolated={graph_summary['isolated_count']}"
        )

    # ---- Create Nodes ----
    nodes: Dict[int, DFCANode] = {}
    for nid in node_ids:
        X, y = node_data[nid]
        node = DFCANode(
            client_id=nid,
            X_train=X,
            y_train=y,
            num_clusters=num_clusters,
            num_classes=num_classes,
            init_seed=seed + nid,
        )

        if dfca_init == "global":
            global_params = OrderedDict(
                (k, v.cpu().clone())
                for k, v in model_template.state_dict().items()
            )
            node.initialize_cluster_bank(
                global_params=global_params,
                init_type="global",
            )
        else:
            node.initialize_cluster_bank(init_type="local")

        node.set_neighbors(graph_neighbors.get(nid, []))
        nodes[nid] = node

    # ---- Determine global params for model template ----
    first_node = nodes[node_ids[0]]
    global_params = first_node.cluster_params[0]

    # ---- Training Loop ----
    history = {
        "round": [],
        "train_loss": [],
        "train_loss_std": [],
        "assignment_changes": [],
        "num_messages": [],
        "participating_nodes": [],
        "cluster_distribution": [],
        "per_cluster_updates": [],
        "assignment_margin_avg": [],
        "round_time": [],
        "test_loss": [],
        "test_accuracy": [],
        "test_precision_macro": [],
        "test_recall_macro": [],
        "test_f1_macro": [],
        "test_f1_weighted": [],
    }

    cluster_history: List[Dict] = []
    prev_assignments: Dict[int, int] = {nid: -1 for nid in node_ids}

    for round_r in range(num_rounds):
        round_start = time.time()

        # ---- Determine participating nodes ----
        if participation_rate >= 1.0:
            participating_ids = list(node_ids)
        else:
            n_participate = max(1, int(len(node_ids) * participation_rate))
            rng_round = random.Random(seed + round_r)
            participating_ids = rng_round.sample(node_ids, n_participate)

        participating_ids_set = set(participating_ids)

        # Assign GPU devices to nodes
        if num_gpus == 0:
            device_map = {nid: "cpu" for nid in participating_ids}
        else:
            device_map = {}
            for i, nid in enumerate(participating_ids):
                device_map[nid] = f"cuda:{i % num_gpus}"

        if verbose:
            print(f"\n--- Round {round_r}/{num_rounds - 1} "
                  f"[{len(participating_ids)}/{len(node_ids)} nodes] ---")

        # ---- Step 1: Cluster Assignment ----
        if debug_assignments:
            print("  [Step 1] AssignCluster...")

        assignment_results: Dict[int, Any] = {}
        train_results: Dict[int, Any] = {}

        if num_gpus == 0:
            cpu_model_template = model_class(input_shape, num_classes)
            for nid in participating_ids:
                node = nodes[nid]
                assigned, losses, margin = node.assign_cluster(
                    model_template=cpu_model_template,
                    device="cpu",
                    verbose=debug_assignments,
                )
                assignment_results[nid] = {
                    "assigned": assigned, "losses": losses, "margin": margin
                }
        else:
            results_lock = Lock()
            threads = []
            for gpu_id in range(num_gpus):
                batch = [
                    nid for i, nid in enumerate(participating_ids)
                    if i % num_gpus == gpu_id
                ]
                if not batch:
                    continue
                t = Thread(
                    target=_worker_assign_and_train,
                    args=(
                        batch, nodes, model_class, input_shape, num_classes,
                        f"cuda:{gpu_id}", optimizer, local_epochs,
                        lr, batch_size, debug_assignments,
                        assignment_results, train_results, results_lock,
                    ),
                )
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

        # ---- Step 2: Local Update ----
        if debug_assignments:
            print("  [Step 2] LocalUpdate...")

        if num_gpus == 0:
            cpu_model_template = model_class(input_shape, num_classes)
            for nid in participating_ids:
                node = nodes[nid]
                result = node.train_assigned_cluster(
                    model_template=cpu_model_template,
                    device="cpu",
                    epochs=local_epochs,
                    batch_size=batch_size,
                    lr=lr,
                    verbose=False,
                )
                train_results[nid] = result
        else:
            # Already done in the same thread as Step 1 above
            pass

        # Collect train losses
        train_losses = [r.get("loss", 0.0) for r in train_results.values()]
        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        std_train_loss = float(np.std(train_losses)) if len(train_losses) > 1 else 0.0

        # ---- Step 3: Decentralized Aggregation ----
        if debug_assignments:
            print("  [Step 3] Decentralized Aggregation...")

        num_messages, per_cluster_updates, messages = _run_message_passing(
            nodes, participating_ids, graph_neighbors, num_clusters,
            debug_messages, debug_message_limit,
        )

        # ---- Detailed message log for this round ----
        msg_log = {}
        for nid in participating_ids:
            node = nodes[nid]
            neighbors_in_round = graph_neighbors.get(nid, [])
            sent_cluster = node.assigned_cluster
            received = {}
            if hasattr(node, "received_messages") and node.received_messages:
                for sender_id, msg in node.received_messages.items():
                    received[sender_id] = list(msg.keys())
            msg_log[nid] = {
                "assigned_cluster": sent_cluster,
                "neighbors": neighbors_in_round,
                "received_from": received,
                "messages_sent": len(messages.get(nid, {})),
            }

        # ---- Update Representative Models ----
        rep_params = _build_representative_models(
            nodes, participating_ids, num_clusters
        )

        # ---- Cluster Distribution & Assignment Changes ----
        cluster_counts = Counter(
            assignment_results.get(nid, {}).get("assigned", 0)
            for nid in participating_ids
        )
        assignment_changes = sum(
            1 for nid in participating_ids
            if prev_assignments.get(nid, -1) != -1
            and assignment_results.get(nid, {}).get("assigned", 0) != prev_assignments.get(nid, 0)
        )
        for nid in participating_ids:
            prev_assignments[nid] = assignment_results.get(nid, {}).get("assigned", 0)

        # Assignment margin
        margins = [
            assignment_results.get(nid, {}).get("margin", 0.0)
            for nid in participating_ids
            if nid in assignment_results
        ]
        avg_margin = float(np.mean(margins)) if margins else 0.0

        # ---- Debug cluster models ----
        if debug_cluster_models:
            zero_update_clusters = [
                cid for cid, cnt in per_cluster_updates.items() if cnt == 0
            ]
            dominated = any(
                cnt / max(1, len(participating_ids)) > 0.8
                for cnt in cluster_counts.values()
            )
            if zero_update_clusters:
                print(f"  [DFCA][clusters] no updates: c{zero_update_clusters}")
            if dominated:
                dominating = max(cluster_counts, key=cluster_counts.get)
                print(
                    f"  [DFCA][clusters] WARNING: cluster c{dominating} "
                    f"has {cluster_counts[dominating]/max(1,len(participating_ids))*100:.0f}% "
                    f"of nodes — possible collapse"
                )

        round_time = time.time() - round_start

        # ---- Evaluation ----
        test_metrics = {}
        eval_this_round = (test_data is not None) and (
            eval_every > 0 and round_r % eval_every == (eval_every - 1)
        )

        if eval_this_round:
            X_test = test_data["X_test"]
            y_test = test_data["y_test"]

            ensemble_metrics = evaluate_ensemble_average(
                nodes, rep_params, X_test, y_test,
                model_class, input_shape, num_classes,
                device=device_main,
                batch_size=batch_size,
            )
            test_metrics = ensemble_metrics

            if verbose:
                print(
                    f"  [Eval] Acc={ensemble_metrics['accuracy']*100:.2f}%, "
                    f"F1={ensemble_metrics['f1_macro']*100:.2f}%, "
                    f"Loss={ensemble_metrics['loss']:.4f}"
                )

        # ---- Log ----
        if verbose:
            cluster_str = ", ".join(f"c{c}={n}" for c, n in sorted(cluster_counts.items()))
            cluster_updates_str = _format_cluster_updates(per_cluster_updates)
            print(
                f"  Cluster dist: [{cluster_str}], "
                f"changes={assignment_changes}, "
                f"messages={num_messages}, "
                f"train_loss={avg_train_loss:.4f}±{std_train_loss:.4f}"
            )
            print(f"  [DFCA] Cluster updates: {cluster_updates_str}")
            print(f"  Round time: {round_time:.2f}s")

        round_record = {
            "round": round_r,
            "participating_nodes": len(participating_ids),
            "train_loss": avg_train_loss,
            "train_loss_std": std_train_loss,
            "assignment_changes": assignment_changes,
            "assignment_margin_avg": avg_margin,
            "cluster_distribution": dict(cluster_counts),
            "num_messages": num_messages,
            "per_cluster_updates": dict(per_cluster_updates),
            "round_time": round_time,
            "test_loss": test_metrics.get("loss"),
            "test_accuracy": test_metrics.get("accuracy"),
            "test_precision_macro": test_metrics.get("precision_macro"),
            "test_recall_macro": test_metrics.get("recall_macro"),
            "test_f1_macro": test_metrics.get("f1_macro"),
            "test_f1_weighted": test_metrics.get("f1_weighted"),
            "msg_log": msg_log,
            "rep_params_summary": _rep_params_summary(rep_params),
        }

        history["round"].append(round_r)
        history["train_loss"].append(avg_train_loss)
        history["train_loss_std"].append(std_train_loss)
        history["assignment_changes"].append(assignment_changes)
        history["num_messages"].append(num_messages)
        history["participating_nodes"].append(len(participating_ids))
        history["cluster_distribution"].append(dict(cluster_counts))
        history["per_cluster_updates"].append(dict(per_cluster_updates))
        history["assignment_margin_avg"].append(avg_margin)
        history["round_time"].append(round_time)

        if test_metrics:
            history["test_loss"].append(test_metrics.get("loss"))
            history["test_accuracy"].append(test_metrics.get("accuracy"))
            history["test_precision_macro"].append(test_metrics.get("precision_macro"))
            history["test_recall_macro"].append(test_metrics.get("recall_macro"))
            history["test_f1_macro"].append(test_metrics.get("f1_macro"))
            history["test_f1_weighted"].append(test_metrics.get("f1_weighted"))

        cluster_history.append({
            "round": round_r,
            "cluster_distribution": dict(cluster_counts),
            "assignment_changes": assignment_changes,
            "per_cluster_updates": dict(per_cluster_updates),
            "test_metrics": test_metrics,
        })

        if round_callback:
            round_callback(round_r, history, round_record)

    # ---- Final State ----
    final_assignments = {
        nid: nodes[nid].assigned_cluster for nid in node_ids
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"DFCA Training Complete — {num_rounds} rounds")
        final_dist = Counter(final_assignments.values())
        print(f"Final cluster distribution: {dict(sorted(final_dist.items()))}")

    return {
        "history": history,
        "final_assignments": final_assignments,
        "cluster_history": cluster_history,
        "graph_summary": graph_summary,
        "representative_params": rep_params,
        "config": config,
    }
