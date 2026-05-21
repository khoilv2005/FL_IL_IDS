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
) -> tuple[int, Dict[int, int], Dict[str, Any]]:
    """
    Simulate peer-to-peer message passing.

    Each participating node exports its assigned cluster model and sends it
    to every neighbor that also participated this round.

    Returns:
        (num_messages, per_cluster_updates, delivery_log)
        - num_messages: total deliveries (sender → each participating neighbor)
        - per_cluster_updates: Dict[cluster_id -> count of received messages for that cluster]
        - delivery_log: Dict[sender_id -> dict with sender, assigned_cluster, recipients, deliveries]
          built BEFORE clearing received_messages so received_from is populated
    """
    participating_set = set(participating_ids)

    # Step A: Export messages
    messages: Dict[int, Dict[int, OrderedDict]] = {}
    for nid in participating_ids:
        node = nodes[nid]
        msg = node.export_assigned_cluster_message()
        messages[nid] = msg
        node.received_messages = {}

    # Step B: Build delivery_log BEFORE delivering (so received_messages is populated)
    delivery_log: Dict[str, Any] = {}
    for nid in participating_ids:
        msg = messages.get(nid, {})
        assigned_cluster = nodes[nid].assigned_cluster
        # Recipients: neighbors that also participated
        recipients = [m for m in neighbors.get(nid, []) if m in participating_set]
        delivery_log[f"node_{nid}"] = {
            "sender_id": nid,
            "assigned_cluster": assigned_cluster,
            "recipients": recipients,
            "delivery_count": len(recipients),
            "clusters_sent": [assigned_cluster],
        }

    # Step C: Deliver messages AND build received_from snapshot before clearing
    received_from_snapshot: Dict[int, Dict[int, List[int]]] = {}  # nid -> {sender_id -> [cluster_ids]}
    for nid in participating_ids:
        node = nodes[nid]
        node_neighbors = neighbors.get(nid, [])
        received_from_snapshot[nid] = {}
        for sender_id in node_neighbors:
            if sender_id not in messages:
                continue
            node.receive_message(sender_id, messages[sender_id])
            if sender_id not in received_from_snapshot[nid]:
                received_from_snapshot[nid][sender_id] = []
            received_from_snapshot[nid][sender_id].extend(list(messages[sender_id].keys()))

    # Step D: Aggregate and count updates
    per_cluster_updates: Dict[int, int] = {c: 0 for c in range(num_clusters)}
    for nid in participating_ids:
        node = nodes[nid]
        update_counts = node.aggregate_received_messages()
        for cid, cnt in update_counts.items():
            per_cluster_updates[cid] = per_cluster_updates.get(cid, 0) + cnt

    # Step E: num_messages = total deliveries
    num_messages = sum(
        len([m for m in neighbors.get(nid, []) if m in participating_set])
        for nid in participating_ids
    )

    # Step F: Console debug (limited)
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

            entry = delivery_log.get(f"node_{nid}", {})
            assigned_c = entry.get("assigned_cluster", "?")
            recipients = entry.get("recipients", [])
            count = entry.get("delivery_count", 0)

            if recipients:
                rec_str = ", ".join(f"node_{r}" for r in sorted(recipients))
                print(
                    f"[DFCA][messages] sender=node_{nid} cluster={assigned_c} "
                    f"recipients=[{rec_str}] count={count}"
                )
            else:
                print(
                    f"[DFCA][messages] sender=node_{nid} cluster={assigned_c} "
                    f"recipients=[] count=0"
                )
            logged_count += 1

    return num_messages, per_cluster_updates, delivery_log, received_from_snapshot


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
    checkpoint_callback: Optional[Callable] = None,
    resume_state: Optional[Dict] = None,
    start_round: int = 0,
    num_rounds_override: Optional[int] = None,
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
        checkpoint_callback: Optional callback to save full DFCA checkpoint after each round.
            Signature: (nodes_dict, graph_neighbors_dict, prev_assign_dict, history_dict,
                       cluster_history_list, rep_params_dict, cfg, current_round_int, num_rounds_int) -> None.
            Called after each round (caller controls when to actually save to disk).
        resume_state: Optional dict with saved DFCA state to resume from.
            When provided, must contain:
                - nodes_state: Dict[node_id -> state_dict of DFCANode]
                - graph_neighbors: Dict[node_id -> List[neighbor_ids]]
                - prev_assignments: Dict[node_id -> cluster_id]
                - history: already-collected history dict
                - cluster_history: already-collected cluster_history list
                - current_round: last completed round index (resume from +1)
            Optional:
                - rng_state: Dict with random/numpy/torch/cuda state

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
    debug_assignments = config.get("dfca_debug_assignments", False)
    debug_messages = config.get("dfca_debug_messages", False)
    debug_message_limit = config.get("dfca_debug_message_limit", 25)
    debug_cluster_models = config.get("dfca_debug_cluster_models", False)

    num_classes = model_template.num_classes
    input_shape = model_template.input_shape
    model_class = model_template.__class__

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if config.get("num_gpus", 0) > 0:
        num_gpus = config["num_gpus"]

    device_main = "cpu" if num_gpus == 0 else "cuda:0"
    nodes: Dict[int, DFCANode] = {}  # initialized early to avoid UnboundLocalError

    is_resumed = resume_state is not None
    # resume_state takes priority over start_round for full state restoration
    if is_resumed:
        resumed_round = resume_state.get("current_round", -1) + 1
    else:
        resumed_round = start_round

    # Determine if we can restore full cluster banks or must reinitialize
    has_cluster_banks = False  # set to True inside is_resumed block if banks exist

    # ---- Restore RNG state if resuming ----
    if is_resumed and "rng_state" in resume_state:
        rng_st = resume_state["rng_state"]
        if "random" in rng_st:
            random.setstate(rng_st["random"])
        if "numpy" in rng_st:
            np.random.set_state(rng_st["numpy"])
        if "torch_cpu" in rng_st:
            torch.set_rng_state(rng_st["torch_cpu"])
        if "torch_cuda" in rng_st and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_st["torch_cuda"])

    # ---- Build / Restore Graph ----
    if verbose:
        print(f"\n{'='*60}")
        print(f"DFCA - Pure Decentralized Federated Clustering Algorithm")
        print(f"{'='*60}")
        print(f"Nodes: {len(node_ids)}, Clusters k={num_clusters}")
        print(f"Init: {dfca_init}, Graph: {dfca_graph}(p={dfca_connectivity})")
        print(f"Optimizer: {optimizer}, LR={lr}, LocalEpochs={local_epochs}")
        print(f"Participation: {participation_rate:.0%}, Rounds: {num_rounds}")
        print(f"Device: {'cpu' if num_gpus == 0 else f'{num_gpus} GPUs'}")

    if is_resumed:
        graph_neighbors = {
            int(nid): [int(nbr) for nbr in nbrs]
            for nid, nbrs in resume_state["graph_neighbors"].items()
        }
        graph_summary = build_graph_summary(graph_neighbors, node_ids)
        if verbose:
            print(f"  [Resume] Graph restored from checkpoint.")
            print(
                f"  Graph: {graph_summary['num_nodes']} nodes, "
                f"{graph_summary['num_edges']} edges, "
                f"avg_deg={graph_summary['avg_degree']:.1f}"
            )
    else:
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

    # ---- Create / Restore Nodes ----
    if is_resumed:
        nodes_state = resume_state.get("nodes_state", {})
        nodes_state = {int(nid): clusters for nid, clusters in nodes_state.items()}
        # If nodes_state is empty (e.g. old checkpoint), reinitialize from scratch
        has_cluster_banks = (
            bool(nodes_state)
            and any(bool(v) for v in nodes_state.values())
        )
        if has_cluster_banks:
            # ---- Full restore: nodes, graph, history ----
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
                node.cluster_params = {}
                if nid in nodes_state and nodes_state[nid]:
                    for cid, state_dict in nodes_state[nid].items():
                        node.cluster_params[cid] = OrderedDict()
                        for k, v in state_dict.items():
                            node.cluster_params[cid][k] = v.cpu().clone()
                node_assignments = {
                    int(saved_nid): int(saved_cid)
                    for saved_nid, saved_cid in resume_state.get("node_assignments", {}).items()
                }
                assigned = node_assignments.get(nid, 0)
                node.assigned_cluster = assigned
                node.set_neighbors(graph_neighbors.get(nid, []))
                nodes[nid] = node

            prev_assignments = {
                int(saved_nid): int(saved_cid)
                for saved_nid, saved_cid in resume_state.get("prev_assignments", {}).items()
            }
            history = _restore_history(resume_state.get("history", {}))
            cluster_history = list(resume_state.get("cluster_history", []))
            if verbose:
                print(f"  [Resume] Restored {len(nodes)} nodes, "
                      f"cluster banks and assignments loaded from checkpoint.")
                print(f"  [Resume] Continuing from round {resumed_round}, "
                      f"history has {len(history['round'])} entries.")
        else:
            raise ValueError(
                "Cannot resume DFCA checkpoint: nodes_state/cluster banks are missing. "
                "Use a full DFCA checkpoint created by build_dfca_checkpoint()."
            )
            # Build fresh nodes (rest of the block below)
    else:
        nodes_state = {}

    # Fresh init (also used when checkpoint had no cluster banks)
    if not (is_resumed and has_cluster_banks):
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
                node.initialize_cluster_bank(
                    template_model=model_template,
                    init_type="local",
                )

            node.set_neighbors(graph_neighbors.get(nid, []))
            nodes[nid] = node

        prev_assignments = {nid: -1 for nid in node_ids}
        history = _make_empty_history()
        cluster_history = []

    # ---- Restore representative params for final return ----
    rep_params = (
        resume_state.get("representative_params", {})
        if (is_resumed and has_cluster_banks) else {}
    )

    # ---- Determine total rounds ----
    if num_rounds_override is not None:
        total_rounds = resumed_round + max(0, int(num_rounds_override))
    else:
        total_rounds = int(num_rounds)

    for round_r in range(resumed_round, total_rounds):
        round_start = time.time()

        # ---- Determine participating nodes ----
        if participation_rate >= 1.0:
            participating_ids = list(node_ids)
        else:
            n_participate = max(1, int(len(node_ids) * participation_rate))
            rng_round = random.Random(seed + round_r)
            participating_ids = rng_round.sample(node_ids, n_participate)

        if verbose:
            print(f"\n--- Round {round_r}/{total_rounds - 1} "
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
            pass  # Already done in the same thread as Step 1 above

        train_losses = [r.get("loss", 0.0) for r in train_results.values()]
        avg_train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        std_train_loss = float(np.std(train_losses)) if len(train_losses) > 1 else 0.0

        # ---- Step 3: Decentralized Aggregation ----
        if debug_assignments:
            print("  [Step 3] Decentralized Aggregation...")

        num_messages, per_cluster_updates, delivery_log, received_from_snapshot = _run_message_passing(
            nodes, participating_ids, graph_neighbors, num_clusters,
            debug_messages, debug_message_limit,
        )

        # ---- Detailed message log for round_record ----
        # received_from_snapshot already captured before aggregate_received_messages cleared messages
        msg_log: Dict[str, Any] = {}
        for nid in participating_ids:
            nid_str = f"node_{nid}"
            entry = delivery_log.get(nid_str, {})
            msg_log[nid_str] = {
                "sender_id": nid,
                "assigned_cluster": nodes[nid].assigned_cluster,
                "recipients": entry.get("recipients", []),
                "delivery_count": entry.get("delivery_count", 0),
                "clusters_sent": entry.get("clusters_sent", []),
                "received_from": {int(k): v for k, v in received_from_snapshot.get(nid, {}).items()},
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
        test_metrics: Dict[str, Any] = {}
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

        # ---- Round Summary ----
        if verbose:
            cluster_str = ", ".join(f"c{c}={n}" for c, n in sorted(cluster_counts.items()))
            cluster_updates_str = _format_cluster_updates(per_cluster_updates)
            active_senders = sum(1 for d in delivery_log.values() if d.get("delivery_count", 0) > 0)
            print(
                f"  Cluster dist: [{cluster_str}], "
                f"changes={assignment_changes}, "
                f"messages={num_messages} (senders={active_senders}), "
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
        history["test_loss"].append(test_metrics.get("loss") if test_metrics else None)
        history["test_accuracy"].append(test_metrics.get("accuracy") if test_metrics else None)
        history["test_precision_macro"].append(test_metrics.get("precision_macro") if test_metrics else None)
        history["test_recall_macro"].append(test_metrics.get("recall_macro") if test_metrics else None)
        history["test_f1_macro"].append(test_metrics.get("f1_macro") if test_metrics else None)
        history["test_f1_weighted"].append(test_metrics.get("f1_weighted") if test_metrics else None)

        cluster_history.append({
            "round": round_r,
            "cluster_distribution": dict(cluster_counts),
            "assignment_changes": assignment_changes,
            "per_cluster_updates": dict(per_cluster_updates),
            "test_metrics": dict(test_metrics) if test_metrics else {},
        })

        if round_callback:
            round_callback(round_r, history, round_record)

        if checkpoint_callback:
            checkpoint_callback(
                nodes_dict=nodes,
                graph_neighbors_dict=graph_neighbors,
                prev_assign_dict=prev_assignments,
                history_dict=history,
                cluster_history_list=cluster_history,
                rep_params_dict=rep_params,
                cfg=config,
                current_round_int=round_r,
                num_rounds_int=total_rounds,
            )

    # ---- Final State ----
    final_assignments = {
        nid: nodes[nid].assigned_cluster for nid in node_ids
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"DFCA Training Complete — {total_rounds} rounds")
        final_dist = Counter(final_assignments.values())
        print(f"Final cluster distribution: {dict(sorted(final_dist.items()))}")

    current_round = history["round"][-1] if history["round"] else resumed_round - 1
    checkpoint_state = build_dfca_checkpoint(
        nodes=nodes,
        graph_neighbors=graph_neighbors,
        prev_assignments=prev_assignments,
        history=history,
        cluster_history=cluster_history,
        representative_params=rep_params,
        config=config,
        current_round=current_round,
        num_rounds=total_rounds,
    )

    return {
        "history": history,
        "final_assignments": final_assignments,
        "cluster_history": cluster_history,
        "graph_summary": graph_summary,
        "graph_neighbors": graph_neighbors,
        "representative_params": rep_params,
        "checkpoint_state": checkpoint_state,
        "config": config,
    }


# =============================================================================
# Checkpoint helpers
# =============================================================================

def _make_empty_history() -> Dict[str, List]:
    return {
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


def _restore_history(saved: Dict[str, List]) -> Dict[str, List]:
    """Restore history dict from checkpoint, filling missing keys with empty lists."""
    template = _make_empty_history()
    restored = {}
    for k, v in template.items():
        restored[k] = list(saved.get(k, []))
    return restored


def build_dfca_checkpoint(
    nodes: Dict[int, Any],
    graph_neighbors: Dict[int, List[int]],
    prev_assignments: Dict[int, int],
    history: Dict[str, List],
    cluster_history: List[Dict],
    representative_params: Dict,
    config: Dict,
    current_round: int,
    num_rounds: int,
) -> Dict[str, Any]:
    """
    Build a full DFCA checkpoint dict containing everything needed to resume.

    Args:
        nodes: Dict[node_id -> DFCANode] with cluster_params and assigned_cluster.
        graph_neighbors: Dict[node_id -> List[neighbor_ids]].
        prev_assignments: Dict[node_id -> cluster_id].
        history: current history dict.
        cluster_history: current cluster_history list.
        representative_params: current representative params dict.
        config: training config.
        current_round: last completed round index.
        num_rounds: total planned rounds.

    Returns:
        Dict suitable for torch.save().
    """
    # Serialize node states — keys must be consistent (int) for both nodes and cluster banks
    nodes_state: Dict[int, Dict[int, Dict[str, torch.Tensor]]] = {}
    node_assignments: Dict[int, int] = {}
    for nid, node in nodes.items():
        if not hasattr(node, "cluster_params"):
            continue
        nodes_state[nid] = {}
        for cid, params in node.cluster_params.items():
            nodes_state[nid][cid] = {k: v.clone().cpu() for k, v in params.items()}
        node_assignments[nid] = getattr(node, "assigned_cluster", 0)

    # RNG state
    rng_state: Dict[str, Any] = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return {
        "version": 1,
        "current_round": current_round,
        "total_rounds": num_rounds,
        "config": config,
        "nodes_state": nodes_state,
        "node_assignments": node_assignments,
        "graph_neighbors": graph_neighbors,
        "prev_assignments": prev_assignments,
        "history": history,
        "cluster_history": cluster_history,
        "representative_params": representative_params,
        "rng_state": rng_state,
    }


def find_latest_checkpoint(base_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in base_dir or any timestamped subdirectory.

    Search patterns:
        {base_dir}/checkpoint_round_*.pt
        {base_dir}_*/checkpoint_round_*.pt
        {base_dir}/final_dfca_state.pt
        {base_dir}_*/final_dfca_state.pt

    Returns:
        Absolute path to the latest checkpoint, or None if not found.
    """
    import glob as glob_module

    candidates: List[tuple[int, float, str]] = []

    def add_round_checkpoint(path: str) -> None:
        try:
            idx = int(os.path.basename(path).split("_")[-1].split(".")[0])
            candidates.append((idx, os.path.getmtime(path), path))
        except (ValueError, OSError):
            pass

    def add_final_checkpoint(path: str) -> None:
        try:
            idx = -1
            try:
                state = torch.load(path, map_location="cpu", weights_only=False)
                idx = int(state.get("current_round", state.get("round", -1)))
            except Exception:
                idx = 10**9
            candidates.append((idx, os.path.getmtime(path), path))
        except OSError:
            pass

    search_dirs = [base_dir]
    search_dirs.extend(
        p for p in glob_module.glob(f"{base_dir}_*") if os.path.isdir(p)
    )
    search_dirs.extend(
        p for p in glob_module.glob(os.path.join(base_dir, "*")) if os.path.isdir(p)
    )

    for directory in search_dirs:
        for path in glob_module.glob(os.path.join(directory, "checkpoint_round_*.pt")):
            add_round_checkpoint(path)
        final_path = os.path.join(directory, "final_dfca_state.pt")
        if os.path.exists(final_path):
            add_final_checkpoint(final_path)

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]
