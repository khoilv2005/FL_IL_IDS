"""
PlexusRunner - Pure Plexus decentralized training (no server, no incremental).

Reference:
    Dhasade et al., "Practical Federated Learning without a Server",
    EuroMLSys 2025, Algorithm 1 & Algorithm 2

This is a pure reimplementation of the Plexus protocol:
- Algorithm 1 (DERIVE_SAMPLE): Hash-based peer sampling
- Algorithm 2 (PUSH-BASED TRAINING): No server, aggregator node performs aggregation

NO incremental learning. NO central server. NO PlexusServer.
Pure decentralized federated learning following the paper exactly.

Usage:
    from fed_learning.plexus.runner import run_plexus_training

    results = run_plexus_training(
        node_ids=list(range(10)),
        node_data={i: (X_train[i], y_train[i]) for i in range(10)},
        model_template=CNN_GRU_Model(input_shape, num_classes),
        config={
            "num_rounds": 10,
            "plexus_sample_size": 4,
            "plexus_success_fraction": 0.8,
            "local_epochs": 1,
            "learning_rate": 0.001,
            "batch_size": 32,
        }
    )
"""

import random
import time
from collections import OrderedDict
from typing import Dict, List, Any, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .sampler import PlexusSampler
from .aggregator import PlexusAggregator
from .orchestrator import NodeWrapper


def run_plexus_training(
    node_ids: List[int],
    node_data: Dict[int, tuple],  # node_id -> (X_train, y_train)
    model_template: nn.Module,
    config: Dict[str, Any],
    test_data: Dict = None,
    initial_global_params: Optional[OrderedDict] = None,
    seen_classes: Optional[List[int]] = None,
    verbose: bool = True,
    round_callback: Optional[Callable[[int, OrderedDict, Dict[str, Any]], None]] = None,
) -> Dict:
    """
    Run pure Plexus decentralized training (Algorithm 1 & 2 from paper).

    NO server. NO incremental learning.
    Each node follows Algorithm 2 autonomously.

    Protocol:
        Round r:
        1. DERIVE_SAMPLE selects K nodes + highest-BW node as aggregator
        2. Each sample node trains locally -> sends to aggregator
        3. Aggregator collects until threshold (K * s_f)
        4. Aggregator aggregates (FedAvg) -> push to next sample
        5. Next round starts

    Args:
        node_ids: List of participating node IDs
        node_data: Dict mapping node_id -> (X_train Tensor, y_train Tensor)
        model_template: Model architecture to clone for each node
        config: Training config with keys:
            - num_rounds: Number of training rounds
            - plexus_sample_size: K (default 4)
            - plexus_success_fraction: s_f (default 0.8)
            - local_epochs: Local epochs per round (default 1)
            - learning_rate: Learning rate (default 0.001)
            - batch_size: Batch size (default 32)
        test_data: Optional test data for evaluation
        verbose: Print progress

    Returns:
        Dict with training history and final global_params
    """
    # Plexus parameters
    sample_size = config.get("plexus_sample_size", 4)
    success_fraction = config.get("plexus_success_fraction", 0.8)
    num_rounds = config.get("num_rounds", 10)
    local_epochs = config.get("local_epochs", 1)
    learning_rate = config.get("learning_rate", 0.001)
    batch_size = config.get("batch_size", 32)

    # Generate simulated bandwidths (log-normal distribution, as in paper)
    rng = random.Random(config.get("seed", 42))
    bandwidths = {
        nid: round(rng.lognormvariate(mu=3.0, sigma=0.8), 2)
        for nid in node_ids
    }

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create PlexusSampler (Algorithm 1)
    sampler = PlexusSampler(
        node_ids=node_ids,
        sample_size=sample_size,
    )

    # Create PlexusAggregator
    aggregator = PlexusAggregator(
        sample_size=sampler.sample_size,
        success_fraction=success_fraction,
    )

    # Create NodeWrappers (each node is autonomous peer)
    nodes = {}
    for nid in node_ids:
        X_train, y_train = node_data[nid]
        nodes[nid] = NodeWrapper(
            node_id=nid,
            X_train=X_train,
            y_train=y_train,
            bandwidth=bandwidths[nid],
            model_template=model_template,
            device=device,
            batch_size=batch_size,
        )

    # Global params (initialized from first sample's model)
    global_params = (
        OrderedDict((k, v.cpu().clone()) for k, v in initial_global_params.items())
        if initial_global_params is not None
        else None
    )

    # History
    history = {
        "round": [],
        "sample": [],
        "aggregator": [],
        "participation": [],
        "loss": [],
        "round_time": [],
        "test_loss": [],
        "test_accuracy": [],
        "test_precision_macro": [],
        "test_recall_macro": [],
        "test_f1_macro": [],
        "test_f1_weighted": [],
    }

    if verbose:
        print("\n" + "=" * 60)
        print("PLEXUS — Pure Decentralized Training (No Server)")
        print(f"Rounds: {num_rounds}, Sample size: {sample_size}")
        print(f"Success fraction: {success_fraction}, Local epochs: {local_epochs}")
        print("=" * 60 + "\n")

    # Run Algorithm 2: Push-based protocol
    for round_r in range(num_rounds):
        round_start = time.time()

        # Step 1: DERIVE_SAMPLE (Algorithm 1) — sample + highest-BW aggregator
        sample_ids, aggregator_id = sampler.derive_sample_with_bandwidths(
            round_r, bandwidths
        )

        # Initialize global params from first node's model
        if global_params is None:
            global_params = OrderedDict(
                (k, v.cpu().clone())
                for k, v in nodes[sample_ids[0]].model.state_dict().items()
            )

        if verbose:
            bw_info = {a: bandwidths.get(a, 0) for a in [aggregator_id]}
            print(f"→ Round {round_r}: sample={sample_ids}, aggregator={aggregator_id} (bw={bw_info[aggregator_id]:.2f})")

        threshold = aggregator.get_threshold()
        train_ids = sample_ids[:threshold]

        # Step 2: Threshold sample nodes train locally. Late participants are
        # ignored once success_fraction is reached in this synchronous simulation.
        results = []
        for nid in train_ids:
            result = nodes[nid].receive_train(
                round_r=round_r,
                global_params=global_params,
                derive_sample_fn=lambda nids, rnd, K: sampler.derive_sample_with_bandwidths(rnd, bandwidths),
                bandwidths=bandwidths,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
            )
            results.append(result)

        # Step 3: Route results to aggregator node (Algorithm 2 step 2)
        for result in results:
            nodes[aggregator_id].receive_for_aggregation(round_r, result)

        if verbose:
            print(f"   Trained {len(results)}/{len(sample_ids)} nodes, threshold={threshold}")

        aggregated_params = None
        if nodes[aggregator_id].can_aggregate(round_r, threshold):
            # Aggregator performs aggregation (NOT orchestrator/server!)
            aggregated_params = nodes[aggregator_id].aggregate(
                round_r,
                aggregator,
                global_params,
            )

            if aggregated_params is not None:
                global_params = aggregated_params
                if verbose:
                    print(f"   Aggregator {aggregator_id} aggregated {len(results)} models")
            else:
                if verbose:
                    print(f"   ⚠️ Aggregation returned None")
        else:
            if verbose:
                print(f"   ⚠️ Threshold not met ({len(results)} < {threshold})")

        # Record history
        round_time = time.time() - round_start
        avg_loss = np.mean([r["loss"] for r in results]) if results else 0.0

        history["round"].append(round_r)
        history["sample"].append(sample_ids)
        history["aggregator"].append(aggregator_id)
        history["participation"].append(len(results) / len(sample_ids) if sample_ids else 0)
        history["loss"].append(avg_loss)
        history["round_time"].append(round_time)

        round_metrics = {
            "train_loss": avg_loss,
            "round_time": round_time,
            "loss": None,
            "accuracy": None,
            "precision_macro": None,
            "recall_macro": None,
            "f1_macro": None,
            "f1_weighted": None,
        }
        if test_data is not None:
            eval_metrics = _evaluate_metrics(
                global_params,
                nodes[sample_ids[0]].model,
                test_data,
                batch_size,
                seen_classes=seen_classes,
            )
            round_metrics.update(eval_metrics)
            history["test_loss"].append(eval_metrics["loss"])
            history["test_accuracy"].append(eval_metrics["accuracy"])
            history["test_precision_macro"].append(eval_metrics["precision_macro"])
            history["test_recall_macro"].append(eval_metrics["recall_macro"])
            history["test_f1_macro"].append(eval_metrics["f1_macro"])
            history["test_f1_weighted"].append(eval_metrics["f1_weighted"])
            if verbose:
                print(
                    "   Metrics -> "
                    f"train_loss={avg_loss:.4f}, test_loss={eval_metrics['loss']:.4f}, "
                    f"accuracy={eval_metrics['accuracy'] * 100:.2f}%, "
                    f"f1={eval_metrics['f1_macro'] * 100:.2f}%, "
                    f"precision={eval_metrics['precision_macro'] * 100:.2f}%, "
                    f"recall={eval_metrics['recall_macro'] * 100:.2f}%"
                )
        else:
            history["test_loss"].append(None)
            history["test_accuracy"].append(None)
            history["test_precision_macro"].append(None)
            history["test_recall_macro"].append(None)
            history["test_f1_macro"].append(None)
            history["test_f1_weighted"].append(None)

        if round_callback is not None:
            round_callback(round_r, global_params, round_metrics)

    if verbose:
        print("\n" + "=" * 60)
        print("PLEXUS Training Complete")
        print(f"Final loss: {history['loss'][-1]:.4f}")
        print(f"Unique aggregators: {len(set(history['aggregator']))}")
        print(f"Avg participation: {np.mean(history['participation']):.2%}")
        print("=" * 60)

    return {
        "history": history,
        "global_params": global_params,
    }


def _evaluate_metrics(
    global_params: OrderedDict,
    model_template: nn.Module,
    test_data: Dict,
    batch_size: int,
    seen_classes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Evaluate global model on test data."""
    if global_params is None or test_data is None:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
        }

    model = model_template.__class__(model_template.input_shape, model_template.num_classes)
    model.load_state_dict({k: v.cpu() for k, v in global_params.items()})
    model.eval()

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    seen_set = set(int(c) for c in seen_classes) if seen_classes else None
    if seen_set:
        mask = torch.tensor([int(y.item()) in seen_set for y in y_test])
        X_test = X_test[mask]
        y_test = y_test[mask]

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_y = y_test[i:i+batch_size]

            output = model(batch_X)
            if seen_set:
                unseen_mask = torch.ones(output.shape[1], dtype=torch.bool, device=output.device)
                for cls_id in seen_set:
                    if 0 <= cls_id < output.shape[1]:
                        unseen_mask[cls_id] = False
                output = output.clone()
                output[:, unseen_mask] = float("-inf")
            loss = criterion(output, batch_y)
            preds = output.argmax(dim=1)

            total_loss += loss.item() * len(batch_y)
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(batch_y.detach().cpu().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    metrics = {
        "loss": total_loss / max(len(y_test), 1),
        "accuracy": accuracy_score(y_true, y_pred) if len(y_true) else 0.0,
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) else 0.0,
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) else 0.0,
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0) if len(y_true) else 0.0,
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0) if len(y_true) else 0.0,
    }

    return metrics
