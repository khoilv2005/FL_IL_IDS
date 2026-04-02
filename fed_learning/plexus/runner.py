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
from typing import Dict, List, Any

import numpy as np
import torch
import torch.nn as nn

from .sampler import PlexusSampler
from .aggregator import PlexusAggregator
from .orchestrator import NodeWrapper


def run_plexus_training(
    node_ids: List[int],
    node_data: Dict[int, tuple],  # node_id -> (X_train, y_train)
    model_template: nn.Module,
    config: Dict[str, Any],
    test_data: Dict = None,
    verbose: bool = True,
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
        sample_size=sample_size,
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
    global_params = None

    # History
    history = {
        "round": [],
        "sample": [],
        "aggregator": [],
        "participation": [],
        "loss": [],
        "round_time": [],
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

        # Step 2: Each sample node trains locally (Algorithm 2 step 1)
        results = []
        for nid in sample_ids:
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

        # Step 4: Aggregator checks threshold and performs FedAvg
        threshold = aggregator.get_threshold()

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

        # Optional eval
        if verbose and test_data is not None and round_r % 5 == 0:
            acc = _evaluate(global_params, nodes[sample_ids[0]].model, test_data, batch_size)
            print(f"   Test accuracy: {acc:.4f}")

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


def _evaluate(
    global_params: OrderedDict,
    model_template: nn.Module,
    test_data: Dict,
    batch_size: int,
) -> float:
    """Evaluate global model on test data."""
    if global_params is None or test_data is None:
        return 0.0

    model = model_template.__class__(model_template.input_shape, model_template.num_classes)
    model.load_state_dict({k: v.cpu() for k, v in global_params.items()})
    model.eval()

    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_X = X_test[i:i+batch_size]
            batch_y = y_test[i:i+batch_size]

            output = model(batch_X)
            preds = output.argmax(dim=1)

            correct += (preds == batch_y).sum().item()
            total += len(batch_y)

    return correct / max(total, 1)
