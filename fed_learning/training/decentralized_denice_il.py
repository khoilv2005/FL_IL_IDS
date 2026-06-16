"""Decentralized DeNICE-IL runner.

This wires Phase 3 of ``DeNICE_micro_adapter_implementation_plan.md`` into a
real no-server simulation:

- every client owns its DeNICE model and context detector;
- clients build NICE Context Capsules after local training;
- Dynamic-K context clustering forms collaboration groups;
- age-aware masked aggregation updates only compatible parameters;
- adapter parameters are averaged only when adapter ids/shapes match.

The runner is intentionally sequential for reproducibility and low complexity.
It is a protocol runner, not a central FedAvg server: the coordinator only
simulates communication rounds and writes artifacts.
"""

from __future__ import annotations

import json
import os
import time
from collections import OrderedDict
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from fed_learning.data.incremental_loader import IncrementalDataLoader
from fed_learning.factories.client_factory import create_client, update_client_data
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.strategies.decentralized import (
    AggregationConfig,
    SimilarityWeights,
    age_aware_aggregate,
    aggregate_adapters,
    aggregation_weights,
    build_context_capsule,
    collaboration_group,
    context_similarity,
    dynamic_ap_cluster,
    merge_neuron_ages,
)
from fed_learning.strategies.fed_incremental.nice import (
    increase_unit_ranks,
    update_freeze_masks,
)
from fed_learning.strategies.incremental import get_incremental_strategy
from fed_learning.strategies.incremental.denice_capacity import (
    CANCConfig,
    CapacityController,
    compute_capacity_state,
    compute_consumption,
)
from fed_learning.strategies.incremental.denice_novelty import NoveltyEstimator
from fed_learning.strategies.incremental.denice_recycling import (
    apply_graceful_recycling,
    revive_due_recycled_neurons,
)
from fed_learning.training.denice_eval import evaluate_denice_model
from fed_learning.training.denice_usage import compute_adapter_usage
from fed_learning.training.checkpoint_state import (
    CHECKPOINT_SCHEMA_VERSION,
    snapshot_denice_state,
)
from fed_learning.training.local_task_loop import (
    _sample_denice_reference,
    _update_local_nice_context_memory,
)
from fed_learning.training.task_loop import _resolve_output_dir
from fed_learning.utils.seed import set_seed


def _state_dict(model: torch.nn.Module) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items())

def _cpu_state_dict(model: torch.nn.Module) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
    )

def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, OrderedDict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value

def _write_json(path: str, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, default=str)


def _delta_to_target(
    target: "OrderedDict[str, torch.Tensor]",
    neighbor: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    delta = OrderedDict()
    for key, value in target.items():
        other = neighbor.get(key)
        if other is None or other.shape != value.shape:
            continue
        delta[key] = other.to(value.device) - value
    return delta


def _label_histogram(y: torch.Tensor) -> Dict[int, float]:
    if y is None or len(y) == 0:
        return {}
    labels, counts = torch.unique(y.detach().cpu(), return_counts=True)
    total = float(counts.sum().item())
    if total <= 0:
        return {}
    return {int(c): float(n.item() / total) for c, n in zip(labels, counts)}


def _seen_classes(data_loader: IncrementalDataLoader, task_id: int) -> List[int]:
    seen: List[int] = []
    for tid in range(task_id + 1):
        seen.extend(int(c) for c in data_loader.get_task_classes(tid))
    return sorted(set(seen))

def _round_checkpoint_path(output_dir: str, task_id: int, round_id: int) -> str:
    return os.path.join(output_dir, f"checkpoint_task_{task_id}_round_{round_id}.pt")

def _client_algorithm_states(
    client_ids: List[int],
    models: Dict[int, DeNICEModel],
    context_detectors: Dict[int, ContextDetector],
) -> Dict[int, Dict[str, Any]]:
    return {
        int(cid): snapshot_denice_state(models[cid], context_detectors.get(cid))
        for cid in client_ids
    }

def _adapter_states(model: DeNICEModel) -> Dict[str, "OrderedDict[str, torch.Tensor]"]:
    return {
        key: OrderedDict((k, v.detach().clone()) for k, v in adapter.state_dict().items())
        for key, adapter in getattr(model, "adapters", {}).items()
    }

def _inject_adapter_states(
    full_state: "OrderedDict[str, torch.Tensor]",
    adapter_states: Dict[str, "OrderedDict[str, torch.Tensor]"],
) -> None:
    for adapter_id, state in adapter_states.items():
        for name, value in state.items():
            full_state[f"adapters.{adapter_id}.{name}"] = value.detach().clone()

def _save_round_artifacts(
    *,
    output_dir: str,
    task_id: int,
    round_id: int,
    client_ids: List[int],
    models: Dict[int, DeNICEModel],
    context_detectors: Dict[int, ContextDetector],
    capsules: Dict[int, Any],
    cluster_summary: Dict[str, Any],
    adapter_usage: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
    seen_classes: List[int],
    round_record: Dict[str, Any],
    save_checkpoint: bool,
) -> None:
    _write_json(
        os.path.join(output_dir, f"context_capsule_task_{task_id}_round_{round_id}.json"),
        {"task": task_id, "round": round_id, "clients": capsules},
    )
    _write_json(
        os.path.join(output_dir, f"cluster_snapshot_task_{task_id}_round_{round_id}.json"),
        cluster_summary,
    )
    _write_json(
        os.path.join(output_dir, f"adapter_registry_task_{task_id}_round_{round_id}.json"),
        {"task": task_id, "round": round_id, "clients": adapter_usage},
    )

    if not save_checkpoint:
        return

    ckpt_path = _round_checkpoint_path(output_dir, task_id, round_id)
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "mode": "decentralized",
            "algorithm": "denice",
            "task_id": int(task_id),
            "round_id": int(round_id),
            "client_model_states": {
                int(cid): _cpu_state_dict(models[cid]) for cid in client_ids
            },
            "client_algorithm_states": _client_algorithm_states(
                client_ids, models, context_detectors
            ),
            "config": config,
            "seen_classes": list(seen_classes),
            "cluster": _json_safe(cluster_summary),
            "metrics": _json_safe(round_record),
        },
        ckpt_path,
    )
    print(f"   Round checkpoint saved: {ckpt_path}")

def _write_phase_outputs(
    output_dir: str,
    history: Dict[str, Any],
    cluster_history: List[Dict[str, Any]],
    adapter_history: List[Dict[str, Any]],
    config: Dict[str, Any],
    completed_task_id: int,
) -> None:
    _write_json(os.path.join(output_dir, "training_history.json"), history)
    _write_json(os.path.join(output_dir, "results.json"), history)
    _write_json(os.path.join(output_dir, "round_metrics.json"), history.get("round_metrics", []))
    _write_json(os.path.join(output_dir, "task_metrics.json"), history.get("task_accuracies", []))
    _write_json(os.path.join(output_dir, "cluster_history.json"), cluster_history)
    _write_json(os.path.join(output_dir, "denice_adapter_usage_summary.json"), {"clients": adapter_history})
    _write_json(os.path.join(output_dir, "all_task_metrics.json"), history.get("task_accuracies", []))
    _write_json(os.path.join(output_dir, "all_cluster_metrics.json"), cluster_history)
    _write_json(os.path.join(output_dir, "all_adapter_metrics.json"), adapter_history)
    _write_json(
        os.path.join(output_dir, "phase_summary.json"),
        {
            "algorithm": "denice",
            "mode": "decentralized",
            "task_start": int(config.get("task_start", 0)),
            "task_end": int(config.get("task_end", completed_task_id)),
            "completed_task": int(completed_task_id),
            "num_task_records": len(history.get("task_accuracies", [])),
            "num_round_records": len(history.get("round_metrics", [])),
            "num_cluster_records": len(cluster_history),
            "num_adapter_records": len(adapter_history),
        },
    )
    _write_json(
        os.path.join(output_dir, "final_report.json"),
        {
            "algorithm": "denice",
            "mode": "decentralized",
            "completed_task": int(completed_task_id),
            "final_metrics": (
                history.get("task_accuracies", [{}])[-1]
                if history.get("task_accuracies")
                else {}
            ),
        },
    )


def _make_model(config: Dict[str, Any], device: torch.device) -> DeNICEModel:
    return DeNICEModel(config["input_shape"], config["num_classes"]).to(device)


def _prepare_client_task(
    *,
    cid: int,
    task_id: int,
    num_tasks: int,
    new_classes: List[int],
    model: DeNICEModel,
    client,
    trainer,
    config: Dict[str, Any],
    device: torch.device,
    context_detector: ContextDetector,
    novelty_estimator: NoveltyEstimator,
    prev_ages: Optional[Dict[str, np.ndarray]],
) -> Dict[str, Any]:
    """Run DeNICE prepare_task / novelty / CANC / adapter activation."""
    if hasattr(trainer, "set_task"):
        trainer.set_task(task_id, new_classes)

    for cls_id in new_classes:
        if 0 <= int(cls_id) < model.num_classes:
            model.unit_ranks["fc2"][int(cls_id)] = 1
    context_detector.episode_classes[task_id] = list(int(c) for c in new_classes)

    per_class = max(1, int(config.get("nice_memo_per_class", config.get("memo_per_class", 50))))
    ref_data = _sample_denice_reference(
        client.X_train, client.y_train, list(new_classes), per_class, str(device)
    )
    revived = revive_due_recycled_neurons(model, task_id, config)

    is_first_task = not novelty_estimator.has_history()
    novelty = 0.0
    if ref_data.numel() > 0:
        model.eval()
        if novelty_estimator.thresholds is None:
            novelty_estimator.calibrate_thresholds(model, ref_data)
        if not is_first_task:
            novelty = float(novelty_estimator.compute_novelty(model, ref_data)["novelty"])

    start_ages = model.get_neuron_ages_state()
    capacity_state = compute_capacity_state(model)
    consumption = compute_consumption(prev_ages, start_ages)
    controller = CapacityController(CANCConfig.from_dict(config))
    plan = controller.plan_task(
        capacity_state,
        novelty,
        consumption,
        is_first_task=is_first_task,
    )
    plan["novelty"] = novelty
    plan["start_ages"] = start_ages
    plan["revived"] = revived

    model.clear_active_adapters()
    for layer in plan["adapters_to_add"]:
        model.add_adapter(task_id, layer, set_active=True)
    apply_graceful_recycling(model, ref_data, task_id, plan, config)

    if plan["freeze_low_layers"]:
        for low in ("conv1", "conv2"):
            ranks = model.unit_ranks.get(low)
            if ranks is not None:
                model.freeze_masks[low] = np.ones(len(ranks), dtype=bool)

    return {"plan": plan, "ref_data": ref_data}


def _build_round_capsule(
    *,
    cid: int,
    task_id: int,
    round_id: int,
    model: DeNICEModel,
    client,
    context_detector: ContextDetector,
    ref_data: torch.Tensor,
    loss: float,
) -> Any:
    reliability = 1.0 / (1.0 + max(0.0, float(loss)))
    labels = sorted(set(int(c) for c in client.y_train.detach().cpu().tolist()))
    sample_count = int(len(client.y_train))
    thresholds = getattr(context_detector, "binarize_thresholds", None)
    device = next(model.parameters()).device
    capsule_data = ref_data if ref_data.numel() > 0 else client.X_train[: min(32, len(client.y_train))]
    capsule_data = capsule_data.to(device)
    return build_context_capsule(
        model,
        capsule_data,
        client_id=int(cid),
        task_id=int(task_id),
        round_id=int(round_id),
        label_histogram=_label_histogram(client.y_train),
        label_set=labels,
        sample_count=sample_count,
        reliability=reliability,
        thresholds=thresholds,
        context_detector=context_detector,
    )


def _aggregate_round(
    *,
    client_ids: List[int],
    models: Dict[int, DeNICEModel],
    capsules: Dict[int, Any],
    config: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Cluster capsules and apply age-aware decentralized aggregation."""
    ordered_caps = [capsules[cid] for cid in client_ids]
    cluster_result = dynamic_ap_cluster(ordered_caps)
    labels = cluster_result["labels"]
    sim_weights = SimilarityWeights()
    agg_config = AggregationConfig(
        eta=float(config.get("denice_aggregation_eta", 1.0)),
        protect_mature=bool(config.get("denice_protect_mature", True)),
    )

    old_states = {cid: _state_dict(models[cid]) for cid in client_ids}
    old_ages = {cid: models[cid].get_neuron_ages_state() for cid in client_ids}
    old_adapter_states = {cid: _adapter_states(models[cid]) for cid in client_ids}
    new_states: Dict[int, OrderedDict] = {}
    new_ages: Dict[int, Dict[str, np.ndarray]] = {}
    groups: Dict[int, List[int]] = {}

    for idx, cid in enumerate(client_ids):
        group_indices = collaboration_group(idx, labels)
        group_ids = [client_ids[g] for g in group_indices]
        groups[int(cid)] = [int(x) for x in group_ids]

        sims = []
        counts = []
        rels = []
        for gid in group_ids:
            if gid == cid:
                sims.append(1.0)
            else:
                sims.append(max(0.0, context_similarity(capsules[cid], capsules[gid], sim_weights)))
            counts.append(float(capsules[gid].sample_count))
            rels.append(float(capsules[gid].reliability))
        self_index = group_ids.index(cid)
        alphas = aggregation_weights(sims, counts, rels, self_index=self_index)

        target_state = old_states[cid]
        deltas = [_delta_to_target(target_state, old_states[gid]) for gid in group_ids]
        new_states[cid] = age_aware_aggregate(
            target_state,
            old_ages[cid],
            deltas,
            alphas,
            agg_config,
        )
        neighbor_adapter_states = []
        neighbor_adapter_weights = []
        for pos, gid in enumerate(group_ids):
            if gid == cid:
                continue
            neighbor_adapter_states.append(old_adapter_states[gid])
            neighbor_adapter_weights.append(float(alphas[pos]))
        merged_adapters = aggregate_adapters(
            old_adapter_states[cid],
            neighbor_adapter_states,
            neighbor_adapter_weights,
        )
        _inject_adapter_states(new_states[cid], merged_adapters)
        new_ages[cid] = merge_neuron_ages(
            old_ages[cid],
            [old_ages[gid] for gid in group_ids if gid != cid],
        )

    for cid in client_ids:
        models[cid].load_state_dict(new_states[cid], strict=False)
        models[cid].set_neuron_ages_state(new_ages[cid])
        update_freeze_masks(models[cid])
        if hasattr(models[cid], "freeze_bn_for_mature"):
            models[cid].freeze_bn_for_mature()
        models[cid].to(device)

    label_map = {int(cid): int(labels[i]) for i, cid in enumerate(client_ids)}
    return {
        "K_t": int(cluster_result["K_t"]),
        "silhouette": (
            None
            if not np.isfinite(cluster_result["silhouette"])
            else float(cluster_result["silhouette"])
        ),
        "valid": bool(cluster_result["valid"]),
        "labels": label_map,
        "groups": groups,
    }


def _evaluate_clients(
    *,
    client_ids: List[int],
    models: Dict[int, DeNICEModel],
    context_detectors: Dict[int, ContextDetector],
    test_data: Dict[str, torch.Tensor],
    seen_classes: List[int],
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    metrics = []
    for cid in client_ids:
        metrics.append(
            evaluate_denice_model(
                models[cid],
                test_data,
                device=str(device),
                context_detector=context_detectors[cid],
                seen_classes=seen_classes,
                batch_size=batch_size,
            )
        )
    if not metrics:
        return {
            "loss": 0.0,
            "accuracy": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
        }
    keys = metrics[0].keys()
    return {key: float(np.mean([m[key] for m in metrics])) for key in keys}


def run_decentralized_denice_il(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run DeNICE with decentralized capsule clustering and age-aware aggregation."""
    set_seed(config.get("random_seed", config.get("seed", 42)))
    output_dir = _resolve_output_dir(config, "decentralized", "denice")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("DECENTRALIZED DeNICE-IL")
    print("=" * 80)

    data_loader = IncrementalDataLoader(data_dir=config["data_dir"])
    config["input_shape"] = data_loader.input_shape
    config["num_classes"] = config["total_classes"]
    _write_json(os.path.join(output_dir, "config.json"), config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = get_incremental_strategy(
        "denice",
        **{k: v for k, v in config.items() if k != "algorithm"},
    )
    rounds_per_task = max(1, int(config.get("rounds_per_task", 5)))
    eval_every = max(1, int(config.get("eval_every", rounds_per_task)))
    checkpoint_every = config.get("round_checkpoint_every", 1)
    if checkpoint_every is not None:
        checkpoint_every = max(1, int(checkpoint_every))
    batch_size = int(config.get("batch_size", 128))
    eval_batch_size = int(config.get("eval_batch_size", 8192))
    max_clients = config.get("denice_max_clients")
    max_clients = None if max_clients is None else int(max_clients)

    models: Dict[int, DeNICEModel] = {}
    clients: Dict[int, Any] = {}
    context_detectors: Dict[int, ContextDetector] = {}
    novelty_estimators: Dict[int, NoveltyEstimator] = {}
    prev_ages: Dict[int, Optional[Dict[str, np.ndarray]]] = {}
    ref_data: Dict[int, torch.Tensor] = {}
    canc_plans: Dict[int, Dict[str, Any]] = {}

    history = {"task_accuracies": [], "task_forgetting": [], "round_metrics": []}
    cluster_history: List[Dict[str, Any]] = []
    adapter_history: List[Dict[str, Any]] = []

    num_tasks = data_loader.get_num_tasks()
    task_start = int(config.get("task_start", 0))
    task_end = int(config.get("task_end", num_tasks - 1))

    for task_id in range(task_start, task_end + 1):
        print(f"\n{'=' * 80}\nTASK {task_id}/{num_tasks - 1} - Decentralized DeNICE-IL\n{'=' * 80}")
        new_classes = data_loader.get_task_classes(task_id)
        active_ids = []
        for cid in data_loader.get_all_client_ids():
            X, y = data_loader.get_client_data(cid, task_id)
            if len(y) > 0:
                active_ids.append(int(cid))
                data = {"X_train": X, "y_train": y}
                if cid not in clients:
                    clients[cid] = create_client(cid, X, y, {**config, "algorithm": "denice"})
                else:
                    update_client_data(clients[cid], data, task_id, new_classes)
                if cid not in models:
                    models[cid] = _make_model(config, device)
                    context_detectors[cid] = ContextDetector(
                        memo_per_class=int(config.get("nice_memo_per_class", config.get("memo_per_class", 50)))
                    )
                    novelty_estimators[cid] = NoveltyEstimator(
                        layer_weights=getattr(trainer, "novelty_layer_weights", None)
                    )
                    prev_ages[cid] = None

        active_ids = sorted(active_ids)
        if max_clients is not None:
            active_ids = active_ids[:max_clients]
        print(f"  Active DeNICE clients: {len(active_ids)}")
        if not active_ids:
            continue

        for cid in active_ids:
            prep = _prepare_client_task(
                cid=cid,
                task_id=task_id,
                num_tasks=num_tasks,
                new_classes=new_classes,
                model=models[cid],
                client=clients[cid],
                trainer=trainer,
                config=config,
                device=device,
                context_detector=context_detectors[cid],
                novelty_estimator=novelty_estimators[cid],
                prev_ages=prev_ages.get(cid),
            )
            canc_plans[cid] = prep["plan"]
            ref_data[cid] = prep["ref_data"]

        for round_id in range(rounds_per_task):
            print(f"  Round {round_id}/{rounds_per_task - 1}")
            start = time.time()
            losses: Dict[int, float] = {}
            for cid in active_ids:
                model = models[cid]
                client = clients[cid]
                client.setup_for_gpu(model, str(device))
                result = client.train(
                    trainer=trainer,
                    epochs=max(1, int(config.get("nice_phase_epochs", 1))),
                    batch_size=batch_size,
                    lr=float(config.get("learning_rate", 0.001)),
                    global_params=None,
                    is_last_task=(task_id == num_tasks - 1),
                    phase_offset=round_id,
                    max_phases_override=1,
                )
                losses[cid] = float((result or {}).get("loss", 0.0))
                _update_local_nice_context_memory(
                    context_detectors[cid],
                    model,
                    client.X_train,
                    client.y_train,
                    task_id,
                    new_classes,
                    str(device),
                )

            capsules = {
                cid: _build_round_capsule(
                    cid=cid,
                    task_id=task_id,
                    round_id=round_id,
                    model=models[cid],
                    client=clients[cid],
                    context_detector=context_detectors[cid],
                    ref_data=ref_data[cid],
                    loss=losses[cid],
                )
                for cid in active_ids
            }
            cluster_summary = _aggregate_round(
                client_ids=active_ids,
                models=models,
                capsules=capsules,
                config=config,
                device=device,
            )
            cluster_summary.update({"task": task_id, "round": round_id})
            cluster_history.append(cluster_summary)

            round_record = {
                "task": task_id,
                "round": round_id,
                "train_loss": float(np.mean(list(losses.values()))),
                "round_time": time.time() - start,
                "test_loss": None,
                "accuracy": None,
                "precision_macro": None,
                "recall_macro": None,
                "f1_macro": None,
                "f1_weighted": None,
                "avg_forgetting": None,
                "evaluated": False,
                "num_clients": len(active_ids),
                "K_t": cluster_summary["K_t"],
                "silhouette": cluster_summary["silhouette"],
            }
            history["round_metrics"].append(round_record)

            if (round_id + 1) % eval_every == 0 and round_id != rounds_per_task - 1:
                test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)
                metrics = _evaluate_clients(
                    client_ids=active_ids,
                    models=models,
                    context_detectors=context_detectors,
                    test_data={"X_test": test_X, "y_test": test_y},
                    seen_classes=_seen_classes(data_loader, task_id),
                    batch_size=eval_batch_size,
                    device=device,
                )
                round_record.update(
                    {
                        "test_loss": metrics.get("loss"),
                        "accuracy": metrics.get("accuracy"),
                        "precision_macro": metrics.get("precision_macro"),
                        "recall_macro": metrics.get("recall_macro"),
                        "f1_macro": metrics.get("f1_macro"),
                        "f1_weighted": metrics.get("f1_weighted"),
                        "evaluated": True,
                    }
                )
                print(
                    f"    eval accuracy={metrics['accuracy'] * 100:.2f}% "
                    f"f1={metrics['f1_macro'] * 100:.2f}% K={cluster_summary['K_t']}"
                )
            else:
                print(
                    f"    Metrics skipped -> train_loss={round_record['train_loss']:.4f}, "
                    f"eval_every={eval_every}"
                )

            save_round_checkpoint = checkpoint_every is not None and (
                round_id == rounds_per_task - 1 or ((round_id + 1) % checkpoint_every == 0)
            )
            _save_round_artifacts(
                output_dir=output_dir,
                task_id=task_id,
                round_id=round_id,
                client_ids=active_ids,
                models=models,
                context_detectors=context_detectors,
                capsules=capsules,
                cluster_summary=cluster_summary,
                adapter_usage={
                    int(cid): compute_adapter_usage(models[cid]) for cid in active_ids
                },
                config=config,
                seen_classes=_seen_classes(data_loader, task_id),
                round_record=round_record,
                save_checkpoint=save_round_checkpoint,
            )

        for cid in active_ids:
            model = models[cid]
            increase_unit_ranks(model)
            update_freeze_masks(model)
            if hasattr(model, "freeze_bn_for_mature"):
                model.freeze_bn_for_mature()
            if ref_data[cid].numel() > 0:
                proto = novelty_estimators[cid].compute_prototype(model, ref_data[cid])
                novelty_estimators[cid].store_prototype(task_id, proto)
            prev_ages[cid] = canc_plans[cid].get("start_ages", model.get_neuron_ages_state())
            adapter_history.append(
                {
                    "task": task_id,
                    "client_id": int(cid),
                    "canc": canc_plans[cid],
                    **compute_adapter_usage(model),
                }
            )
            model.clear_active_adapters()

        test_X, test_y = data_loader.get_test_data(task_id, cumulative=True)
        metrics = _evaluate_clients(
            client_ids=active_ids,
            models=models,
            context_detectors=context_detectors,
            test_data={"X_test": test_X, "y_test": test_y},
            seen_classes=_seen_classes(data_loader, task_id),
            batch_size=eval_batch_size,
            device=device,
        )
        final_round_id = rounds_per_task - 1
        history["task_accuracies"].append(
            {"task": task_id, "final_round": final_round_id, **metrics, "avg_forgetting": None}
        )
        history["task_forgetting"].append({"task": task_id, "avg_forgetting": None})
        print(
            "  Task summary -> "
            f"accuracy={metrics['accuracy'] * 100:.2f}%, "
            f"f1={metrics['f1_macro'] * 100:.2f}%"
        )

        torch.save(
            {
                "task": task_id,
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "mode": "decentralized",
                "algorithm": "denice",
                "final_round_id": final_round_id,
                "client_model_states": {
                    int(cid): _cpu_state_dict(models[cid])
                    for cid in active_ids
                },
                "client_algorithm_states": _client_algorithm_states(
                    active_ids, models, context_detectors
                ),
                "client_neuron_ages": {
                    int(cid): models[cid].get_neuron_ages_state()
                    for cid in active_ids
                },
                "config": config,
                "seen_classes": _seen_classes(data_loader, task_id),
                "metrics": metrics,
            },
            os.path.join(output_dir, f"checkpoint_task_{task_id}.pt"),
        )
        _write_phase_outputs(
            output_dir, history, cluster_history, adapter_history, config, task_id
        )

    return {
        "output_dir": output_dir,
        "history": history,
        "cluster_history": cluster_history,
        "adapter_history": adapter_history,
    }
