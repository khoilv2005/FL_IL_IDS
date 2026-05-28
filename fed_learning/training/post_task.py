"""
Post-task processing for algorithm-specific end-of-task logic.
"""

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset

from fed_learning.strategies.fed_incremental.fedlwf import FedLwFTrainer


def post_task_processing(
    trainer,
    server,
    client_data: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
    participating_clients: Optional[List] = None,
):
    """
    Handle post-task logic after each task finishes training.

    Supported hooks:
    - CGoFed representation-space update
    - EWC Fisher consolidation
    - LwF teacher snapshot
    - FedCBDR/DER exemplar updates
    - NICE end-task age/context update
    - GLFC snapshot/exemplar update
    """
    algo = config["algorithm"].lower()
    device = getattr(server, "primary_device", None) or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if algo == "cgofed":
        if hasattr(trainer, "build_space_from_clients") and participating_clients:
            trainer.build_space_from_clients(
                model=server.global_model,
                clients=participating_clients,
                config=config,
                device=device,
            )
        elif hasattr(trainer, "build_space_from_client_data"):
            trainer.build_space_from_client_data(
                model=server.global_model,
                client_data=client_data,
                config=config,
                device=device,
            )

    elif "ewc" in algo:
        if hasattr(trainer, "consolidate"):
            print("\nComputing Fisher Information for EWC...")
            all_X, all_y = [], []
            for data in client_data.values():
                if len(data.get("y_train", [])) > 0:
                    all_X.append(data["X_train"])
                    all_y.append(data["y_train"])

            if all_X:
                X = torch.cat(all_X)
                y = torch.cat(all_y)
                if len(y) > config.get("fisher_samples", 200):
                    idx = torch.randperm(len(y))[: config["fisher_samples"]]
                    X, y = X[idx], y[idx]

                loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
                trainer.consolidate(server.global_model, loader, device)

    elif isinstance(trainer, FedLwFTrainer) or "lwf" in algo:
        print("\nSaving model snapshot for LwF...")
        if hasattr(trainer, "save_model_snapshot"):
            trainer.save_model_snapshot(server.global_model)
        elif hasattr(server, "save_global_snapshot"):
            server.save_global_snapshot()

    if algo == "fedcbdr" and hasattr(server, "coordinate_gdr"):
        print("\nUpdating Replay Buffers (GDR)...")
        server.coordinate_gdr(participating_clients, verbose=True)

    if algo in ("der", "rne", "rne_compress") and hasattr(server, "coordinate_exemplar_update"):
        label = "RNE-compress" if algo == "rne_compress" else ("RNE" if algo == "rne" else "DER")
        print(f"\n{label}: Updating exemplar buffers...")
        server.coordinate_exemplar_update(participating_clients, verbose=True)

    if algo == "nice" and hasattr(server, "end_task"):
        server.end_task()

    if algo == "glfc":
        if hasattr(trainer, "save_model_snapshot"):
            trainer.save_model_snapshot(server.global_model)
        if hasattr(server, "coordinate_exemplar_update"):
            print("\nUpdating GLFC exemplar sets...")
            server.coordinate_exemplar_update(participating_clients, verbose=True)
        if participating_clients:
            for client in participating_clients:
                if hasattr(client, "save_model_snapshot"):
                    client.save_model_snapshot(server.global_model)
