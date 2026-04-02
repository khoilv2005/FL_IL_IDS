"""
Post-task processing - Handle algorithm-specific end-of-task logic.

After each task's training completes, different algorithms need specific
operations: Fisher computation (EWC), model snapshots (LwF, GLFC),
SVD space building (CGoFed), buffer updates (FedCBDR, DER), etc.
"""

from typing import Dict, Any, List, Optional

import torch
from torch.utils.data import TensorDataset, DataLoader

from fed_learning.strategies.fed_incremental.fedlwf import FedLwFTrainer


def post_task_processing(
    trainer,
    server,
    client_data: Dict[int, Dict[str, Any]],
    config: Dict[str, Any],
    participating_clients: Optional[List] = None,
):
    """
    Handle post-task logic (Fisher, Snapshot, SVD, Buffer Update).

    Called after training completes for each task. Dispatches to the
    appropriate algorithm-specific post-processing.

    Args:
        trainer: The active trainer instance (subclass of BaseTrainer)
        server: The active server instance (subclass of IncrementalServer)
        client_data: Dict mapping client_id -> {"X_train": tensor, "y_train": tensor}
        config: Full training configuration
        participating_clients: List of client instances that participated in this task
    """
    algo = config["algorithm"].lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. CGoFed: Build Representation Space via SVD
    if algo == "cgofed":
        if hasattr(trainer, "build_space_from_client_data"):
            trainer.build_space_from_client_data(
                model=server.global_model,
                client_data=client_data,
                config=config,
                device=device,
            )

    # 2. EWC: Consolidate (Compute Fisher Information)
    elif "ewc" in algo:  # fedavg_ewc, fedprox_ewc
        if hasattr(trainer, "consolidate"):
            print(f"\n🔐 Computing Fisher Information for EWC...")
            # Aggregate data for Fisher computation
            all_X, all_y = [], []
            for data in client_data.values():
                if len(data.get("y_train", [])) > 0:
                    all_X.append(data["X_train"])
                    all_y.append(data["y_train"])

            if all_X:
                X = torch.cat(all_X)
                y = torch.cat(all_y)
                # Limit samples for efficiency
                if len(y) > config.get("fisher_samples", 200):
                    idx = torch.randperm(len(y))[: config["fisher_samples"]]
                    X, y = X[idx], y[idx]

                loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
                trainer.consolidate(server.global_model, loader, device)

    # 3. LwF: Save Model Snapshot for next task's distillation
    elif isinstance(trainer, FedLwFTrainer) or "lwf" in algo:
        print(f"\n📸 Saving model snapshot for LwF...")
        if hasattr(trainer, "save_model_snapshot"):
            trainer.save_model_snapshot(server.global_model)
        elif hasattr(server, "save_global_snapshot"):
            server.save_global_snapshot()

    # 4. FedCBDR: GDR Update (Gradient-based Data Replay)
    if algo == "fedcbdr" and hasattr(server, "coordinate_gdr"):
        print(f"\n🔄 Updating Replay Buffers (GDR)...")
        server.coordinate_gdr(participating_clients, verbose=True)

    # 5. DER: Exemplar Buffer Update (herding selection)
    if algo == "der" and hasattr(server, "coordinate_exemplar_update"):
        print(f"\n📸 DER: Updating exemplar buffers...")
        server.coordinate_exemplar_update(participating_clients, verbose=True)

    # 6. NICE: End-task processing (age transition, freeze masks, context detector)
    if algo == "nice" and hasattr(server, "end_task"):
        server.end_task()

    # 7. GLFC: Save model snapshot and coordinate exemplar updates
    if algo == "glfc":
        if hasattr(trainer, "save_model_snapshot"):
            trainer.save_model_snapshot(server.global_model)
        if hasattr(server, "coordinate_exemplar_update"):
            print(f"\n  Updating GLFC exemplar sets...")
            server.coordinate_exemplar_update(participating_clients, verbose=True)
        # Save snapshot to all clients for next task's distillation
        if participating_clients:
            for client in participating_clients:
                if hasattr(client, "save_model_snapshot"):
                    client.save_model_snapshot(server.global_model)

    # 8. Re-Fed: No special post-task processing needed
    # PIM caching is done at the START of each new task (before training)

    # 9. PlexusDER: Exemplar Buffer Update (herding selection)
    if algo == "plexus_der" and hasattr(server, "coordinate_exemplar_update"):
        print(f"\n📸 PlexusDER: Updating exemplar buffers...")
        server.coordinate_exemplar_update(participating_clients, verbose=True)

    # 10. PlexusNICE: End-task processing (age transition, freeze masks)
    if algo == "plexus_nice" and hasattr(server, "end_task"):
        server.end_task()
