"""
DFCA Client - Decentralized Federated Clustering Algorithm Client.

Each DFCA client:
- Maintains k cluster models (one per cluster) in cluster_params
- Assigns itself to the cluster with minimum local loss
- Trains only the assigned cluster model
- Exchanges trained models with neighbors and aggregates via running average

Cluster = group of clients with similar data distribution, NOT a class label.
"""

import contextlib
from collections import OrderedDict
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn

try:
    from torch.amp import autocast as torch_autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as torch_autocast, GradScaler

from .client import FederatedClient


class DFCAClient(FederatedClient):
    """
    DFCA client with per-cluster model banks and decentralized aggregation.

    State:
        cluster_params: Dict[cluster_id -> OrderedDict] — the k cluster model banks
        assigned_cluster: int — current cluster assignment
        assignment_losses: Dict[cluster_id -> float] — loss on each cluster model
        current_task: int — current task ID
        seen_classes: set — classes seen so far
        neighbor_messages: Dict[client_id -> Dict[cluster_id -> OrderedDict]] — received messages
        prev_assigned_cluster: int — previous cluster for tracking assignment changes
        _initialized: bool — whether cluster bank has been initialized

    DFCA Three-Step Round:
        1. assign_cluster()     — pick cluster with min local loss
        2. train_assigned_cluster() — train only assigned cluster
        3. export_assigned_cluster_message() + receive_neighbor_message() + aggregate_received_messages()
    """

    def __init__(
        self,
        client_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        num_clusters: int = 10,
        init_seed: int = 42,
    ):
        super().__init__(client_id, X_train, y_train)
        self.num_clusters = num_clusters
        self.init_seed = init_seed

        # Per-cluster model banks: cluster_id -> OrderedDict of params
        self.cluster_params: Dict[int, OrderedDict] = {}

        # Current cluster assignment and losses
        self.assigned_cluster: int = 0
        self.assignment_losses: Dict[int, float] = {}
        self.prev_assigned_cluster: int = 0

        # Task context
        self.current_task: int = 0
        self.seen_classes: set = set()
        self.new_classes: list = []

        # Messages received from neighbors this round
        # neighbor_id -> {cluster_id -> params}
        self.neighbor_messages: Dict[int, Dict[int, OrderedDict]] = {}

        self._initialized: bool = False

    def initialize_cluster_bank(
        self,
        global_params: Optional[OrderedDict] = None,
        template_model: Optional[nn.Module] = None,
    ) -> None:
        """
        Initialize all k cluster model banks.

        DFCA-GI (Global Initialization): all k models start from the same params.
        Each client gets the same initial global params and copies it k times.

        Args:
            global_params: Initial params from server. If None, each cluster
                           is initialized with a deterministic seed (DFCA-LI style).
            template_model: Optional model to copy state dict structure from.
        """
        self._initialized = True

        if global_params is not None:
            # DFCA-GI: same params for all clusters
            for cid in range(self.num_clusters):
                self.cluster_params[cid] = OrderedDict(
                    (k, v.clone()) for k, v in global_params.items()
                )
        elif template_model is not None:
            # DFCA-GI via template model
            for cid in range(self.num_clusters):
                self.cluster_params[cid] = OrderedDict(
                    (k, v.clone()) for k, v in template_model.state_dict().items()
                )
        else:
            # DFCA-LI: each cluster gets different initialization (rarely used)
            for cid in range(self.num_clusters):
                torch.manual_seed(self.init_seed + cid)
                self.cluster_params[cid] = OrderedDict()

    def set_task_data(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        task_id: int,
        new_classes: List[int],
    ) -> None:
        """Update local data and task context for a new task."""
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(y_train)
        self.current_task = task_id
        self.new_classes = list(new_classes)
        self.seen_classes = self.seen_classes | set(new_classes)

    def set_task(self, task_id: int, seen_classes: List[int], new_classes: List[int]) -> None:
        """Set task context for incremental learning."""
        self.current_task = task_id
        self.seen_classes = set(seen_classes)
        self.new_classes = list(new_classes)

    # =======================================================================
    # STEP 1: Cluster Assignment
    # =======================================================================

    def _evaluate_cluster_loss(
        self,
        cluster_params: OrderedDict,
        batch_size: int = 512,
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Evaluate a cluster model's loss on local data.

        Returns:
            (avg_loss, X_batch, y_batch) — loss for the entire local dataset
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Client not set up for GPU. Call setup_for_gpu() first.")

        self.model.eval()

        # Load cluster params
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in cluster_params.items()}
        )

        total_loss = 0.0
        total_samples = 0
        criterion = nn.CrossEntropyLoss(reduction="sum")

        with torch.no_grad():
            indices = torch.randperm(self.num_samples)
            for i in range(0, self.num_samples, batch_size):
                batch_idx = indices[i : i + batch_size]
                X_batch = self.X_train[batch_idx].to(self.device, non_blocking=True)
                y_batch = self.y_train[batch_idx].to(self.device, non_blocking=True)

                out = self.model(X_batch)
                loss = criterion(out, y_batch)
                total_loss += loss.item()
                total_samples += len(y_batch)

        return total_loss / max(1, total_samples)

    def assign_cluster(self, verbose: bool = False) -> int:
        """
        DFCA Step 1: Assign this client to the cluster with minimum local loss.

        Evaluates ALL k cluster models on local data and picks argmin loss.
        This is a LOCAL decision — no server coordination needed.

        Args:
            verbose: If True, print assignment details.

        Returns:
            The assigned cluster ID.
        """
        if not self._initialized:
            raise RuntimeError(
                f"Client {self.client_id}: cluster bank not initialized. "
                "Call initialize_cluster_bank() first."
            )

        self.prev_assigned_cluster = self.assigned_cluster
        self.assignment_losses = {}

        for cid in range(self.num_clusters):
            loss = self._evaluate_cluster_loss(self.cluster_params[cid])
            self.assignment_losses[cid] = loss

        self.assigned_cluster = min(
            self.assignment_losses,
            key=lambda c: self.assignment_losses[c]
        )

        if verbose:
            loss_str = ", ".join(
                f"c{c}={self.assignment_losses[c]:.4f}"
                for c in sorted(self.assignment_losses)
            )
            print(
                f"  Client {self.client_id}: assign_cluster -> "
                f"{self.assigned_cluster} [{loss_str}]"
            )

        return self.assigned_cluster

    # =======================================================================
    # STEP 2: Local Update
    # =======================================================================

    def _get_seen_class_mask(self, num_classes: int) -> Optional[torch.Tensor]:
        """Get a boolean mask for seen classes. Returns None if all classes seen."""
        if not self.seen_classes:
            return None
        seen = sorted(self.seen_classes)
        if len(seen) >= num_classes:
            return None
        mask = torch.ones(num_classes, dtype=torch.bool)
        for cls_id in seen:
            if 0 <= cls_id < num_classes:
                mask[cls_id] = False
        return mask

    def train_assigned_cluster(
        self,
        trainer,
        epochs: int = 1,
        batch_size: int = 2048,
        lr: float = 0.001,
        **kwargs
    ) -> Dict[str, Any]:
        """
        DFCA Step 2: Train ONLY the assigned cluster model on local data.

        Only parameters of cluster `assigned_cluster` are updated.
        All other cluster banks remain unchanged.

        Args:
            trainer: DFCATrainer instance for loss computation and hooks
            epochs: Number of local epochs
            batch_size: Batch size
            lr: Learning rate

        Returns:
            Dict with loss and updated params for the assigned cluster only
        """
        if self.model is None or self.device is None:
            raise RuntimeError("Client not set up for GPU. Call setup_for_gpu() first.")
        if not self._initialized:
            raise RuntimeError("Cluster bank not initialized.")

        cluster_id = self.assigned_cluster
        cluster_state = self.cluster_params[cluster_id]

        # Load assigned cluster params into model
        self.model.load_state_dict(
            {k: v.to(self.device) for k, v in cluster_state.items()}
        )
        self.model.train()

        optimizer_cls = trainer.get_optimizer_class()
        optimizer = optimizer_cls(self.model.parameters(), lr=lr)
        scaler = GradScaler(enabled=self.use_amp)

        trainer.pre_train(self.model, None, lr=lr, **kwargs)

        total_loss = 0.0
        total_samples = 0

        # Track original params of unassigned clusters
        unassigned_params_before: Dict[int, Dict[str, torch.Tensor]] = {}
        for cid in range(self.num_clusters):
            if cid != cluster_id:
                unassigned_params_before[cid] = {
                    k: v.clone().cpu()
                    for k, v in self.cluster_params[cid].items()
                }

        for ep in range(epochs):
            for X_batch, y_batch in self._create_batches(batch_size):
                optimizer.zero_grad()

                with self._amp_ctx():
                    out = self.model(X_batch)
                    loss = trainer.compute_loss(
                        self.model, out, y_batch, None, **kwargs
                    )

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    trainer.pre_step(self.model, None, **kwargs)
                    scaler.step(optimizer)
                    scaler.update()
                    trainer.post_step(self.model, None, **kwargs)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    trainer.pre_step(self.model, None, **kwargs)
                    optimizer.step()
                    trainer.post_step(self.model, None, **kwargs)

                bs = len(y_batch)
                total_loss += loss.item() * bs
                total_samples += bs

        trainer.post_train(self.model, None, **kwargs)

        # Extract updated params for assigned cluster
        updated_params = OrderedDict(
            (k, v.cpu().clone()) for k, v in self.model.state_dict().items()
        )
        self.cluster_params[cluster_id] = updated_params

        # Verify unassigned clusters are unchanged
        for cid in range(self.num_clusters):
            if cid != cluster_id:
                for k in self.cluster_params[cid]:
                    if not torch.equal(self.cluster_params[cid][k], unassigned_params_before[cid][k]):
                        # Restore if accidentally changed (shouldn't happen with correct optimizer)
                        self.cluster_params[cid] = OrderedDict(
                            (k2, v.clone()) for k2, v in unassigned_params_before[cid].items()
                        )
                        break

        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "assigned_cluster": cluster_id,
            "loss": total_loss / max(1, total_samples),
            "params": updated_params,
        }

    # =======================================================================
    # STEP 3: Decentralized Aggregation (peer-to-peer)
    # =======================================================================

    def receive_neighbor_message(self, sender_id: int, message: Dict[int, OrderedDict]) -> None:
        """
        Receive a message from a neighbor.

        Args:
            sender_id: ID of the sending client
            message: Dict mapping cluster_id -> params from that neighbor
        """
        self.neighbor_messages[sender_id] = message

    def aggregate_received_messages(self) -> None:
        """
        DFCA Step 3: Sequential running average across received messages.

        For each cluster j, updates cluster_params[j] by incorporating
        params from all neighbors that were assigned to cluster j.

        Formula (per incoming neighbor m for cluster j):
            theta_i,j = ((r+1)/(r+2)) * theta_i,j + (1/(r+2)) * theta_m,j

        Where r = number of neighbors already incorporated.

        Only neighbors assigned to cluster j contribute to theta_i,j.
        This is unweighted averaging (paper: each peer equally weighted).
        """
        for cluster_id in range(self.num_clusters):
            # Collect params from neighbors assigned to this cluster
            contributors = {}
            for sender_id, msg in self.neighbor_messages.items():
                if cluster_id in msg:
                    contributors[sender_id] = msg[cluster_id]

            if not contributors:
                continue

            # Sequential running average: unweighted per peer
            r = 0  # number of neighbors already incorporated
            for sender_id, params in contributors.items():
                for key in self.cluster_params[cluster_id]:
                    local_tensor = self.cluster_params[cluster_id][key]
                    incoming_tensor = params[key].to(local_tensor.device)

                    if local_tensor.dtype.is_floating_point and incoming_tensor.dtype.is_floating_point:
                        alpha = (r + 1.0) / (r + 2.0)
                        beta = 1.0 / (r + 2.0)
                        self.cluster_params[cluster_id][key] = (
                            alpha * local_tensor + beta * incoming_tensor
                        )
                    # Non-floating-point (BN stats) are replaced, not averaged

                r += 1

        # Clear received messages for next round
        self.neighbor_messages = {}

    def export_assigned_cluster_message(self) -> Dict[int, OrderedDict]:
        """
        Export this client's trained params for the assigned cluster.

        Returns:
            {assigned_cluster_id -> params} — only the trained cluster's params.
            The caller (server) will distribute this to neighbors.
        """
        return {
            self.assigned_cluster: self.cluster_params[self.assigned_cluster]
        }

    def export_all_cluster_params(self) -> Dict[int, OrderedDict]:
        """Export all k cluster params (for debugging or evaluation)."""
        return {cid: params.copy() for cid, params in self.cluster_params.items()}

    def load_cluster_params_from_server(self, cluster_params: Dict[int, OrderedDict]) -> None:
        """Load all cluster params from server (for initialization or resync)."""
        for cid, params in cluster_params.items():
            self.cluster_params[cid] = OrderedDict((k, v.clone()) for k, v in params.items())
        self._initialized = True

    def get_cluster_params(self, cluster_id: int) -> Optional[OrderedDict]:
        """Get params for a specific cluster."""
        return self.cluster_params.get(cluster_id)

    def get_cluster_assignment(self) -> int:
        """Get current cluster assignment."""
        return self.assigned_cluster

    def get_assignment_losses(self) -> Dict[int, float]:
        """Get loss values for all clusters (from last assignment)."""
        return dict(self.assignment_losses)
