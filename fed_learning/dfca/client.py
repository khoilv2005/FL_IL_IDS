"""
DFCANode - Pure DFCA Client (no Incremental Learning concepts).

Each node:
- Maintains k cluster models (theta_i,1 ... theta_i,k)
- Assigns itself to cluster with minimum local loss (Step 1)
- Trains only the assigned cluster (Step 2)
- Sends trained cluster model to neighbors
- Receives and aggregates cluster models from neighbors (Step 3)
"""

import contextlib
import random
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from torch.amp import autocast as torch_autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as torch_autocast, GradScaler


class DFCANode:
    """
    Pure DFCA node with no incremental-learning concepts.

    State:
        client_id: int
        X_train, y_train: torch.Tensor (CPU)
        cluster_params: Dict[cluster_id -> OrderedDict of params]
        assigned_cluster: int — current cluster assignment c(i)
        assignment_losses: Dict[cluster_id -> float]
        neighbors: List[int] — neighbor client IDs
        received_messages: Dict[sender_id -> Dict[cluster_id -> params]]
        model: Optional[nn.Module] — per-node model instance (not shared)
        device: str
        num_clusters: int
        num_classes: int
        _initialized: bool

    Paper Algorithm 1 round steps per client:
        Step 1: c(i) = argmin_j F_client(theta_i,j, D_i)
        Step 2: Train only theta_i,c(i) via local SGD
        Step 3: Send theta_i,c(i) to neighbors; receive and aggregate
    """

    def __init__(
        self,
        client_id: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        num_clusters: int = 10,
        num_classes: int = 34,
        init_seed: int = 42,
    ):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(y_train)
        self.num_clusters = num_clusters
        self.num_classes = num_classes
        self.init_seed = init_seed

        self.cluster_params: Dict[int, OrderedDict] = {}
        self.assigned_cluster: int = 0
        self.assignment_losses: Dict[int, float] = {}
        self.neighbors: List[int] = []
        self.received_messages: Dict[int, Dict[int, OrderedDict]] = {}
        self.model: Optional[nn.Module] = None
        self.device: str = "cpu"
        self.use_amp: bool = False
        self._initialized: bool = False

    # =======================================================================
    # Initialization
    # =======================================================================

    def initialize_cluster_bank(
        self,
        global_params: Optional[OrderedDict] = None,
        template_model: Optional[nn.Module] = None,
        init_type: str = "global",
    ) -> None:
        """
        Initialize all k cluster model banks.

        Args:
            global_params: If provided, all clusters start from these params (DFCA-GI).
            template_model: Required for "local" init to get state_dict structure.
            init_type: "global" (DFCA-GI) or "local" (DFCA-LI).
        """
        self._initialized = True

        if init_type == "global" and global_params is not None:
            for cid in range(self.num_clusters):
                self.cluster_params[cid] = OrderedDict(
                    (k, v.clone().cpu()) for k, v in global_params.items()
                )
        elif init_type == "global" and template_model is not None:
            for cid in range(self.num_clusters):
                self.cluster_params[cid] = OrderedDict(
                    (k, v.clone().cpu()) for k, v in template_model.state_dict().items()
                )
        elif init_type == "local" and template_model is not None:
            # DFCA-LI: each client initializes independently using template structure
            for cid in range(self.num_clusters):
                torch.manual_seed(self.init_seed + self.client_id * self.num_clusters + cid)
                # Create a fresh model to get the right key names and structure
                temp = template_model.__class__(
                    getattr(template_model, "input_shape", 4),
                    self.num_classes
                )
                self.cluster_params[cid] = OrderedDict(
                    (k, v.clone().cpu()) for k, v in temp.state_dict().items()
                )
        elif init_type == "local":
            # Fallback: use template_model if available, else just mark initialized
            if template_model is not None:
                for cid in range(self.num_clusters):
                    torch.manual_seed(self.init_seed + self.client_id * self.num_clusters + cid)
                    temp = template_model.__class__(
                        getattr(template_model, "input_shape", 4),
                        self.num_classes
                    )
                    self.cluster_params[cid] = OrderedDict(
                        (k, v.clone().cpu()) for k, v in temp.state_dict().items()
                    )
        else:
            raise ValueError(
                "initialize_cluster_bank: for 'local' init, "
                "template_model must be provided."
            )

    def set_neighbors(self, neighbors: List[int]) -> None:
        """Set this node's communication neighbors."""
        self.neighbors = list(neighbors)

    # =======================================================================
    # Step 1: Cluster Assignment
    # =======================================================================

    def _ensure_model_on_device(self, model_template, device: str) -> nn.Module:
        """Ensure node has its own model instance on the correct device."""
        model = getattr(self, "model", None)
        wrong_device = False

        if model is not None and device.startswith("cuda"):
            try:
                p = next(model.parameters(), None)
                if p is not None and p.device.type != "cuda":
                    wrong_device = True
            except (StopIteration, ValueError):
                wrong_device = True
        elif model is not None and device == "cpu":
            try:
                p = next(model.parameters(), None)
                if p is not None and p.device.type == "cuda":
                    wrong_device = True
            except (StopIteration, ValueError):
                wrong_device = True

        if wrong_device or model is None:
            new_model = model_template.__class__(model_template.input_shape, self.num_classes)
            assigned = getattr(self, "assigned_cluster", 0)
            if assigned in self.cluster_params:
                cpu_params = OrderedDict(
                    (k, v.clone().cpu()) for k, v in self.cluster_params[assigned].items()
                )
                new_model.load_state_dict(cpu_params)
            new_model.to(device)
            self.model = new_model
            self.device = device
            self.use_amp = device.startswith("cuda")

        return self.model

    def _evaluate_cluster_loss(
        self,
        cluster_params: OrderedDict,
        model: nn.Module,
        batch_size: int = 512,
    ) -> float:
        """Evaluate a cluster model's loss on local data."""
        model.eval()
        model.load_state_dict(
            {k: v.to(self.device) for k, v in cluster_params.items()}
        )

        total_loss = 0.0
        total_samples = 0
        indices = torch.randperm(self.num_samples)

        with torch.no_grad():
            for i in range(0, self.num_samples, batch_size):
                batch_idx = indices[i : i + batch_size]
                X_batch = self.X_train[batch_idx].to(self.device, non_blocking=True)
                y_batch = self.y_train[batch_idx].to(self.device, non_blocking=True)
                out = model(X_batch)
                loss = nn.CrossEntropyLoss(reduction="mean")(out, y_batch)
                total_loss += loss.item() * len(y_batch)
                total_samples += len(y_batch)

        return total_loss / max(1, total_samples)

    def assign_cluster(
        self,
        model_template: nn.Module,
        device: str = "cpu",
        verbose: bool = False,
    ) -> Tuple[int, Dict[int, float], float]:
        """
        Step 1: Assign this node to the cluster with minimum local loss.

        c(i) = argmin_j F_client(theta_i,j, D_i)

        Args:
            model_template: Model class/template to use for evaluation.
            device: Device to run evaluation on.
            verbose: Print assignment details.

        Returns:
            (assigned_cluster_id, assignment_losses, margin)
            margin = second_best_loss - best_loss (positive = clear assignment)
        """
        model = self._ensure_model_on_device(model_template, device)

        self.assignment_losses = {}
        for cid in range(self.num_clusters):
            loss = self._evaluate_cluster_loss(self.cluster_params[cid], model)
            self.assignment_losses[cid] = loss

        sorted_losses = sorted(self.assignment_losses.values())
        best_loss = sorted_losses[0]
        second_best_loss = sorted_losses[1] if len(sorted_losses) > 1 else best_loss + 1e6
        margin = second_best_loss - best_loss

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
                f"  Node {self.client_id}: assign_cluster -> "
                f"c{self.assigned_cluster} "
                f"[margin={margin:.4f}] [{loss_str}]"
            )

        return self.assigned_cluster, self.assignment_losses, margin

    # =======================================================================
    # Step 2: Local Update
    # =======================================================================

    def _amp_ctx(self):
        if self.use_amp:
            return torch_autocast(device_type="cuda" if self.device.startswith("cuda") else "cpu",
                                   dtype=torch.float16)
        return contextlib.nullcontext()

    def train_assigned_cluster(
        self,
        model_template: nn.Module,
        device: str = "cpu",
        epochs: int = 5,
        batch_size: int = 2048,
        lr: float = 0.1,
        verbose: bool = False,
    ) -> Dict:
        """
        Step 2: Train ONLY the assigned cluster model via local SGD.

        Args:
            model_template: Model class/template.
            device: Device to train on.
            epochs: Local epochs.
            batch_size: Batch size.
            lr: Learning rate.
            verbose: Print training details.

        Returns:
            Dict with loss, assigned_cluster, params.
        """
        model = self._ensure_model_on_device(model_template, device)
        cluster_id = self.assigned_cluster
        cluster_state = self.cluster_params[cluster_id]

        model.load_state_dict(
            {k: v.to(self.device) for k, v in cluster_state.items()}
        )
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scaler = GradScaler(enabled=self.use_amp)

        total_loss = 0.0
        total_samples = 0

        # Record unassigned cluster params before training (to verify unchanged)
        unassigned_before = {}
        for cid in range(self.num_clusters):
            if cid != cluster_id:
                unassigned_before[cid] = {
                    k: v.clone().cpu()
                    for k, v in self.cluster_params[cid].items()
                }

        for _ in range(epochs):
            indices = torch.randperm(self.num_samples)
            for i in range(0, self.num_samples, batch_size):
                batch_idx = indices[i : i + batch_size]
                X_batch = self.X_train[batch_idx].to(self.device, non_blocking=True)
                y_batch = self.y_train[batch_idx].to(self.device, non_blocking=True)

                optimizer.zero_grad()
                with self._amp_ctx():
                    out = model(X_batch)
                    loss = nn.CrossEntropyLoss(reduction="mean")(out, y_batch)

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                bs = len(y_batch)
                total_loss += loss.item() * bs
                total_samples += bs

        updated_params = OrderedDict(
            (k, v.cpu().clone()) for k, v in model.state_dict().items()
        )
        self.cluster_params[cluster_id] = updated_params

        # Verify unassigned clusters are unchanged
        for cid in range(self.num_clusters):
            if cid != cluster_id:
                for k in self.cluster_params[cid]:
                    if not torch.equal(self.cluster_params[cid][k], unassigned_before[cid][k]):
                        self.cluster_params[cid] = OrderedDict(
                            (k2, v.clone()) for k2, v in unassigned_before[cid].items()
                        )
                        break

        avg_loss = total_loss / max(1, total_samples)

        if verbose:
            print(f"  Node {self.client_id}: train_assigned_cluster c{cluster_id} -> loss={avg_loss:.4f}")

        return {
            "client_id": self.client_id,
            "assigned_cluster": cluster_id,
            "loss": avg_loss,
            "params": updated_params,
        }

    # =======================================================================
    # Step 3: Decentralized Aggregation
    # =======================================================================

    def receive_message(self, sender_id: int, message: Dict[int, OrderedDict]) -> None:
        """Receive a message from a neighbor."""
        self.received_messages[sender_id] = message

    def aggregate_received_messages(self) -> Dict[int, int]:
        """
        Step 3: Sequential running average across received messages.

        For each cluster j:
            N_i,j = {m in neighbors | c(m) = j}
            For each neighbor m in N_i,j:
                theta_i,j <- (r/(r+1)) * theta_i,j + (1/(r+1)) * theta_m,j

        Returns:
            Dict[cluster_id -> update_count] — how many messages were aggregated per cluster.
        """
        update_counts: Dict[int, int] = {c: 0 for c in range(self.num_clusters)}

        for cluster_id in range(self.num_clusters):
            contributors = {}
            for sender_id, msg in self.received_messages.items():
                if cluster_id in msg:
                    contributors[sender_id] = msg[cluster_id]

            if not contributors:
                continue

            r = 0  # number of neighbors already incorporated
            for sender_id, params in contributors.items():
                for key in self.cluster_params[cluster_id]:
                    local_tensor = self.cluster_params[cluster_id][key]
                    incoming = params[key].to(local_tensor.device)

                    if local_tensor.dtype.is_floating_point and incoming.dtype.is_floating_point:
                        alpha = (r + 1.0) / (r + 2.0)
                        beta = 1.0 / (r + 2.0)
                        self.cluster_params[cluster_id][key] = (
                            alpha * local_tensor + beta * incoming
                        )

                r += 1
                update_counts[cluster_id] += 1

        self.received_messages = {}
        return update_counts

    def export_assigned_cluster_message(self) -> Dict[int, OrderedDict]:
        """
        Export this node's trained params for the assigned cluster.

        Returns:
            {assigned_cluster_id -> params}
        """
        return {
            self.assigned_cluster: self.cluster_params[self.assigned_cluster]
        }

    def export_all_cluster_params(self) -> Dict[int, OrderedDict]:
        """Export all k cluster params."""
        return {cid: params.copy() for cid, params in self.cluster_params.items()}

    # =======================================================================
    # Helpers
    # =======================================================================

    def get_assignment_losses(self) -> Dict[int, float]:
        """Get loss values for all clusters from last assignment."""
        return dict(self.assignment_losses)

    def get_cluster_assignment(self) -> int:
        """Get current cluster assignment."""
        return self.assigned_cluster
