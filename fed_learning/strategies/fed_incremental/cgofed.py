"""
CGoFed Strategy - Constrained Gradient Optimization for Federated Class Incremental Learning.

Reference:
    "CGoFed: Constrained Gradient Optimization Strategy for Federated Class
    Incremental Learning", IEEE TKDE, Vol. 37, No. 5, May 2025
    Authors: Jiyuan Feng, Xu Yang, Liwen Liang, Weihong Han, Binxing Fang, Qing Liao

Paper Algorithm Summary:
========================
1. Representation Space (Eq. 2-4):
   - R^t = F(Θ^t, X^t) = [z_1, ..., z_n] - representations from FORWARD propagation
   - SVD: R^t = U^t Σ^t (V^t)^T
   - Select κ basis vectors based on energy threshold
   - M^t = [u_1, ..., u_κ] - basis from LEFT singular vectors (U)

2. Importance Weights (Eq. 5):
   - Λ^t = sigmoid(Σ^t) = 1 / (1 + exp(-σ_i))
   - Weighted basis: M^t = [λ_1 u_1, ..., λ_κ u_κ]

3. Relaxation Coefficient (Eq. 7-8):
   - f(α, t) = α^t (power decay, NOT exponential)
   - μ_t = μ_init * α^(t - t_τ) when AF >= τ

4. Gradient Projection (Eq. 9):
   - ∇L ← ∇L - μ_t * (∇L) @ M^t @ (M^t)^T
   - Remove components in old task representation space

5. Cross-Task Regularization (Eq. 10-14):
   - Compute similarity between current and historical tasks
   - Select TOP-K most similar historical models
   - Weighted aggregation with history
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple
import copy
import os
import gc
import shutil
import uuid

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import BaseTrainer, BaseAggregator


REPRESENTATION_ARTIFACT_TYPE = "cgofed_representation_artifact"
MODEL_ARTIFACT_TYPE = "cgofed_model_artifact"


def is_representation_artifact(rep: Any) -> bool:
    """Return True when `rep` is a lightweight persisted representation artifact."""
    return isinstance(rep, dict) and rep.get("_artifact_type") == REPRESENTATION_ARTIFACT_TYPE


def is_model_artifact(obj: Any) -> bool:
    """Return True when `obj` points to a disk-backed model snapshot."""
    return isinstance(obj, dict) and obj.get("_artifact_type") == MODEL_ARTIFACT_TYPE


def coerce_representation_matrix(rep: Any) -> Optional[torch.Tensor]:
    """Normalize a representation into a stable 2D float tensor on CPU."""
    if rep is None or not isinstance(rep, torch.Tensor):
        return None
    if torch.isnan(rep).any() or torch.isinf(rep).any():
        return None
    if rep.dim() == 1:
        return rep.detach().cpu().float().unsqueeze(0)
    if rep.dim() == 2:
        return rep.detach().cpu().float()
    if rep.dim() > 2:
        return rep.detach().cpu().float().view(rep.shape[0], -1)
    return None


def representation_signature(
    rep: Any, spectral_rank: int = 8
) -> Optional[torch.Tensor]:
    """
    Build a permutation-invariant representation signature.

    The paper compares representation matrices with an L2-style distance.
    In practice, row order is arbitrary because samples are randomly drawn and
    shuffled, so we compare a stable summary instead:
    - feature mean
    - leading singular values of the centered representation matrix
    """
    if is_representation_artifact(rep):
        signature = rep.get("signature")
        if isinstance(signature, torch.Tensor):
            return signature.detach().cpu().float()
        return None

    matrix = coerce_representation_matrix(rep)
    if matrix is None:
        return None

    mean = matrix.mean(dim=0)
    centered = matrix - mean

    if centered.numel() == 0:
        spectral = torch.zeros(spectral_rank, dtype=mean.dtype)
    else:
        sv = torch.linalg.svdvals(centered)
        spectral = torch.zeros(spectral_rank, dtype=mean.dtype)
        take = min(spectral_rank, sv.shape[0])
        if take > 0:
            scale = max(float(matrix.shape[0]) ** 0.5, 1.0)
            spectral[:take] = sv[:take] / scale

    row_norms = torch.norm(matrix, dim=1, p=2)
    row_stats = torch.tensor(
        [
            row_norms.mean().item(),
            row_norms.std(unbiased=False).item(),
        ],
        dtype=mean.dtype,
    )

    return torch.cat([mean, spectral, row_stats], dim=0)


def compute_representation_similarity(
    rep_a: Any, rep_b: Any
) -> float:
    """Return a higher-is-better similarity score for two representations."""
    sig_a = representation_signature(rep_a)
    sig_b = representation_signature(rep_b)
    if sig_a is None or sig_b is None or sig_a.shape != sig_b.shape:
        return -1e6
    return -torch.norm(sig_a - sig_b, p=2).item()


def build_representation_artifact(rep: Any) -> Optional[Dict[str, Any]]:
    """Convert a representation tensor into a lightweight history artifact."""
    if is_representation_artifact(rep):
        return clone_representation_state(rep)

    matrix = coerce_representation_matrix(rep)
    if matrix is None:
        return None

    signature = representation_signature(matrix)
    if signature is None:
        return None

    mean_vector = matrix.mean(dim=0)
    return {
        "_artifact_type": REPRESENTATION_ARTIFACT_TYPE,
        "signature": signature.detach().cpu().float(),
        "shape": tuple(matrix.shape),
        "mean_vector": mean_vector.detach().cpu().float(),
        "mean_norm": torch.norm(mean_vector, p=2).item(),
    }


def clone_representation_state(rep: Any) -> Any:
    """Clone either a dense representation tensor or a lightweight artifact."""
    if is_representation_artifact(rep):
        cloned = dict(rep)
        if isinstance(rep.get("signature"), torch.Tensor):
            cloned["signature"] = rep["signature"].detach().cpu().clone()
        if isinstance(rep.get("mean_vector"), torch.Tensor):
            cloned["mean_vector"] = rep["mean_vector"].detach().cpu().clone()
        return cloned

    matrix = coerce_representation_matrix(rep)
    if matrix is not None:
        return matrix.detach().cpu().clone()
    return rep


def clone_model_reference(obj: Any) -> Any:
    """Clone a lightweight model reference without materializing tensors."""
    if is_model_artifact(obj):
        return dict(obj)
    if isinstance(obj, OrderedDict):
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in obj.items())
    return obj


def persist_model_artifact(
    params: OrderedDict,
    history_dir: str,
    artifact_key: str,
) -> Dict[str, Any]:
    """Persist model parameters to disk and return a lightweight reference."""
    os.makedirs(history_dir, exist_ok=True)
    safe_key = artifact_key.replace(os.sep, "_").replace(":", "_")
    path = os.path.join(history_dir, f"{safe_key}.pt")
    state = OrderedDict((k, v.detach().cpu().clone()) for k, v in params.items())
    torch.save(state, path)
    return {
        "_artifact_type": MODEL_ARTIFACT_TYPE,
        "path": path,
        "key": artifact_key,
    }


def load_model_state(obj: Any) -> OrderedDict:
    """Materialize a model snapshot from either an artifact ref or an OrderedDict."""
    if is_model_artifact(obj):
        path = obj.get("path")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Missing CGoFed model artifact: {path}")
        state = torch.load(path, map_location="cpu")
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in state.items())

    if isinstance(obj, OrderedDict):
        return OrderedDict((k, v.detach().cpu().clone()) for k, v in obj.items())

    raise TypeError(f"Unsupported model state type for CGoFed: {type(obj)!r}")


def clone_model_state_dict(params: OrderedDict) -> OrderedDict:
    """Clone a materialized model state dict onto CPU."""
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in params.items())


def restore_model_reference(obj: Any) -> OrderedDict:
    """
    Restore a persisted model entry from either:
    - a materialized OrderedDict of tensors, or
    - a lightweight artifact reference dict from older continuation files.
    """
    return clone_model_state_dict(load_model_state(obj))


def serialize_model_reference(obj: Any) -> OrderedDict:
    """
    Convert any supported model reference into an in-memory state dict.

    Continuation files must not depend on temp artifact paths from a previous
    Kaggle session, so we always materialize to tensors here.
    """
    return clone_model_state_dict(load_model_state(obj))


def serialize_model_reference_map(model_map: Dict[Any, Any]) -> Dict[Any, OrderedDict]:
    """Materialize a flat map of model refs into CPU state dicts."""
    return {
        key: serialize_model_reference(value)
        for key, value in model_map.items()
    }


def representation_shape(rep: Any) -> Optional[Tuple[int, ...]]:
    """Return a best-effort shape tuple for a tensor or representation artifact."""
    if is_representation_artifact(rep):
        shape = rep.get("shape")
        if shape is not None:
            return tuple(shape)
        return None
    matrix = coerce_representation_matrix(rep)
    if matrix is None:
        return None
    return tuple(matrix.shape)


def representation_mean_norm(rep: Any) -> Optional[float]:
    """Return the norm of the mean representation vector for logging/debug."""
    if is_representation_artifact(rep):
        value = rep.get("mean_norm")
        return float(value) if value is not None else None
    matrix = coerce_representation_matrix(rep)
    if matrix is None:
        return None
    return torch.norm(matrix.mean(dim=0), p=2).item()


def aggregate_representation_artifacts(
    reps: List[Any],
) -> Optional[Dict[str, Any]]:
    """Build a task-level lightweight artifact from per-client representations."""
    artifacts = []
    for rep in reps:
        artifact = rep if is_representation_artifact(rep) else build_representation_artifact(rep)
        if artifact is not None:
            artifacts.append(artifact)

    if not artifacts:
        return None

    total_samples = sum(int(artifact["shape"][0]) for artifact in artifacts)
    if total_samples <= 0:
        return None

    weights = torch.tensor(
        [artifact["shape"][0] / total_samples for artifact in artifacts],
        dtype=torch.float32,
    )
    stacked_signatures = torch.stack([artifact["signature"] for artifact in artifacts], dim=0)
    stacked_means = torch.stack([artifact["mean_vector"] for artifact in artifacts], dim=0)

    signature = torch.sum(stacked_signatures * weights.unsqueeze(1), dim=0)
    mean_vector = torch.sum(stacked_means * weights.unsqueeze(1), dim=0)

    first_shape = artifacts[0]["shape"]
    return {
        "_artifact_type": REPRESENTATION_ARTIFACT_TYPE,
        "signature": signature.detach().cpu(),
        "shape": (total_samples, int(first_shape[1])),
        "mean_vector": mean_vector.detach().cpu(),
        "mean_norm": torch.norm(mean_vector, p=2).item(),
    }


class CGoFedTrainer(BaseTrainer):
    """
    CGoFed local training with ACTIVATION-based SVD gradient constraint.

    Hiểu ngắn gọn:
    - Thu thập không gian biểu diễn của task cũ bằng activation + SVD.
    - Khi học task mới, chiếu bớt gradient theo không gian cũ để giảm quên.
    - Đồng thời dùng thông tin tương đồng task để regularize qua lịch sử.

    Implements the Relax-Constrained Gradient Update from paper Section 5.1.

    Key mechanism (Paper Eq. 2-9):
    - R^t = F(Θ^t, X^t): Collect ACTIVATIONS via forward propagation
    - SVD: R^t = U^t Σ^t (V^t)^T
    - Basis M^t = [u_1, ..., u_κ]: LEFT singular vectors (U)
    - Importance Λ^t = sigmoid(Σ^t) (scales basis vectors, Paper Eq. 5)
    - Weighted basis M^t = [λ_1*u_1, ..., λ_κ*u_κ]
    - Projection: ∇L ← ∇L - μ_t * (∇L @ M^t @ M^t^T)  (importance-weighted projector)
    - Adaptive μ_t relaxation: μ_t = α^(t - t_τ)

    Args:
        mu: Proximal regularization weight for FedProx term (optional, default 0.01)
        mu_projection: Gradient projection coefficient (paper Eq. 9, separate from proximal)
            If None, falls back to mu for backward compatibility.
        lambda_decay: Decay rate α for relaxation (paper Eq. 7: f(α,t) = α^t)
        theta_threshold: AF threshold τ to reset μ (paper Eq. 8)
        energy_threshold: SVD energy threshold for rank κ selection
        beta: Scaling factor for sigmoid importance weights
        num_samples_rep: Number of samples n_s for building representation
        temp_dir: Directory for storing SVD basis matrices
    """

    def __init__(
        self,
        mu: float = 0.0,  # Paper Eq. 14: NO proximal term in CGoFed
        mu_projection: Optional[float] = None,
        lambda_decay: float = 0.8,
        theta_threshold: float = 0.1,
        energy_threshold: float = 0.95,
        beta: float = 1.0,
        num_samples_rep: int = 100,
        temp_dir: str = "./temp_svd_storage",
        lambda_cross_task: float = 0.1,  # Paper Eq. 14: cross-task regularization weight
        max_old_tasks: Optional[int] = None,
    ):
        self.mu = mu  # No proximal term (Paper Eq. 14 doesn't have it)
        self.mu_projection = (
            mu_projection if mu_projection is not None else mu
        )  # Gradient projection (paper Eq. 9)
        self.lambda_decay = lambda_decay
        self.theta_threshold = theta_threshold
        self.energy_threshold = energy_threshold
        self.beta = beta
        self.num_samples_rep = num_samples_rep
        self.lambda_cross_task = lambda_cross_task  # Paper Eq. 14
        self.max_old_tasks = max_old_tasks

        # Historical models for cross-task regularization (Paper Eq. 11, 14)
        # {task_id: {param_name: param_tensor}}
        self.historical_models: Dict[int, OrderedDict] = {}

        # Projection statistics (for monitoring)
        self._projection_stats = {"projected": 0, "skipped": 0, "total_reduction": 0.0}
        self._pre_step_logged_this_task: bool = False

        # Temp directory for SVD matrices (lazy loading)
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)

        # Task tracking
        self.current_task: int = 0
        self.seen_classes: Set[int] = set()

        # Per-layer representation space
        # Key: task_key -> {layer_idx: {"basis": path, "importance": path}}
        self.layer_bases: Dict[str, Dict[int, Dict[str, str]]] = {}

        # Legacy attributes for compatibility
        self.old_space: Dict[str, str] = {}
        self.importance_weights: Dict[str, str] = {}

        # Gradient dimension (set when first building representation)
        self.gradient_dim: Optional[int] = None

        # μ_t relaxation coefficient (paper Eq. 7-8)
        # μ_t = μ_init * α^(t - t_reset) where α = lambda_decay
        self.mu_coefficient: float = 1.0  # Starts at 1.0, decays each task
        self.t_reset: int = (
            0  # Reset point when AF > θ; init=0 so task 1 decays as α^(1-0)=α^1
        )

        # Accuracies for computing AF (Average Forgetting)
        self.best_acc_per_task: Dict[int, float] = {}
        self.current_acc_per_task: Dict[int, float] = {}
        self.last_af: float = 0.0

        # Cache for per-layer projection matrices (loaded once per device, used many times)
        # {device_str: {layer_name: projection_matrix}}
        self._cached_proj_matrices_per_device: Dict[str, Dict[str, torch.Tensor]] = {}

    def set_task(self, task_id: int, new_classes: List[int]):
        """
        Được gọi khi bắt đầu task mới.

        Hàm này cập nhật task hiện tại, reset cache projection matrix,
        và điều chỉnh hệ số `mu_coefficient` theo cơ chế relaxation của paper.
        """
        self.current_task = task_id
        self.seen_classes.update(new_classes)

        # Invalidate cache when task changes (new basis may be added)
        self._cached_proj_matrices_per_device = {}

        # Reset debug log flag for new task
        self._pre_step_logged_this_task = False

        # Update μ with power decay (paper Eq. 7-8)
        # μ_t = μ_init * α^(t - t_reset)
        if task_id > 0:
            # Paper: f(α, t) = α^t, so μ_t = μ_init * α^(t - t_reset)
            self.mu_coefficient = self.lambda_decay ** (task_id - self.t_reset)

            print(
                f"  μ_projection = {self.mu_projection} * {self.mu_coefficient:.4f} = {self.mu_projection * self.mu_coefficient:.4f}"
                f"  (proximal μ = {self.mu})"
            )

    def update_forgetting(self, task_accuracies: Dict[int, float]):
        """
        Cập nhật Average Forgetting và reset hệ số nếu quên vượt ngưỡng.

        Đây là thành phần giúp CGoFed tự nới hoặc siết mức projection
        theo mức độ quên quan sát được qua các task trước.
        """
        self.current_acc_per_task = task_accuracies.copy()

        # Update best accuracies
        for tid, acc in task_accuracies.items():
            if tid not in self.best_acc_per_task:
                self.best_acc_per_task[tid] = acc
            else:
                self.best_acc_per_task[tid] = max(self.best_acc_per_task[tid], acc)

        # Calculate AF (paper eq. 16)
        self.last_af = 0.0
        if len(self.best_acc_per_task) > 1:
            forgetting = []
            for tid in range(self.current_task):
                if tid in self.best_acc_per_task and tid in self.current_acc_per_task:
                    f = self.best_acc_per_task[tid] - self.current_acc_per_task[tid]
                    forgetting.append(max(0, f))

            if forgetting:
                self.last_af = sum(forgetting) / len(forgetting)

                # Check if need to reset μ (paper Eq. 8)
                if self.last_af > self.theta_threshold:
                    self.t_reset = self.current_task
                    self.mu_coefficient = 1.0
                    print(
                        f"⚠️ AF={self.last_af:.4f} > θ={self.theta_threshold}, reset μ to 1.0"
                    )

    def get_current_af(self) -> float:
        """Trả về giá trị Average Forgetting mới nhất của CGoFed."""
        return self.last_af

    def _get_weight_modules(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """
        Lấy các module có trọng số để gắn activation hook.

        Đây là nguồn dữ liệu để xây không gian biểu diễn của task cũ.
        """
        weight_modules = []
        for name, module in model.named_modules():
            # Only Conv and Linear layers
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                # Skip batch norm (they don't have 'weight' in typical naming)
                if "bn" in name or "batch" in name.lower():
                    continue
                weight_modules.append((name, module))
        return weight_modules

    def _get_projection_target_modules(
        self, model: nn.Module
    ) -> List[Tuple[str, nn.Module]]:
        """
        Lấy các module phù hợp để áp dụng gradient projection.

        Áp dụng cho TẤT CẢ các layer có weights (Linear, Conv1d, Conv2d).
        Bao gồm cả output layer (fc2) vì forgetting xảy ra chủ yếu ở
        classification layer - nếu không projection, weights cho old classes
        sẽ bị overwrite hoàn toàn.
        """
        target_modules = []
        for name, module in model.named_modules():
            # Include Conv1d, Conv2d, and Linear (ALL layers with weights)
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                # Skip batch norm if any slipped through
                if "bn" in name.lower():
                    continue
                target_modules.append((name, module))
        return target_modules

    def _prepare_activation_chunk_root(self) -> str:
        """Create a clean per-task directory for out-of-core activation chunks."""
        task_key = f"task_{self.current_task}"
        chunk_root = os.path.join(self.temp_dir, f"{task_key}_activation_chunks")
        if os.path.isdir(chunk_root):
            shutil.rmtree(chunk_root, ignore_errors=True)
        os.makedirs(chunk_root, exist_ok=True)
        return chunk_root

    @staticmethod
    def _cleanup_activation_chunk_root(chunk_root: Optional[str]) -> None:
        """Remove temporary activation chunk files after basis construction."""
        if chunk_root and os.path.isdir(chunk_root):
            shutil.rmtree(chunk_root, ignore_errors=True)

    def _persist_activation_chunk(
        self,
        chunk_root: str,
        layer_name: str,
        chunk_index: int,
        features: torch.Tensor,
        chunk_prefix: str,
    ) -> Dict[str, Any]:
        """Persist one activation chunk to disk and return lightweight metadata."""
        safe_layer_name = layer_name.replace(".", "_")
        layer_dir = os.path.join(chunk_root, safe_layer_name)
        os.makedirs(layer_dir, exist_ok=True)

        tensor = features.detach().cpu().float().contiguous()
        path = os.path.join(layer_dir, f"{chunk_prefix}_{chunk_index:06d}.pt")
        torch.save(tensor, path)
        return {
            "path": path,
            "shape": tuple(tensor.shape),
        }

    def _collect_activations(
        self,
        model: nn.Module,
        data_loader,
        device: str,
        num_samples: int,
        chunk_root: str,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Collect per-layer INPUT ACTIVATIONS during forward pass.

        Paper Eq. 2: R^t = F(Θ^t, X^t)
        This is the representation obtained through forward propagation.

        NOTE: We only collect activations for Linear layers (not Conv) because
        gradient projection only works well when activation_dim matches weight_dim.
        Conv layers have mismatched dimensions due to spatial pooling.

        Args:
            model: Model to collect activations from
            data_loader: Data loader for samples
            device: Device to use
            num_samples: Number of samples to use

        Returns:
            Dict mapping module_name -> list of on-disk activation chunk refs
        """
        # Only collect for projection target modules (Linear layers)
        weight_modules = self._get_projection_target_modules(model)
        layer_activations: Dict[str, List[Dict[str, Any]]] = {
            name: [] for name, _ in weight_modules
        }

        # Storage for captured activations
        captured: Dict[str, Dict[str, Any]] = {}
        handles = []
        chunk_counts: Dict[str, int] = {name: 0 for name, _ in weight_modules}
        chunk_prefix = uuid.uuid4().hex

        def make_hook(layer_name: str):
            def hook_fn(module, inp, out):
                # inp is a tuple, take first element
                if isinstance(inp, tuple) and len(inp) > 0:
                    activation = inp[0]
                else:
                    activation = inp

                # Handle different layer types for correct projection dimension
                if isinstance(module, nn.Linear):
                    # Linear: [batch, in_features]
                    # No unfolding needed, just flatten batch
                    features = activation.detach().view(activation.size(0), -1)

                elif isinstance(module, nn.Conv2d):
                    # Conv2d: [batch, channel, height, width]
                    # Need to UNFOLD into patches: [batch, channel*kernel_h*kernel_w, L]
                    try:
                        inp_unf = F.unfold(
                            activation.detach(),
                            kernel_size=module.kernel_size,
                            dilation=module.dilation,
                            padding=module.padding,
                            stride=module.stride,
                        )  # [B, C*Kh*Kw, L]
                        # Transpose to [B, L, C*Kh*Kw] and flatten to [B*L, C*Kh*Kw]
                        features = (
                            inp_unf.transpose(1, 2)
                            .contiguous()
                            .view(-1, inp_unf.size(1))
                        )
                    except Exception as e:
                        print(f"⚠️ Unfold failed for {layer_name}: {e}")
                        return

                elif isinstance(module, nn.Conv1d):
                    # Conv1d: [batch, channel, length]
                    # F.unfold only supports 4D, so we unsqueeze to [batch, channel, 1, length]
                    try:
                        inp_4d = activation.detach().unsqueeze(2)
                        # Kernel size (1, k)
                        k = (
                            module.kernel_size[0]
                            if isinstance(module.kernel_size, tuple)
                            else module.kernel_size
                        )
                        s = (
                            module.stride[0]
                            if isinstance(module.stride, tuple)
                            else module.stride
                        )
                        p = (
                            module.padding[0]
                            if isinstance(module.padding, tuple)
                            else module.padding
                        )
                        d = (
                            module.dilation[0]
                            if isinstance(module.dilation, tuple)
                            else module.dilation
                        )

                        inp_unf = F.unfold(
                            inp_4d,
                            kernel_size=(1, k),
                            dilation=(1, d),
                            padding=(0, p),
                            stride=(1, s),
                        )  # [B, C*1*K, L_out]

                        features = (
                            inp_unf.transpose(1, 2)
                            .contiguous()
                            .view(-1, inp_unf.size(1))
                        )
                    except Exception as e:
                        print(f"⚠️ Unfold1d failed for {layer_name}: {e}")
                        return

                else:
                    # Fallback
                    features = activation.detach().view(activation.size(0), -1)

                captured[layer_name] = self._persist_activation_chunk(
                    chunk_root=chunk_root,
                    layer_name=layer_name,
                    chunk_index=chunk_counts[layer_name],
                    features=features,
                    chunk_prefix=chunk_prefix,
                )
                chunk_counts[layer_name] += 1

            return hook_fn

        # Register hooks
        for name, module in weight_modules:
            handle = module.register_forward_hook(make_hook(name))
            handles.append(handle)

        # Collect activations via forward pass
        model.eval()
        sample_count = 0

        with torch.no_grad():
            for X, _y in data_loader:
                if sample_count >= num_samples:
                    break

                X = X.to(device)
                batch_size = X.size(0)

                # Forward pass triggers hooks
                _ = model(X)

                # Store captured activations for each layer
                for name, _ in weight_modules:
                    if name in captured:
                        # Append each sample's activation
                        layer_activations[name].append(captured[name])

                sample_count += batch_size
                captured.clear()

        # Remove hooks
        for handle in handles:
            handle.remove()

        return layer_activations

    def build_space_from_clients(
        self, model: nn.Module, clients: List, config: Dict, device: str = "cuda"
    ):
        """
        Build the task representation space from client-local samples.

        Instead of centralizing all raw client tensors on the server, we ask
        each client for a small local sample loader and aggregate activation
        summaries only after they are computed on-device.
        """
        print(
            "\n🔐 Building activation-based representation space from client-local samples..."
        )

        active_clients = [client for client in clients if getattr(client, "num_samples", 0) > 0]
        if not active_clients:
            print("⚠️ No client data available for representation space")
            return

        num_samples = int(config.get("num_samples_rep", self.num_samples_rep))
        if num_samples <= 0:
            print("⚠️ num_samples_rep must be positive to build representation space")
            return
        samples_per_client = (num_samples + len(active_clients) - 1) // len(active_clients)
        merged_layer_activations: Dict[str, List[Dict[str, Any]]] = {}
        chunk_root = self._prepare_activation_chunk_root()

        try:
            for client in active_clients:
                if not hasattr(client, "build_representation_loader"):
                    continue

                rep_loader = client.build_representation_loader(
                    num_samples=samples_per_client,
                    batch_size=32,
                )
                if rep_loader is None:
                    continue

                client_activations = self._collect_activations(
                    model=model,
                    data_loader=rep_loader,
                    device=device,
                    num_samples=samples_per_client,
                    chunk_root=chunk_root,
                )
                for layer_name, activation_chunks in client_activations.items():
                    if not activation_chunks:
                        continue
                    merged_layer_activations.setdefault(layer_name, []).extend(activation_chunks)

            if not merged_layer_activations:
                print("⚠️ No client activations collected for representation space")
                return

            self._build_representation_space_from_activations(
                model=model,
                layer_activations=merged_layer_activations,
            )
        finally:
            self._cleanup_activation_chunk_root(chunk_root)

    def build_space_from_client_data(
        self, model: nn.Module, client_data: Dict, config: Dict, device: str = "cuda"
    ):
        """
        Backward-compatible fallback for legacy callers.

        The paper-faithful path uses `build_space_from_clients(...)`, but this
        fallback is kept to avoid breaking older scripts that still pass a
        centralized client-data dictionary.
        """
        print(
            "\n⚠️ Legacy CGoFed path: building representation space from centralized client_data."
        )
        self.build_representation_space_from_legacy_client_data(
            model=model,
            client_data=client_data,
            config=config,
            device=device,
        )

    def build_representation_space_from_legacy_client_data(
        self, model: nn.Module, client_data: Dict, config: Dict, device: str = "cuda"
    ):
        """Compatibility wrapper for older centralized simulations."""
        active_cids = [
            cid for cid, data in client_data.items() if len(data.get("y_train", [])) > 0
        ]
        if len(active_cids) == 0:
            print("⚠️ No client data available for representation space")
            return

        all_X, all_y = [], []
        num_samples = int(config.get("num_samples_rep", self.num_samples_rep))
        if num_samples <= 0:
            print("⚠️ num_samples_rep must be positive to build representation space")
            return
        samples_per_client = (num_samples + len(active_cids) - 1) // len(active_cids)

        for cid in active_cids:
            X = client_data[cid]["X_train"]
            y = client_data[cid]["y_train"]
            if len(y) > samples_per_client:
                indices = torch.randperm(len(y))[:samples_per_client]
                X, y = X[indices], y[indices]

            all_X.append(X)
            all_y.append(y)

        rep_X = torch.cat(all_X, dim=0)
        rep_y = torch.cat(all_y, dim=0)
        if len(rep_y) > num_samples:
            indices = torch.randperm(len(rep_y))[:num_samples]
            rep_X, rep_y = rep_X[indices], rep_y[indices]

        print(f"   Using {len(rep_y)} centralized samples from {len(active_cids)} clients for SVD.")

        from torch.utils.data import TensorDataset, DataLoader

        rep_loader = DataLoader(
            TensorDataset(rep_X, rep_y),
            batch_size=32,
            shuffle=False,
        )
        self.build_representation_space(model=model, data_loader=rep_loader, device=device)

    def build_representation_space(
        self, model: nn.Module, data_loader, device: str = "cuda"
    ):
        """
        Build PER-LAYER representation space using ACTIVATIONS (Paper Eq. 2-5).

        Paper Eq. 2: R^t = F(Θ^t, X^t)
        - R^t is the representation from forward propagation

        Paper Eq. 3: SVD of R^t = U Σ V^T
        - Basis M^t = [u_1, ..., u_κ] from left singular vectors U

        Paper Eq. 5: Importance Λ = sigmoid(Σ)
        - λ_i = 1 / (1 + exp(-σ_i))

        For each layer, we:
        1. Collect input activations during forward pass
        2. SVD on activation matrix
        3. Store U[:, :k] as basis and sigmoid(S[:k]) as importance
        """
        was_training = model.training

        print(f"  → Building ACTIVATION-based representation space (Paper Eq. 2-5)...")

        chunk_root = self._prepare_activation_chunk_root()
        layer_activations: Optional[Dict[str, List[Dict[str, Any]]]] = None
        try:
            layer_activations = self._collect_activations(
                model,
                data_loader,
                device,
                self.num_samples_rep,
                chunk_root,
            )
            if not layer_activations:
                print("⚠️ No activations collected")
                return

            self._build_representation_space_from_activations(model, layer_activations)
        finally:
            self._cleanup_activation_chunk_root(chunk_root)
            if layer_activations is not None:
                del layer_activations
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if was_training:
                model.train()

    def _build_representation_space_from_activations(
        self, model: nn.Module, layer_activations: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Run out-of-core SVD statistics and persist per-layer bases."""
        weight_modules = self._get_projection_target_modules(model)
        print(
            f"  → Found {len(weight_modules)} layers for projection (Linear/Conv)"
        )

        task_key = f"task_{self.current_task}"
        self.layer_bases[task_key] = {}

        for layer_name, _module in weight_modules:
            if layer_name not in layer_activations or not layer_activations[layer_name]:
                continue

            try:
                gram = None
                n_samples = 0
                d = None

                for chunk_ref in layer_activations[layer_name]:
                    chunk_path = chunk_ref.get("path")
                    if not chunk_path or not os.path.exists(chunk_path):
                        continue

                    chunk = torch.load(chunk_path, map_location="cpu")
                    chunk = chunk.detach().cpu().float()
                    if chunk.dim() != 2 or chunk.numel() == 0:
                        continue

                    if d is None:
                        d = chunk.shape[1]
                        gram = torch.zeros((d, d), dtype=torch.float64)
                    if chunk.shape[1] != d:
                        raise ValueError(
                            f"Inconsistent activation dimension for {layer_name}: "
                            f"expected {d}, got {chunk.shape[1]}"
                        )

                    gram += chunk.T.double().mm(chunk.double())
                    n_samples += chunk.shape[0]

                if gram is None or d is None or n_samples == 0:
                    continue

                eigvals, eigvecs = torch.linalg.eigh(gram)
                order = torch.argsort(eigvals, descending=True)
                eigvals = torch.clamp(eigvals[order], min=0.0)
                eigvecs = eigvecs[:, order]
                significant = eigvals > 1e-12
                if not torch.any(significant):
                    continue

                eigvals = eigvals[significant]
                eigvecs = eigvecs[:, significant]
                S = torch.sqrt(eigvals.float())
                U = eigvecs.float()

                energy = eigvals.float()
                cum_energy = torch.cumsum(energy, dim=0)
                total_energy = cum_energy[-1] + 1e-10
                ratio = cum_energy / total_energy

                k = (ratio < self.energy_threshold).sum().item() + 1
                k = min(k, len(S))

                basis = U[:, :k]
                importance = torch.sigmoid(self.beta * S[:k])

                basis_path = os.path.join(
                    self.temp_dir, f"{task_key}_{layer_name}_basis.pt"
                )
                importance_path = os.path.join(
                    self.temp_dir, f"{task_key}_{layer_name}_importance.pt"
                )

                torch.save(basis.clone(), basis_path)
                torch.save(importance.clone(), importance_path)

                self.layer_bases[task_key][layer_name] = {
                    "basis": basis_path,
                    "importance": importance_path,
                    "shape": (d, k),
                }

                print(f"    Layer {layer_name}: R=[{n_samples}, {d}], k={k}")

            except Exception as e:
                print(f"  ⚠️ SVD failed for layer {layer_name}: {e}")
                continue

        total_bases = sum(1 for task in self.layer_bases.values() for _ in task)
        print(
            f"  → Built {len(self.layer_bases[task_key])} layer bases for task {self.current_task}"
        )
        print(f"  → Total bases across all tasks: {total_bases}")
        print(
            f"✓ Built activation-based representation space for task {self.current_task}"
        )

    def compute_loss(
        self,
        model: nn.Module,
        output: torch.Tensor,
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute loss with cross-task regularization (Paper Eq. 14).

        Paper Eq. 14: Local objective includes cross-task regularization term A(Θ):
        Loss = CE(output, target) + A(Θ_k^t, Θ_old)

        Where A(Θ_k^t, Θ_old) encourages the local model to stay close to
        similar historical task models during training.
        """
        ce_loss = F.cross_entropy(output, target)

        # Paper Eq. 14: Add cross-task regularization A(Θ_k^t, Θ_old)
        # Get historical models and similarity weights from kwargs (passed by server)
        historical_models = kwargs.get("historical_models", {})
        similarity_weights = kwargs.get("similarity_weights", {})

        if historical_models and self.current_task > 0:
            reg_term = 0.0
            for task_id, hist_params in historical_models.items():
                # Get similarity weight for this historical task
                weight = similarity_weights.get(task_id, 1.0 / len(historical_models))

                # Compute ||Θ - Θ_hist||^2
                task_reg = 0.0
                for name, param in model.named_parameters():
                    if name in hist_params:
                        hist_param = hist_params[name].to(param.device)
                        task_reg += torch.sum((param - hist_param) ** 2)

                reg_term += weight * task_reg

            if reg_term > 0:
                # Scale by lambda_cross_task
                lambda_reg = getattr(self, "lambda_cross_task", 0.1)
                total_loss = ce_loss + (lambda_reg / 2) * reg_term
                # DEBUG[Loss]: Log cross-task regularization contribution
                if not hasattr(self, '_loss_log_count'):
                    self._loss_log_count = 0
                self._loss_log_count += 1
                if self._loss_log_count <= 3:
                    print(f"    DEBUG[Loss]: CE={ce_loss.item():.4f}, reg={reg_term.item():.4f}, lambda={lambda_reg}, total={total_loss.item():.4f}")
                return total_loss

        return ce_loss

    def pre_step(
        self, model: nn.Module, global_params: Optional[OrderedDict] = None, **kwargs
    ) -> None:
        """
        Gradient projection BEFORE optimizer step (paper eq. 8).

        This is called between backward() and optimizer.step() to modify
        gradients before they are applied to weights.

        Paper CGoFed eq. 8:
        g' = g - α * Σ_j Σ_i (w_{j,i} * (v_{j,i}^T @ g) * v_{j,i})

        OPTIMIZED VERSION with NaN prevention:
        - Cache basis matrices in memory (no disk I/O per step)
        - Vectorized projection (no Python loop)
        - Matrix operation: proj = P @ g where P = importance-weighted projector
        - Union of subspaces across tasks via SVD re-orthogonalization
        - Comprehensive NaN/Inf checks at every step
        """
        # Skip for first task (no old space to project against)
        if self.current_task == 0 or not self.layer_bases:
            return

        with torch.no_grad():
            device = next(model.parameters()).device
            device_key = str(device)

            # Lazy load and cache projection matrices PER DEVICE
            # This avoids race conditions when multiple GPU threads share the trainer
            if device_key not in self._cached_proj_matrices_per_device:
                self._cache_projection_matrices(device_key)

            cached = self._cached_proj_matrices_per_device.get(device_key, {})

            # Skip if no cached projections
            if not cached:
                return

            # Get module name to param mapping (only Linear layers for projection)
            module_params = {}
            for name, module in self._get_projection_target_modules(model):
                if hasattr(module, "weight") and module.weight is not None:
                    module_params[name] = module.weight

            # DEBUG[8]: Log projection coverage once per task (first pre_step call)
            if not self._pre_step_logged_this_task:
                # List ALL model layers and show which ones are protected
                all_layers = [
                    (n, m)
                    for n, m in model.named_modules()
                    if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d))
                ]
                projected_names = set(module_params.keys())
                cached_names = set(cached.keys())
                print(f"  DEBUG[8]: Projection coverage for task {self.current_task}:")
                print(f"    Model layers (Linear/Conv): {[n for n, _ in all_layers]}")
                print(f"    Layers selected for projection: {sorted(projected_names)}")
                print(f"    Layers with cached projector: {sorted(cached_names)}")
                print(
                    f"    Layers ACTUALLY projected (intersection): {sorted(projected_names & cached_names)}"
                )
                # Show which layers are NOT protected
                unprotected = set(n for n, _ in all_layers) - (
                    projected_names & cached_names
                )
                print(f"    Layers NOT protected: {sorted(unprotected)}")

            # Apply projection per layer (Paper Eq. 9)
            per_layer_reduction = {}  # For debug logging
            for layer_name, Uf in cached.items():
                if layer_name not in module_params:
                    continue

                param = module_params[layer_name]
                if param.grad is None:
                    continue

                # No need to move device -- cached per-device already on correct device

                # Get gradient shape info
                grad_shape = param.grad.shape  # [out, in] or [out, in, k, k] or [out]
                out_dim = grad_shape[0]

                # For bias (1D), skip projection (no input dimension to project)
                if len(grad_shape) == 1:
                    continue

                # Compute input dimension (flatten all dims except output dim)
                in_dim = param.grad.numel() // out_dim

                # Check dimension compatibility with Uf
                # Uf shape: [d, d] where d = activation dimension
                # For the projection to work, in_dim must match Uf.shape[0]
                # Check dimension compatibility with Uf
                # Uf shape: [d, d]. For Conv layers, d = C_in * K (kernel size) thanks to Unfold.
                # grad_in_dim should now match d.
                if in_dim != Uf.shape[0]:
                    # If mismatch persists, it typically means Unfold logic didn't match Kernel size
                    self._projection_stats["skipped"] += 1
                    if self._projection_stats["skipped"] <= 5:
                        print(
                            f"    ⚠️ CGoFed: Dim mismatch {layer_name}: grad_in={in_dim} vs Uf={Uf.shape[0]}"
                        )
                    continue

                # Reshape gradient to 2D: [out_dim, in_dim]
                grad_2d = param.grad.view(out_dim, in_dim)

                # Apply projection: grad = grad - μ_t * (grad @ P)
                # Paper Eq. 5+9: g' = g - μ_t * g @ M @ M^T
                # where M = importance-weighted basis, P = M @ M^T
                try:
                    projected = torch.mm(grad_2d, Uf)  # [out_dim, in_dim]

                    if torch.isnan(projected).any() or torch.isinf(projected).any():
                        self._projection_stats["skipped"] += 1
                        continue

                    # Compute adaptive μ_t (paper Eq. 8)
                    # Sử dụng mu_projection riêng biệt với mu proximal
                    mu_t = self.mu_projection * self.mu_coefficient

                    # Apply projection
                    grad_new = grad_2d - mu_t * projected

                    # Track projection statistics
                    orig_norm = grad_2d.norm().item()
                    new_norm = grad_new.norm().item()
                    if orig_norm > 1e-8:
                        reduction = (orig_norm - new_norm) / orig_norm
                        self._projection_stats["total_reduction"] += reduction
                        self._projection_stats["projected"] += 1
                        # Collect per-layer reduction for debug log
                        if not self._pre_step_logged_this_task:
                            per_layer_reduction[layer_name] = {
                                "reduction_pct": reduction * 100,
                                "orig_norm": orig_norm,
                                "new_norm": new_norm,
                                "proj_rank": Uf.shape[0],
                            }

                    # Copy back to param.grad
                    param.grad.copy_(grad_new.view_as(param.grad))

                except RuntimeError as e:
                    # Handle any runtime errors during projection
                    self._projection_stats["skipped"] += 1
                    if self._projection_stats["skipped"] <= 5:
                        print(f"    ⚠️ CGoFed: Projection error for {layer_name}: {e}")
                    continue

            # DEBUG[9]: Log per-layer gradient reduction (first pre_step call only)
            if not self._pre_step_logged_this_task and per_layer_reduction:
                mu_t = self.mu_projection * self.mu_coefficient
                print(
                    f"  DEBUG[9]: Per-layer gradient reduction (first batch, task {self.current_task}):"
                )
                print(f"    mu_projection={self.mu_projection}, mu_coefficient={self.mu_coefficient:.4f}, mu_t={mu_t:.4f}")
                for ln, info in per_layer_reduction.items():
                    print(
                        f"    {ln}: reduction={info['reduction_pct']:.1f}% | "
                        f"grad_norm {info['orig_norm']:.4f} → {info['new_norm']:.4f} | "
                        f"projector_dim={info['proj_rank']}"
                    )
                self._pre_step_logged_this_task = True

    def _cache_projection_matrices(self, device: str):
        """
        Build and cache projection matrices for all layers from all old tasks.

        Paper Eq. 5 + Eq. 9:
        - M^t = [λ_1*u_1, ..., λ_κ*u_κ]  (importance-weighted basis, Eq. 5)
        - P = M^t @ (M^t)^T                (projection matrix, Eq. 9)
        - g' = g - μ_t * g @ P             (gradient update, Eq. 9)

        FIX: Instead of SUMMING per-task projection matrices (which causes
        eigenvalues > 1 and AMPLIFIES gradients), we concatenate importance-
        weighted basis vectors from all old tasks and re-orthogonalize via SVD.
        The re-orthogonalized projector captures the union of all old task
        subspaces with eigenvalues bounded in [0, 1].

        If `max_old_tasks` is set, only the most recent tasks are retained in
        the projection union. The paper-faithful default is `None`, meaning all
        historical tasks contribute to the projector.
        """
        task_keys = sorted(
            [k for k in self.layer_bases.keys() if k.startswith("task_")],
            key=lambda x: int(x.split("_")[1])
        )
        old_task_keys = [k for k in task_keys if int(k.split("_")[1]) < self.current_task]
        if self.max_old_tasks is not None:
            old_task_keys = old_task_keys[-self.max_old_tasks:]
            if len(task_keys) > self.max_old_tasks + 1:
                print(
                    f"  📦 [MEMORY] Using last {self.max_old_tasks} tasks for projection "
                    "(discarding older bases)"
                )

        # Step 1: Collect importance-weighted basis vectors per layer across recent tasks
        layer_all_weighted_bases: Dict[str, List[torch.Tensor]] = {}

        for task_key in old_task_keys:
            layer_dict = self.layer_bases.get(task_key, {})
            for layer_name, info in layer_dict.items():
                try:
                    # Load basis and importance directly to the target device
                    basis = torch.load(info["basis"], map_location=device)
                    importance = torch.load(info["importance"], map_location=device)

                    # Validate
                    if torch.isnan(basis).any() or torch.isinf(basis).any():
                        continue
                    if torch.isnan(importance).any() or torch.isinf(importance).any():
                        continue

                    # Paper Eq. 5: M^t = [λ_1*u_1, ..., λ_κ*u_κ]
                    # basis: [d, k], importance: [k] -> weighted_basis: [d, k]
                    weighted_basis = basis * importance  # broadcasting

                    if layer_name not in layer_all_weighted_bases:
                        layer_all_weighted_bases[layer_name] = []
                    layer_all_weighted_bases[layer_name].append(weighted_basis)

                except Exception as e:
                    print(f"⚠️ Failed to load basis for {layer_name}: {e}")

        # Step 2: Build projector per layer via union of importance-weighted subspaces
        cached = {}

        for layer_name, bases_list in layer_all_weighted_bases.items():
            try:
                # Concatenate all importance-weighted basis vectors: [d, k1+k2+...]
                all_weighted = torch.cat(bases_list, dim=1)

                # Re-orthogonalize via SVD to get the union of subspaces
                # This ensures the projector eigenvalues are bounded in [0, 1]
                U, S, Vh = torch.linalg.svd(all_weighted, full_matrices=False)

                # Keep only significant components (singular value > threshold)
                significant = S > 1e-6
                U_orth = U[:, significant]  # [d, r] orthonormal basis
                S_sig = S[significant]  # [r] singular values

                # Normalize singular values to [0, 1] range for importance weighting
                # S_sig reflects the importance-weighted magnitude of each direction
                # Max singular value caps the projection strength
                S_normalized = S_sig / (S_sig.max() + 1e-10)

                # Build importance-weighted projector: P = U @ diag(S_norm²) @ U^T
                # S_norm² ensures eigenvalues of P are in [0, 1]
                # This preserves the paper's intent: more important directions
                # (higher singular value) get stronger projection
                Uf = torch.mm(U_orth * (S_normalized**2), U_orth.T)  # [d, d]

                # DEBUG #4: Projection matrix stats
                try:
                    eigenvalues = torch.linalg.eigvalsh(Uf)
                    print(
                        f"    DEBUG[4]: {layer_name} | shape={Uf.shape} | rank={U_orth.shape[1]} "
                        f"(from {all_weighted.shape[1]} weighted basis vectors) | "
                        f"min_eig={eigenvalues.min().item():.6f} | max_eig={eigenvalues.max().item():.6f}"
                    )
                except:
                    print(
                        f"    DEBUG[4]: {layer_name} | shape={Uf.shape} | (eig check failed)"
                    )

                cached[layer_name] = Uf

            except Exception as e:
                print(f"⚠️ Failed to build projector for {layer_name}: {e}")

        # Store in per-device cache (thread-safe: each device writes to its own key)
        self._cached_proj_matrices_per_device[device] = cached

        if cached:
            print(f"  📦 Cached {len(cached)} layer projection matrices on {device}")
            # Print which layers will be projected
            proj_layers = list(cached.keys())
            print(f"     Target layers: {proj_layers}")

    def get_optimizer_class(self) -> type:
        """Use SGD as specified in paper experiments."""
        return torch.optim.SGD

    def get_projection_stats(self, reset: bool = True) -> Dict:
        """
        Get and optionally reset projection statistics for monitoring.

        Returns:
            Dict with 'projected' (count), 'skipped' (count), 'avg_reduction' (float).
        """
        stats = self._projection_stats.copy()
        if stats["projected"] > 0:
            stats["avg_reduction"] = stats["total_reduction"] / stats["projected"]
        else:
            stats["avg_reduction"] = 0.0

        if reset:
            self._projection_stats = {
                "projected": 0,
                "skipped": 0,
                "total_reduction": 0.0,
            }

        return stats

    def log_projection_stats(self, reset: bool = True):
        """
        Log projection statistics to console.

        Call this after each round or task to monitor projection effectiveness.
        """
        stats = self.get_projection_stats(reset=reset)
        total = stats["projected"] + stats["skipped"]
        if total > 0:
            proj_rate = stats["projected"] / total * 100
            print(
                f"  📊 CGoFed Projection Stats: {stats['projected']}/{total} "
                f"({proj_rate:.1f}%) projected, "
                f"avg_reduction={stats['avg_reduction']:.4f}"
            )
        else:
            print(f"  📊 CGoFed Projection Stats: No projections attempted")

    def get_resume_state(self) -> Dict[str, Any]:
        """
        Export all CGoFed trainer state needed for split-run continuation.

        The critical part is materializing basis/importance tensors into the
        continuation file, so resume no longer depends on the previous session's
        temp directory.
        """
        serialized_layer_bases: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for task_key, layer_dict in self.layer_bases.items():
            serialized_layer_bases[task_key] = {}
            for layer_name, info in layer_dict.items():
                basis_path = info.get("basis")
                importance_path = info.get("importance")
                if not basis_path or not importance_path:
                    continue
                if not os.path.exists(basis_path) or not os.path.exists(importance_path):
                    continue
                serialized_layer_bases[task_key][layer_name] = {
                    "basis_tensor": torch.load(basis_path, map_location="cpu"),
                    "importance_tensor": torch.load(importance_path, map_location="cpu"),
                    "shape": tuple(info.get("shape", ())),
                }

        return {
            "mu": self.mu,
            "mu_projection": self.mu_projection,
            "lambda_decay": self.lambda_decay,
            "theta_threshold": self.theta_threshold,
            "energy_threshold": self.energy_threshold,
            "beta": self.beta,
            "num_samples_rep": self.num_samples_rep,
            "lambda_cross_task": self.lambda_cross_task,
            "max_old_tasks": self.max_old_tasks,
            "historical_models": serialize_model_reference_map(self.historical_models),
            "_projection_stats": dict(self._projection_stats),
            "current_task": self.current_task,
            "seen_classes": sorted(self.seen_classes),
            "gradient_dim": self.gradient_dim,
            "mu_coefficient": self.mu_coefficient,
            "t_reset": self.t_reset,
            "best_acc_per_task": dict(self.best_acc_per_task),
            "current_acc_per_task": dict(self.current_acc_per_task),
            "last_af": self.last_af,
            "_loss_log_count": getattr(self, "_loss_log_count", 0),
            "layer_bases_state": serialized_layer_bases,
        }

    def load_resume_state(self, state: Dict[str, Any]) -> None:
        """Restore trainer state and re-materialize basis tensors to temp_dir."""
        self.mu = float(state.get("mu", self.mu))
        self.mu_projection = float(state.get("mu_projection", self.mu_projection))
        self.lambda_decay = float(state.get("lambda_decay", self.lambda_decay))
        self.theta_threshold = float(state.get("theta_threshold", self.theta_threshold))
        self.energy_threshold = float(state.get("energy_threshold", self.energy_threshold))
        self.beta = float(state.get("beta", self.beta))
        self.num_samples_rep = int(state.get("num_samples_rep", self.num_samples_rep))
        self.lambda_cross_task = float(
            state.get("lambda_cross_task", self.lambda_cross_task)
        )
        self.max_old_tasks = state.get("max_old_tasks", self.max_old_tasks)

        historical_models = state.get("historical_models", {})
        self.historical_models = {
            int(task_id): clone_model_state_dict(params)
            for task_id, params in historical_models.items()
        }

        self._projection_stats = dict(
            state.get(
                "_projection_stats",
                {"projected": 0, "skipped": 0, "total_reduction": 0.0},
            )
        )
        self.current_task = int(state.get("current_task", self.current_task))
        self.seen_classes = set(state.get("seen_classes", []))
        self.gradient_dim = state.get("gradient_dim", self.gradient_dim)
        self.mu_coefficient = float(
            state.get("mu_coefficient", self.mu_coefficient)
        )
        self.t_reset = int(state.get("t_reset", self.t_reset))
        self.best_acc_per_task = {
            int(task_id): float(acc)
            for task_id, acc in state.get("best_acc_per_task", {}).items()
        }
        self.current_acc_per_task = {
            int(task_id): float(acc)
            for task_id, acc in state.get("current_acc_per_task", {}).items()
        }
        self.last_af = float(state.get("last_af", self.last_af))
        self._loss_log_count = int(state.get("_loss_log_count", 0))

        os.makedirs(self.temp_dir, exist_ok=True)
        self.layer_bases = {}
        for task_key, layer_dict in state.get("layer_bases_state", {}).items():
            self.layer_bases[task_key] = {}
            for layer_name, info in layer_dict.items():
                basis_tensor = info.get("basis_tensor")
                importance_tensor = info.get("importance_tensor")
                if not isinstance(basis_tensor, torch.Tensor):
                    continue
                if not isinstance(importance_tensor, torch.Tensor):
                    continue
                basis_path = os.path.join(
                    self.temp_dir, f"{task_key}_{layer_name}_basis.pt"
                )
                importance_path = os.path.join(
                    self.temp_dir, f"{task_key}_{layer_name}_importance.pt"
                )
                torch.save(basis_tensor.detach().cpu().clone(), basis_path)
                torch.save(importance_tensor.detach().cpu().clone(), importance_path)
                shape = info.get("shape")
                if shape is None or len(shape) != 2:
                    shape = tuple(basis_tensor.shape)
                self.layer_bases[task_key][layer_name] = {
                    "basis": basis_path,
                    "importance": importance_path,
                    "shape": tuple(shape),
                }

        self.old_space = {}
        self.importance_weights = {}
        self._cached_proj_matrices_per_device = {}
        self._pre_step_logged_this_task = False


class CGoFedAggregator(BaseAggregator):
    """
    CGoFed aggregation với regularization theo lịch sử task.

    Aggregator này không chỉ average model hiện tại mà còn lưu lịch sử,
    đo độ tương đồng giữa các task, rồi chọn ra các model cũ phù hợp nhất
    để hỗ trợ regularization ở vòng sau.
    """

    def __init__(
        self,
        cross_task_weight: float = 0.3,
        top_k: int = 2,
        rounds_per_task: int = 5,
        history_dir: str = "./temp_cgofed_history",
    ):
        self.cross_task_weight = cross_task_weight
        self.top_k = top_k
        self.rounds_per_task = rounds_per_task
        self.history_dir = history_dir
        self._client_history_dir = os.path.join(self.history_dir, "clients")
        self._global_history_dir = os.path.join(self.history_dir, "global")
        os.makedirs(self._client_history_dir, exist_ok=True)
        os.makedirs(self._global_history_dir, exist_ok=True)

        # Store activation representations from clients (Paper Eq. 2)
        # {client_id: {task_id: activation_vector}}
        self.client_representations: Dict[int, Dict[int, torch.Tensor]] = {}

        # Historical global models: {task_id: model_artifact}
        self.task_global_models: Dict[int, Dict[str, Any]] = {}

        # Personalized historical models per client (Paper Eq. 12)
        # {client_id: {task_id: model_artifact}}
        self.client_historical_models: Dict[int, Dict[int, Dict[str, Any]]] = {}

        # Mean representation per task (aggregated from clients)
        self.task_representations: Dict[int, torch.Tensor] = {}

        # Full representation matrices per task (Paper Eq. 2, 10)
        # {task_id: tensor[n_samples, hidden_dim]}
        self.task_representation_matrices: Dict[int, torch.Tensor] = {}

        # Current similarity weights and historical models for Eq. 14 local regularization
        self._current_similarity_weights: Dict[int, float] = {}
        self._current_historical_models: Dict[int, OrderedDict] = {}

        self.current_task: int = 0
        self._round_in_task: int = 0  # Track round within current task (Issue 5)

    def set_task(self, task_id: int):
        """Cập nhật task hiện tại và reset bộ đếm round trong task."""
        self.current_task = task_id
        self._round_in_task = 0
        # Reset current regularization info for new task
        self._current_similarity_weights = {}
        self._current_historical_models = {}

    def get_local_regularization_info(self) -> Dict:
        """
        Trả về historical models và similarity weights cho local regularization.

        Server sẽ lấy thông tin này sau aggregate rồi chuyền xuống client,
        để local loss của CGoFed biết cần regularize theo task cũ nào.
        """
        return {
            "historical_models": self._current_historical_models.copy(),
            "similarity_weights": self._current_similarity_weights.copy(),
        }

    def get_resume_state(self) -> Dict[str, Any]:
        """
        Export aggregator state with all artifact-backed models materialized.

        Continuation files must be self-contained, so we serialize historical
        models into in-memory state dicts instead of keeping stale temp paths.
        """
        return {
            "cross_task_weight": self.cross_task_weight,
            "top_k": self.top_k,
            "rounds_per_task": self.rounds_per_task,
            "current_task": self.current_task,
            "_round_in_task": self._round_in_task,
            "client_representations": {
                int(client_id): {
                    int(task_id): clone_representation_state(rep)
                    for task_id, rep in task_map.items()
                }
                for client_id, task_map in self.client_representations.items()
            },
            "task_global_models": {
                int(task_id): serialize_model_reference(model_ref)
                for task_id, model_ref in self.task_global_models.items()
            },
            "client_historical_models": {
                int(client_id): {
                    int(task_id): serialize_model_reference(model_ref)
                    for task_id, model_ref in task_map.items()
                }
                for client_id, task_map in self.client_historical_models.items()
            },
            "task_representations": {
                int(task_id): rep.detach().cpu().clone()
                for task_id, rep in self.task_representations.items()
            },
            "task_representation_matrices": {
                int(task_id): clone_representation_state(rep)
                for task_id, rep in self.task_representation_matrices.items()
            },
            "_current_similarity_weights": dict(self._current_similarity_weights),
            "_current_historical_models": {
                key: serialize_model_reference(model_ref)
                for key, model_ref in self._current_historical_models.items()
            },
        }

    def load_resume_state(self, state: Dict[str, Any]) -> None:
        """Restore aggregator state without requiring old history artifact files."""
        self.cross_task_weight = float(
            state.get("cross_task_weight", self.cross_task_weight)
        )
        self.top_k = int(state.get("top_k", self.top_k))
        self.rounds_per_task = int(state.get("rounds_per_task", self.rounds_per_task))
        self.current_task = int(state.get("current_task", self.current_task))
        self._round_in_task = int(state.get("_round_in_task", 0))

        self.client_representations = {
            int(client_id): {
                int(task_id): clone_representation_state(rep)
                for task_id, rep in task_map.items()
            }
            for client_id, task_map in state.get("client_representations", {}).items()
        }
        self.task_global_models = {
            int(task_id): restore_model_reference(params)
            for task_id, params in state.get("task_global_models", {}).items()
        }
        self.client_historical_models = {
            int(client_id): {
                int(task_id): restore_model_reference(params)
                for task_id, params in task_map.items()
            }
            for client_id, task_map in state.get("client_historical_models", {}).items()
        }
        self.task_representations = {
            int(task_id): rep.detach().cpu().clone()
            for task_id, rep in state.get("task_representations", {}).items()
            if isinstance(rep, torch.Tensor)
        }
        self.task_representation_matrices = {
            int(task_id): clone_representation_state(rep)
            for task_id, rep in state.get("task_representation_matrices", {}).items()
        }
        self._current_similarity_weights = {
            key: float(value)
            for key, value in state.get("_current_similarity_weights", {}).items()
        }
        self._current_historical_models = {
            key: restore_model_reference(params)
            for key, params in state.get("_current_historical_models", {}).items()
        }

    def _store_client_representations(self, results: List[Dict]):
        """
        Trích xuất và lưu representation từ kết quả client.

        Các representation này là đầu vào để đo độ tương đồng giữa task hiện tại
        và các task lịch sử.
        """
        task_reps: List[Dict[str, Any]] = []

        for r in results:
            if "representation" in r and r["representation"] is not None:
                client_id = r["client_id"]
                rep_artifact = build_representation_artifact(r["representation"])
                if rep_artifact is None:
                    print(f"  ⚠️ Skipping NaN representation from client {client_id}")
                    continue

                # Store per-client per-task
                if client_id not in self.client_representations:
                    self.client_representations[client_id] = {}
                self.client_representations[client_id][self.current_task] = rep_artifact

                task_reps.append(rep_artifact)

        # Build a lightweight task-level artifact from all client representations.
        if task_reps:
            task_artifact = aggregate_representation_artifacts(task_reps)
            if task_artifact is not None:
                if not hasattr(self, "task_representation_matrices"):
                    self.task_representation_matrices: Dict[int, Dict[str, Any]] = {}
                self.task_representation_matrices[self.current_task] = task_artifact

                mean_vector = task_artifact.get("mean_vector")
                if isinstance(mean_vector, torch.Tensor):
                    self.task_representations[self.current_task] = mean_vector.detach().cpu().clone()

                shape = task_artifact.get("shape")
                total_samples = int(shape[0]) if shape is not None else 0
                print(
                    f"  → Stored {len(task_reps)} client representation artifacts for task {self.current_task}, "
                    f"total samples: {total_samples}, shape: {shape}"
                )

    def _compute_similarity(self, R1: torch.Tensor, R2: torch.Tensor) -> float:
        """
        Compute similarity between two representation matrices (Paper Eq. 10).

        Paper Eq. 10: Uses NEGATIVE Frobenius norm distance on representation matrices.
        sim(R1, R2) = -||R1 - R2||_F

        Where R1, R2 are representation matrices [n_samples, hidden_dim] from Eq. 2.
        Closer representations → smaller Frobenius distance → higher (less negative) similarity.

        Note: Using activation-based representation matrices (not gradient-based).
        """
        return compute_representation_similarity(R1, R2)

    def _select_top_k_similar(self) -> List[Dict]:
        """
        Select TOP-K most similar historical task models.

        Paper Eq. 10: Select K historical models with highest similarity
        using representation matrices R^t (not mean vectors).

        Returns:
            List of {task_id, similarity, params}
        """
        if self.current_task == 0:
            return []

        # Use representation matrices for proper similarity computation (Paper Eq. 10)
        if not hasattr(self, "task_representation_matrices"):
            return []

        current_rep_state = self.task_representation_matrices.get(self.current_task)
        if current_rep_state is None:
            return []

        # Compute similarity with all previous tasks using matrices
        similarities = []
        for tid in range(self.current_task):
            if (
                tid in self.task_representation_matrices
                and tid in self.task_global_models
            ):
                hist_rep_state = self.task_representation_matrices[tid]
                # Compute similarity on full matrices (Paper Eq. 10)
                sim = self._compute_similarity(current_rep_state, hist_rep_state)
                similarities.append(
                    {
                        "task_id": tid,
                        "similarity": sim,
                        "params": self.task_global_models[tid],
                    }
                )

        # DEBUG[10]: Log raw similarity scores and representation info
        print(f"  DEBUG[10]: Cross-task similarity details (task {self.current_task}):")
        current_shape = representation_shape(current_rep_state)
        current_mean_norm = representation_mean_norm(current_rep_state)
        if current_mean_norm is not None:
            print(
                f"    Current rep matrix shape: {current_shape}, "
                f"mean norm: {current_mean_norm:.4f}"
            )
        else:
            print(f"    Current rep matrix shape: {current_shape}, mean norm: n/a")
        for s in similarities:
            tid = s["task_id"]
            hist_shape = representation_shape(self.task_representation_matrices[tid])
            print(
                f"    Task {tid}: neg_l2_dist={s['similarity']:.6f} | "
                f"matrix shape: {hist_shape}"
            )
        if len(similarities) >= 2:
            sims = [s["similarity"] for s in similarities]
            print(
                f"    Sim range: [{min(sims):.6f}, {max(sims):.6f}], "
                f"diff={max(sims) - min(sims):.6f}"
            )
            # Show what softmax would produce (with max-subtraction normalization)
            sim_t = torch.tensor(sims)
            sim_t_norm = sim_t - sim_t.max()
            sw = F.softmax(sim_t_norm, dim=0)
            print(
                f"    normalized({[f'{s:.4f}' for s in sim_t_norm.tolist()]}) "
                f"→ softmax = {[f'{w:.4f}' for w in sw.tolist()]}"
            )

        # Sort by similarity (descending) and select TOP-K
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        selected = similarities[: self.top_k]

        return selected

    def aggregate(
        self, results: List[Dict], global_params: Optional[OrderedDict] = None, **kwargs
    ) -> OrderedDict:
        """
        Aggregate with cross-task regularization (paper Section 5.2, Algorithm 1).

        Paper Algorithm 1 (ON SERVER):
        - Standard FedAvg aggregation of client updates
        - Store representations for cross-task similarity (Paper Eq. 10)
        - Save historical models for future reference
        - Compute similarity weights for NEXT task's local regularization (Eq. 14)

        NOTE: Paper Eq. 11 A(Θ) is a LOCAL training loss term (applied in
        compute_loss()), NOT a server-side model blend. The server only computes
        similarities and stores historical data. Per-client personalized
        aggregation (Eq. 12) is handled by CGoFedServer.

        Steps:
        1. Standard weighted average of client updates (FedAvg)
        2. Store client representations for similarity (Paper Eq. 2)
        3. Save current model for future reference (Paper Eq. 12)
        4. Select TOP-K similar historical models (Paper Eq. 10)
        5. Weighted blend with history (Paper Eq. 11) - ONLY at end of task
        """
        # Track round within task (Issue 5)
        self._round_in_task += 1
        is_last_round = self._round_in_task >= self.rounds_per_task

        # 1. Standard weighted average (FedAvg)
        agg_params = self._weighted_average(results)

        # 2. Store representations from clients (Paper Eq. 2)
        self._store_client_representations(results)

        # 3. Save per-client models for personalized aggregation (Paper Eq. 12)
        # Store each client's specific params (not aggregated global params)
        for result in results:
            client_id = result.get("client_id")
            if client_id is not None and "params" in result:
                if client_id not in self.client_historical_models:
                    self.client_historical_models[client_id] = {}
                # Store client's OWN model at end of task (not agg_params)
                # This is the client's local model after training, before aggregation
                if is_last_round:
                    self.client_historical_models[client_id][self.current_task] = (
                        persist_model_artifact(
                            result["params"],
                            history_dir=self._client_history_dir,
                            artifact_key=f"client_{client_id}_task_{self.current_task}",
                        )
                    )

        # 3b. Save global model for task (at end of task)
        if is_last_round:
            self.task_global_models[self.current_task] = persist_model_artifact(
                agg_params,
                history_dir=self._global_history_dir,
                artifact_key=f"global_task_{self.current_task}",
            )

        # 4. Compute cross-task similarity at end of task (Paper Eq. 10)
        # Store similarity weights and historical models for NEXT task's local
        # training regularization (Paper Eq. 11, 14).
        # NOTE: Paper Eq. 11 A(Θ) is a LOCAL training loss term applied in
        # compute_loss(), NOT a server-side model blend. The server only
        # computes similarities and stores historical data here.
        # Paper Eq. 12 per-client personalization is handled in CGoFedServer.
        if self.current_task > 0 and is_last_round:
            selected = self._select_top_k_similar()
            if selected:
                task_ids = [s["task_id"] for s in selected]
                print(
                    f"📊 Cross-task: Selected TOP-{len(selected)} similar tasks: {task_ids}"
                )
                # Store similarity weights and historical models for local regularization
                # Paper Eq. 14: These will be passed to clients for A(Θ) computation
                self._current_similarity_weights = {
                    s["task_id"]: float(
                        torch.softmax(
                            torch.tensor([x["similarity"] for x in selected]), dim=0
                        )[i]
                    )
                    for i, s in enumerate(selected)
                }
                self._current_historical_models = {
                    s["task_id"]: self.task_global_models[s["task_id"]]
                    for s in selected
                }

        return agg_params
