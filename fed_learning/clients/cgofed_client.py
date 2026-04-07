"""
CGoFed Client - Specialized client for Class Incremental Learning.

Reference:
    "CGoFed: Constrained Gradient Optimization Strategy for Federated Class
    Incremental Learning", IEEE TKDE, 2025

Extends FederatedClient with activation-based representation computation
for cross-task similarity (paper Section 5.2, Eq. 2, 10).
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional, Dict, Any, List
from torch.utils.data import DataLoader, TensorDataset

from .client import FederatedClient
from ..core import BaseTrainer


class CGoFedClient(FederatedClient):
    """
    Client for CGoFed algorithm with activation representation computation.

    Inherits all standard FL functionality from FederatedClient,
    adds compute_activation_representation() for cross-task similarity.

    Paper Reference:
    - Representation R is computed as activation vector from last hidden layer (Paper Eq. 2)
    - Used by server to compute similarity between tasks (Paper Eq. 10)
    - Enables personalized aggregation with historical models (Paper Eq. 12)
    """

    def train(
        self,
        trainer: BaseTrainer,
        epochs: int,
        batch_size: int,
        lr: float,
        global_params: Optional[OrderedDict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train and compute activation representation for CGoFed.

        Extends parent train() by computing representation after training.
        This representation is sent to server for cross-task similarity.
        """
        # Standard training with gradient projection (via trainer.pre_step)
        result = super().train(trainer, epochs, batch_size, lr, global_params, **kwargs)

        # Compute activation representation for cross-task similarity (Paper Eq. 2, 10)
        if self.model is not None:
            requested_samples = getattr(trainer, "num_samples_rep", self.num_samples)
            try:
                requested_samples = int(requested_samples)
            except (TypeError, ValueError):
                requested_samples = self.num_samples
            if requested_samples <= 0:
                requested_samples = self.num_samples
            result["representation"] = self.compute_activation_representation(
                model=self.model,
                num_samples=requested_samples,
            )

        return result

    def build_representation_loader(
        self, num_samples: Optional[int], batch_size: int = 32
    ) -> Optional[DataLoader]:
        """
        Build a small local loader for representation-space construction.

        This keeps post-task basis construction client-local instead of
        centralizing raw tensors into a server-side aggregate dataset.
        """
        if self.num_samples <= 0:
            return None

        if num_samples is None or num_samples <= 0:
            n_available = self.num_samples
        else:
            n_available = min(num_samples, self.num_samples)
        if n_available <= 0:
            return None

        indices = torch.randperm(self.num_samples)[:n_available]
        dataset = TensorDataset(self.X_train[indices], self.y_train[indices])
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def compute_activation_representation(
        self, model: Optional[nn.Module], num_samples: Optional[int] = None
    ) -> torch.Tensor:
        """
        Compute activation representation matrix from last hidden layer (Paper Eq. 2).

        Paper CGoFed Section 5.1, Eq. 2:
        R^t = F(Θ^t, X^t) = [z_1, ..., z_n]^T - representation matrix from forward propagation

        We collect activations from the last hidden layer (fc1) which produces
        a representation matrix suitable for SVD (Eq. 3) and cross-task similarity (Eq. 10).

        Args:
            model: The trained model
            num_samples: Number of samples to use for activation computation

        Returns:
            Activation representation matrix [num_samples, hidden_dim]
        """
        if model is None:
            return torch.zeros(256)  # Default hidden dimension

        model.eval()  # type: ignore
        device = next(model.parameters()).device  # type: ignore

        # Sample indices
        if num_samples is None or num_samples <= 0:
            n_available = self.num_samples
        else:
            n_available = min(num_samples, self.num_samples)
        indices = torch.randperm(self.num_samples)[:n_available]

        activations = []

        # Hook to capture INPUT activations from last hidden layer
        # Consistent with SVD code in CGoFedTrainer._collect_activations()
        activation = {}

        def get_activation(name):
            def hook(module, inp, output):
                if isinstance(inp, tuple) and len(inp) > 0:
                    activation[name] = inp[0].detach()
                else:
                    activation[name] = inp.detach()

            return hook

        # Register hook on fc1 (last hidden layer before output)
        handle = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "fc1" in name:
                handle = module.register_forward_hook(get_activation("fc1"))
                break

        if handle is None:
            # Fallback: use penultimate layer
            linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
            if len(linear_layers) >= 2:
                handle = linear_layers[-2].register_forward_hook(
                    get_activation("hidden")
                )

        try:
            with torch.no_grad():
                # Process in small batches to avoid memory issues
                batch_size = 32
                for i in range(0, len(indices), batch_size):
                    batch_idx = indices[i : i + batch_size]
                    X_batch = self.X_train[batch_idx].to(device)

                    # Forward pass to get activations
                    _ = model(X_batch)

                    # Get activation from hook
                    if activation:
                        key = list(activation.keys())[0]
                        act = activation[key]
                        if act.dim() > 2:
                            # Flatten if needed (e.g., conv output)
                            act = act.view(act.size(0), -1)
                        activations.append(act.cpu())
        finally:
            if handle is not None:
                handle.remove()

        # Return representation matrix R^t (Paper Eq. 2)
        # R^t has shape [num_samples, hidden_dim] for SVD computation (Eq. 3)
        if activations:
            all_activations = torch.cat(activations, dim=0)
            return all_activations  # Return matrix [n_samples, hidden_dim], not mean vector
        else:
            # Return zero matrix if no activations collected
            # Try to infer hidden dim from model
            hidden_dim = 256  # Default
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    hidden_dim = module.out_features
            return torch.zeros(num_samples, hidden_dim)
