"""
Base Trainer - Abstract base class for local training strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class BaseTrainer(ABC):
    """
    Abstract base class for local training strategies.
    
    Different learning algorithms (FedAvg, FedProx, EWC, etc.) extend this
    class to implement their specific training logic.
    """
    
    @property
    def name(self) -> str:
        """Return the name of this trainer."""
        return self.__class__.__name__.replace("Trainer", "")
    
    def compute_loss(
        self, 
        model: nn.Module,
        output: torch.Tensor, 
        target: torch.Tensor,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute loss for training.
        
        Override this method to add regularization terms (e.g., proximal term
        for FedProx, EWC penalty for incremental learning).
        
        Args:
            model: The model being trained
            output: Model predictions
            target: Ground truth labels
            global_params: Global model parameters (for regularization)
            **kwargs: Additional algorithm-specific parameters
            
        Returns:
            Loss tensor
        """
        return self._seen_class_cross_entropy(output, target)

    def _seen_class_cross_entropy(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Cross entropy over classes already introduced in the incremental stream.

        Fixed-head models such as CNN_GRU_Model always produce logits for all
        34 classes. In class-incremental training, unseen output nodes should
        not participate in softmax, otherwise they receive negative gradients
        before their task arrives. Trainers that expose `seen_classes` get a
        sliced-logit CE with remapped targets; other trainers fall back to
        standard CE.
        """
        seen_classes = getattr(self, "seen_classes", None)
        if not seen_classes:
            return F.cross_entropy(output, target, reduction=reduction)

        num_classes = int(output.shape[1])
        class_ids = sorted(
            {
                int(cls_id)
                for cls_id in seen_classes
                if 0 <= int(cls_id) < num_classes
            }
        )
        if not class_ids or len(class_ids) >= num_classes:
            return F.cross_entropy(output, target, reduction=reduction)

        device = output.device
        class_tensor = torch.tensor(class_ids, dtype=torch.long, device=device)
        mapping = torch.full((num_classes,), -1, dtype=torch.long, device=device)
        mapping[class_tensor] = torch.arange(len(class_ids), device=device)

        target_long = target.long()
        target_in_range = (target_long >= 0) & (target_long < num_classes)
        if not bool(target_in_range.all()):
            return F.cross_entropy(output, target, reduction=reduction)

        remapped = mapping[target_long]
        if not bool((remapped >= 0).all()):
            return F.cross_entropy(output, target, reduction=reduction)

        seen_logits = output.index_select(dim=1, index=class_tensor)
        return F.cross_entropy(seen_logits, remapped, reduction=reduction)
    
    def pre_step(
        self,
        model: nn.Module,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> None:
        """
        Pre-optimization step hook (AFTER backward, BEFORE optimizer.step).
        
        Override this method to modify gradients before optimizer applies them
        (e.g., CGoFed gradient projection onto old task subspace).
        
        Called between loss.backward() and optimizer.step().
        
        Args:
            model: The model being trained
            global_params: Global model parameters
            **kwargs: Additional algorithm-specific parameters
        """
        pass
    
    def post_step(
        self,
        model: nn.Module,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> None:
        """
        Post-optimization step hook (AFTER optimizer.step).
        
        Override this method to apply corrections after optimizer.step()
        (e.g., Fed+ correction step).
        
        Args:
            model: The model being trained
            global_params: Global model parameters
            **kwargs: Additional algorithm-specific parameters
        """
        pass
    
    def pre_train(
        self,
        model: nn.Module,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> None:
        """
        Pre-training hook.
        
        Override for any setup needed before training starts.
        
        Args:
            model: The model being trained
            global_params: Global model parameters
            **kwargs: Additional algorithm-specific parameters
        """
        pass
    
    def post_train(
        self,
        model: nn.Module,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> None:
        """
        Post-training hook.
        
        Override for any cleanup or final adjustments after training.
        
        Args:
            model: The model being trained
            global_params: Global model parameters
            **kwargs: Additional algorithm-specific parameters
        """
        pass
    
    def get_optimizer_class(self) -> type:
        """
        Return the optimizer class to use.
        
        Override to use different optimizers (e.g., SGD for Fed+).
        
        Returns:
            Optimizer class (default: Adam)
        """
        return torch.optim.Adam
