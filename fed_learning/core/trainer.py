"""
Base Trainer - Abstract base class for local training strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
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
        return nn.CrossEntropyLoss()(output, target)
    
    def post_step(
        self,
        model: nn.Module,
        global_params: Optional[OrderedDict] = None,
        **kwargs
    ) -> None:
        """
        Post-optimization step hook.
        
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
