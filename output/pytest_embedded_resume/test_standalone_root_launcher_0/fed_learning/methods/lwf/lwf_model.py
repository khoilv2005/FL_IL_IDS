"""
LwF Model Wrapper for CNN-GRU.

This module provides a clean model wrapper that combines:
1. CNN-GRU architecture (from DeepFed paper)
2. LwF training strategy (from Li & Hoiem)

The model can be used directly for incremental learning without
needing to separately manage trainer and model.

Usage:
    from fed_learning.methods.lwf.lwf_model import LwFModel
    
    model = LwFModel(input_shape=(46,), num_classes=6)
    model.set_task([6, 7, 8, 9, 10, 11])  # New task classes
    model.train(train_loader)
    accuracy = model.evaluate(test_loader)
"""

from typing import List, Tuple, Optional, Dict, Any
from collections import OrderedDict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from fed_learning.models.cnn_gru import CNN_GRU_Model
from .lwf_trainer import MultiClassCrossEntropy, kaiming_normal_init


class LwFModel(nn.Module):
    """
    CNN-GRU model with Learning without Forgetting capability.
    
    This class wraps the CNN-GRU architecture and adds LwF-specific
    functionality for incremental learning:
    - Model snapshot saving for distillation
    - Classifier expansion for new classes
    - Combined CE + KD loss computation
    
    Architecture:
        - CNN: 1D convolutional layers for local pattern extraction
        - GRU: Gated recurrent unit for temporal dependencies
        - MLP: Fully connected layers with dropout for classification
    
    Training Strategy (LwF):
        - Save old model snapshot before learning new task
        - Train with CE loss on new data + KD loss on old predictions
        - Expand classifier for new classes
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_classes: int = 1,
        init_lr: float = 0.001,
        num_epochs: int = 20,
        batch_size: int = 64,
        lwf_alpha: float = 1.0,
        temperature: float = 2.0,
        momentum: float = 0.9,
        weight_decay: float = 0.0001,
    ):
        """
        Initialize LwF CNN-GRU model.
        
        Args:
            input_shape: Input shape (e.g., (46,) for 46 timesteps)
            num_classes: Initial number of classes (usually 1 for incremental)
            init_lr: Initial learning rate
            num_epochs: Number of epochs per task
            batch_size: Batch size
            lwf_alpha: Weight for distillation loss (α in paper)
            temperature: Temperature for distillation (T in paper)
            momentum: SGD momentum
            weight_decay: L2 weight decay
        """
        super().__init__()
        
        # Store hyperparameters
        self.input_shape = input_shape
        self.init_lr = init_lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lwf_alpha = lwf_alpha
        self.temperature = temperature
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # Build CNN-GRU model
        self.backbone = CNN_GRU_Model(input_shape, num_classes=num_classes)
        
        # Incremental learning state
        self.n_classes = num_classes
        self.n_known = num_classes
        self.n_old_classes_for_kd = 0
        self.classes_map: Dict[int, int] = {i: i for i in range(num_classes)}
        self.reverse_map: Dict[int, int] = {i: i for i in range(num_classes)}
        
        # Old model snapshot for distillation
        self.prev_model: Optional[nn.Module] = None
        
        # Training state
        self.current_task = 0
        self.is_training = False
        
        # Output layer reference
        self.fc = self.backbone.fc2
        
        print(f"[LwFModel] Initialized with input_shape={input_shape}, "
              f"num_classes={num_classes}, α={lwf_alpha}, T={temperature}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN-GRU.
        
        Args:
            x: Input tensor of shape [batch, seq_len] or [batch, seq_len, features]
            
        Returns:
            Logits of shape [batch, num_classes]
        """
        return self.backbone(x)
    
    def get_fused_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get fused CNN+GRU representation.
        
        This is useful for visualization or analysis.
        
        Args:
            x: Input tensor
            
        Returns:
            Fused feature vector
        """
        return self.backbone.get_fused_representation(x)
    
    def set_task(self, new_classes: List[int]) -> None:
        """
        Prepare model for a new task.
        
        This should be called before training on a new task.
        It expands the classifier and saves the old model snapshot.
        
        Args:
            new_classes: List of new class IDs for this task
        """
        self.n_old_classes_for_kd = self.n_classes if self.current_task > 0 else 0
        if self.current_task > 0:
            self._save_prev_model()
        
        # Expand classifier for new classes
        self._increment_classes(new_classes)
        
        self.current_task += 1
        print(f"[LwFModel] Task {self.current_task}: Added classes {new_classes}, "
              f"total classes: {self.n_classes}")
    
    def _save_prev_model(self) -> None:
        """
        Save current model snapshot for distillation.
        
        The saved snapshot is used as the teacher model to compute
        knowledge distillation loss for subsequent tasks.
        """
        if self.n_classes > 0:
            self.prev_model = copy.deepcopy(self.backbone)
            self.prev_model.eval()
            # Freeze parameters
            for param in self.prev_model.parameters():
                param.requires_grad = False
            
            print(f"[LwFModel] Saved model snapshot (n_classes={self.n_classes})")
    
    def _increment_classes(self, new_classes: List[int]) -> None:
        """
        Expand classifier to accommodate new classes.
        
        This creates a new output layer with more units, preserving
        the weights for existing classes.
        
        Args:
            new_classes: List of new class IDs to add
        """
        unseen_classes = [cls for cls in new_classes if cls not in self.classes_map]
        n_new = len(unseen_classes)
        
        # Get current classifier
        old_fc = self.backbone.fc2
        in_features = old_fc.in_features
        out_features = old_fc.out_features
        old_weight = old_fc.weight.data.clone()
        
        # Calculate new output size
        required_outputs = max(new_classes) + 1 if new_classes else out_features
        new_out_features = max(out_features + n_new, required_outputs, out_features)
        if new_out_features == out_features:
            for cls in new_classes:
                if cls not in self.classes_map and cls < out_features:
                    self.classes_map[cls] = cls
            self.reverse_map = {v: k for k, v in self.classes_map.items()}
            self.n_classes = out_features
            self.n_known = self.n_classes
            print(f"[LwFModel] Expanded classifier: {out_features} -> {new_out_features}")
            return
        
        # Create new classifier
        new_fc = nn.Linear(in_features, new_out_features, bias=False)
        new_fc.apply(kaiming_normal_init)
        
        # Copy old weights
        new_fc.weight.data[:out_features] = old_weight
        
        # Replace classifier
        self.backbone.fc2 = new_fc
        self.fc = new_fc
        
        # Update class tracking
        for cls in new_classes:
            if cls not in self.classes_map:
                self.classes_map[cls] = cls if cls < new_out_features else len(self.classes_map)
        
        # Update reverse map
        self.reverse_map = {v: k for k, v in self.classes_map.items()}
        
        # Update counters
        self.n_classes = new_out_features
        self.n_known = self.n_classes
        
        print(f"[LwFModel] Expanded classifier: {out_features} -> {new_out_features}")
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute LwF loss (CE + KD).
        
        Args:
            logits: Model outputs
            labels: Ground truth labels
            inputs: Input data (for distillation)
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Remap labels to internal indices
        remapped_labels = torch.tensor(
            [self.classes_map.get(int(l), int(l)) for l in labels],
            device=logits.device,
            dtype=torch.long
        )
        
        # Cross-entropy loss on new data
        ce_loss = F.cross_entropy(logits, remapped_labels)
        
        loss_dict = {'ce_loss': ce_loss.item(), 'kd_loss': 0.0}
        total_loss = ce_loss
        
        # Knowledge distillation loss (if not first task)
        if self.prev_model is not None and self.n_old_classes_for_kd > 0:
            with torch.no_grad():
                old_logits = self.prev_model(inputs)
            
            # Distill on old class logits only
            num_old_classes = self.n_old_classes_for_kd
            if num_old_classes > 0:
                kd_loss = MultiClassCrossEntropy(
                    logits[:, :num_old_classes],
                    old_logits[:, :num_old_classes],
                    self.temperature
                )
                total_loss = ce_loss + self.lwf_alpha * kd_loss
                loss_dict['kd_loss'] = kd_loss.item()
        
        return total_loss, loss_dict
    
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        optimizer: optim.Optimizer
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            inputs: Input batch
            labels: Label batch
            optimizer: Optimizer
            
        Returns:
            Dictionary of loss values
        """
        self.train()
        
        optimizer.zero_grad()
        logits = self(inputs)
        loss, loss_dict = self._compute_loss(logits, labels, inputs)
        loss.backward()
        optimizer.step()

        if self.prev_model is None:
            self.prev_model = copy.deepcopy(self.backbone)
            self.prev_model.eval()
            for param in self.prev_model.parameters():
                param.requires_grad = False
        
        return loss_dict
    
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify input data.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class indices (internal)
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            _, preds = torch.max(logits, dim=1)
        return preds
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict with original class labels.
        
        Args:
            x: Input tensor
            
        Returns:
            Predicted class IDs (original)
        """
        internal_preds = self.classify(x)
        original_preds = torch.tensor(
            [self.reverse_map.get(int(p), int(p)) for p in internal_preds],
            device=internal_preds.device
        )
        return original_preds
    
    def get_params(self) -> OrderedDict:
        """Get model parameters as state dict."""
        return OrderedDict(
            (k, v.cpu().clone()) for k, v in self.state_dict().items()
        )
    
    def set_params(self, state_dict: OrderedDict) -> None:
        """Set model parameters from state dict."""
        self.load_state_dict({k: v for k, v in state_dict.items()})
    
    @classmethod
    def from_state_dict(cls, state_dict: OrderedDict, **kwargs) -> 'LwFModel':
        """
        Create model from state dict.
        
        Args:
            state_dict: Model state dictionary
            **kwargs: Additional arguments for initialization
            
        Returns:
            New model instance with loaded weights
        """
        # Extract input shape from state dict if not provided
        if 'input_shape' not in kwargs:
            # Try to infer from model architecture
            pass
        
        model = cls(**kwargs)
        model.set_params(state_dict)
        return model


def create_lwf_model(
    input_shape: Tuple[int, ...],
    num_classes: int = 6,
    **kwargs
) -> LwFModel:
    """
    Factory function to create LwF CNN-GRU model.
    
    Args:
        input_shape: Input shape (e.g., (46,) for 46 timesteps)
        num_classes: Initial number of classes
        **kwargs: Additional arguments for LwFModel
        
    Returns:
        LwFModel instance
    """
    return LwFModel(input_shape=input_shape, num_classes=num_classes, **kwargs)
