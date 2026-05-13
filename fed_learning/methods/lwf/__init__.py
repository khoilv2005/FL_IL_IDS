"""
LwF (Learning without Forgetting) Implementation for CNN-GRU.

This module implements the LwF method for class-incremental learning using
the CNN-GRU architecture from DeepFed paper.

Reference:
    Li & Hoiem, "Learning without Forgetting", ECCV 2016, IEEE TPAMI 2018

Paper URL: https://arxiv.org/abs/1606.09282

Key Method from Paper:
    L_total = L_CE(new_data) + α * L_KD(old_model, new_model)

Where:
    - L_CE: Cross-entropy loss on new task data
    - L_KD: Knowledge distillation loss (distillation from old model)
    - α: Balance weight between CE and KD (default: 1.0)

Author's Original Implementation (model.py):
    - Uses ResNet34 as backbone
    - Incremental classifier expansion
    - Multi-class cross-entropy for distillation

This Implementation:
    - Uses CNN-GRU as backbone (from DeepFed paper)
    - Maintains same LwF logic
    - Adapts to IDS/network intrusion detection domain

Files:
    - lwf_trainer.py: Main trainer class with LwF loss computation
    - lwf_model.py: Model wrapper for easy usage
    - lwf_main.py: Training script and CLI interface

Usage:
    from fed_learning.methods.lwf import LwFModel
    
    model = LwFModel(input_shape=(46,), num_classes=6)
    model.set_task([6, 7, 8, 9, 10, 11])
    model.train(train_loader)
"""

from .lwf_trainer import (
    MultiClassCrossEntropy,
    kaiming_normal_init,
    LwFTrainer,
    CNN_GRU_LwF,
)
from .lwf_model import LwFModel, create_lwf_model

__all__ = [
    # Core functions
    'MultiClassCrossEntropy',
    'kaiming_normal_init',
    # Trainer class
    'LwFTrainer',
    # Convenience classes
    'CNN_GRU_LwF',
    'LwFModel',
    'create_lwf_model',
]
