"""
Seed utility - Set random seed for reproducibility.

Extracted from train_incremental_kaggle.py to be reusable
across training scripts and tests.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set random seed for full reproducibility.

    Affects: Python random, NumPy, PyTorch (CPU + all CUDA devices),
    and cuDNN (deterministic mode).

    Args:
        seed: Random seed value (default: 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Seed set to {seed}")
