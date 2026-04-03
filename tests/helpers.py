"""
Shared test helpers for federated learning tests.
"""

import sys
import os
import copy
from collections import OrderedDict

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fed_learning.models import CNN_GRU_Model


def make_simple_model(input_dim=10, num_classes=5):
    """Create a simple linear model for testing."""
    model = nn.Sequential(
        nn.Linear(input_dim, 20),
        nn.ReLU(),
        nn.Linear(20, num_classes),
    )
    return model


def make_cnn_gru_model(seq_length=40, num_classes=10):
    """Create a CNN-GRU model for testing."""
    return CNN_GRU_Model(input_shape=(seq_length,), num_classes=num_classes)


def make_client_results(num_clients=3, input_dim=10, num_classes=5):
    """Create fake client results for aggregation tests."""
    results = []
    for i in range(num_clients):
        model = make_simple_model(input_dim, num_classes)
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.1 * (i + 1))
        results.append(
            {
                "params": copy.deepcopy(model.state_dict()),
                "num_samples": 100 * (i + 1),
                "loss": 0.5 - 0.1 * i,
            }
        )
    return results


def make_dataloader(num_samples=32, input_dim=10, num_classes=5, batch_size=8):
    """Create a simple DataLoader for testing."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)
