"""
Tests for GLFC server/client regression cases.
"""

from collections import OrderedDict

import torch
import torch.nn as nn

from fed_learning.clients.glfc_client import GLFCClient
from fed_learning.servers.glfc_server import GLFCServer
from fed_learning.strategies.fed_incremental.glfc import GLFCTrainer


class TestGLFC:
    def test_glfc_server_update_clients(self):
        """GLFCServer should support task-loop client refresh across tasks."""
        clients_a = [GLFCClient(0, torch.randn(4, 32), torch.randint(0, 2, (4,)))]
        clients_b = [GLFCClient(1, torch.randn(5, 32), torch.randint(0, 2, (5,)))]
        server = GLFCServer(
            clients=clients_a,
            test_data={"X_test": torch.randn(8, 32), "y_test": torch.randint(0, 2, (8,))},
            config={"input_shape": (32,), "num_classes": 34, "num_gpus": 0},
        )

        server.update_clients(clients_b)
        assert server.clients is clients_b

    def test_glfc_prototype_gradients_use_full_output_dim(self):
        """Prototype BCE target must match the model's full classifier width."""
        client = GLFCClient(
            client_id=0,
            X_train=torch.randn(6, 32),
            y_train=torch.tensor([0, 0, 0, 1, 1, 1]),
        )
        client.signal = True
        client.current_class = [0, 1]

        model = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 34),
        )
        trainer = GLFCTrainer()
        trainer.numclass = 6

        grads = client.compute_prototype_gradients(model, trainer, device="cpu")

        assert grads is not None
        assert len(grads) == 2
        assert all(isinstance(g, list) for g in grads)
