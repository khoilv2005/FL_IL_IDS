"""
Factories module - Create algorithm-specific clients and servers.

Centralizes the creation logic so that the main training script
only needs to provide a CONFIG dict.
"""

from .client_factory import create_clients
from .server_factory import create_server

__all__ = [
    "create_clients",
    "create_server",
]
