"""Clients module"""

from .client import FederatedClient
from .cgofed_client import CGoFedClient
from .der_client import DERClient
from .nice_client import NICEClient
from .glfc_client import GLFCClient
from .refed_client import ReFedClient
from .plexus_der_client import PlexusDERClient
from .plexus_nice_client import PlexusNICEClient
from .dfca_client import DFCAClient

__all__ = [
    "FederatedClient",
    "CGoFedClient",
    "DERClient",
    "NICEClient",
    "GLFCClient",
    "ReFedClient",
    "PlexusDERClient",
    "PlexusNICEClient",
    "DFCAClient",
]
