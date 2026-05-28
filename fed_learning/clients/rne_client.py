"""RNE client.

RNE reuses DER's replay buffer, two-stage local training, and exemplar update
machinery. The model itself supplies the recurrent expert chain and decoupled
classifier heads.
"""

from .der_client import DERClient


class RNEClient(DERClient):
    """Client for Recurrent Network Expansion."""

    pass
