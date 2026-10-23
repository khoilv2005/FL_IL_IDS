"""CANDLE V3 peer teaching. No imports or changes to the V1 runtime."""

from .contracts import TeachingConfig, SnapshotIdentity, PeerAnnouncement
from .knowledge import record_local_update, announce
from .discovery import select_teachers

__all__ = ['TeachingConfig', 'SnapshotIdentity', 'PeerAnnouncement',
           'record_local_update', 'announce', 'select_teachers']
