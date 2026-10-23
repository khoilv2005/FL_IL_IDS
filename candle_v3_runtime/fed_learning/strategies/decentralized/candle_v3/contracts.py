"""Versioned, explicit contracts for receiver-specific current-task teaching."""
from dataclasses import asdict, dataclass
import math

SCHEMA = 'candle_v3_peer_teaching_v1'


@dataclass(frozen=True)
class TeachingConfig:
    enabled: bool = False
    transfer_mode: str = 'receiver_head_learning'
    head_rounds: int = 8
    learning_rate: float = .01
    max_donors: int = 2
    max_targets: int = 3
    batch_size: int = 512
    base_per_class: int = 4096
    fit_per_class: int = 256
    cal_per_class: int = 512
    sketch_per_class: int = 50
    min_fit_positive: int = 32
    min_cal_positive: int = 32
    min_cal_negative: int = 32
    recall_gate: float = .95
    far_gate: float = .001
    min_gain: float = .001
    communication_budget_mib: float = 128.
    graph: str = 'announced_current_peers'

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('enabled must be a boolean')
        if self.transfer_mode not in ('receiver_head_learning', 'direct_fc2'):
            raise ValueError('Unknown transfer mode; no automatic fallback between variants')
        for name in ('head_rounds', 'max_donors', 'max_targets', 'batch_size',
                     'base_per_class', 'fit_per_class', 'cal_per_class',
                     'sketch_per_class', 'min_fit_positive', 'min_cal_positive',
                     'min_cal_negative'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('learning_rate', 'communication_budget_mib'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('recall_gate', 'far_gate', 'min_gain'):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must lie in [0,1]')
        if self.graph != 'announced_current_peers':
            raise ValueError('Explicit teaching graph contract required')

    @classmethod
    def from_config(cls, config):
        values = config.get('candle_v3_teaching', {})
        if not isinstance(values, dict):
            raise ValueError('candle_v3_teaching must be an object')
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'Unknown teaching controls: {sorted(unknown)}')
        return cls(**values)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SnapshotIdentity:
    receiver: int
    task: int
    reference_sha256: str
    preprocessing_sha256: str
    feature_contract: str = 'denice_masked_eval_penultimate_v1'
    schema: str = SCHEMA

    def __post_init__(self):
        if self.receiver < 0 or self.task < 0:
            raise ValueError('Invalid receiver/task')
        for value in (self.reference_sha256, self.preprocessing_sha256):
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Full SHA-256 identity required')


@dataclass(frozen=True)
class PeerAnnouncement:
    client: int
    task: int
    current_counts: tuple  # ((class_id, actual current BASE count), ...)
    local_seen_classes: tuple
    registered_classes: tuple
    peer_acquired_classes: tuple
    preprocessing_sha256: str
    architecture: str
    schema: str = SCHEMA

    def __post_init__(self):
        if self.client < 0 or self.task < 0:
            raise ValueError('Invalid announcement owner/task')
        labels = [c for c, _ in self.current_counts]
        if len(labels) != len(set(labels)) or any(c < 0 or n <= 0 for c, n in self.current_counts):
            raise ValueError('Current counts must contain unique classes and positive counts')
        if len(self.preprocessing_sha256) != 64 or any(c not in '0123456789abcdef' for c in self.preprocessing_sha256):
            raise ValueError('Announcement requires preprocessing SHA-256')

    @property
    def current_classes(self):
        return set(c for c, _ in self.current_counts)
