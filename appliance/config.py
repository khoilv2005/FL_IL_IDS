from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Protocol:
    version: str = 'appliance_pairwise_v1'
    integration_mode: str = 'supplement'
    compatibility: str = 'exact'
    wire_dtype: str = 'float32'
    max_patch_bytes: int = 32768
    max_incoming_bytes: int = 131072
    max_outgoing_bytes: int = 524288
    min_positive_count: int = 32
    min_predicted_count: int = 32
    min_quality_lcb: float = .50
    acceptance_min_rows: int = 32
    acceptance_max_rows: int = 256
    max_accuracy_drop: float = 0.
    max_new_pinned_fraction: float = .10
    allow_unverified_positive_install: bool = True
    seed: int = 20261007
    fidelity_atol: float = 1e-6
    fidelity_rtol: float = 1e-5

    def validate(self):
        if (self.integration_mode != 'supplement' or self.compatibility != 'exact'
                or self.wire_dtype != 'float32'):
            raise ValueError('V1 implements supplement/exact/FP32 only')
        for key in ('max_patch_bytes','max_incoming_bytes','max_outgoing_bytes',
                    'min_positive_count','min_predicted_count','acceptance_min_rows','acceptance_max_rows'):
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError(key)
        if self.acceptance_min_rows>self.acceptance_max_rows:raise ValueError('Acceptance row bounds')
        for key in ('max_new_pinned_fraction','min_quality_lcb','max_accuracy_drop'):
            value=getattr(self,key)
            if not math.isfinite(value) or not 0<=value<=1:raise ValueError(key)
        for key in ('fidelity_atol','fidelity_rtol'):
            if not math.isfinite(getattr(self,key)) or getattr(self,key)<0:raise ValueError(key)
        return self

    def locked(self):return asdict(self.validate())


class Rejected(Exception):
    def __init__(self,reason,detail=''):
        self.reason=reason;self.detail=detail
        super().__init__(f'{reason}: {detail}')
