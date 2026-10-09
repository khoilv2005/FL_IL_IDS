"""Observed provenance stays distinct from installed and quality evidence."""
from dataclasses import dataclass,field,asdict
import math
from .config import Rejected


def wilson_lower(h,n,z=1.96):
    if n==0:return None
    if not 0<=h<=n:raise ValueError('Invalid evidence counts')
    p=h/n
    return (p+z*z/(2*n)-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)


@dataclass
class Ledger:
    observed: dict = field(default_factory=dict)
    installed: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)

    def observe(self,class_id,count,task,source):
        if count<0:raise ValueError('Negative local count')
        if count:self.observed[int(class_id)]=dict(count=int(count),task=int(task),source=source)

    def missing(self,seen):
        return sorted(c for c in seen if c not in self.observed and
                      self.installed.get(c,{}).get('valid',False) is not True)

    def install(self,class_id,patch_id,donor,boundary_hash):
        # Deliberately never mutates observed or creates receiver-positive evidence.
        self.installed[int(class_id)]=dict(patch_id=patch_id,donor=int(donor),valid=True,
            boundary_hash=boundary_hash,receiver_positive_quality='unknown')

    def invalidate(self,class_id,reason):
        entry=self.installed.get(int(class_id))
        if entry:entry.update(valid=False,reason=reason)

    def to_dict(self):return asdict(self)


def quality(counts,protocol):
    n=int(counts['positive']);h=int(counts['correct_positive']);p=int(counts['predicted'])
    if min(n,p)<min(protocol.min_positive_count,protocol.min_predicted_count):
        raise Rejected('INSUFFICIENT_DONOR_EVIDENCE',f'positive={n}, predicted={p}')
    if n<protocol.min_positive_count or p<protocol.min_predicted_count:
        raise Rejected('INSUFFICIENT_DONOR_EVIDENCE')
    result=min(wilson_lower(h,n),wilson_lower(h,p))
    if result<protocol.min_quality_lcb:raise Rejected('LOW_DONOR_QUALITY',str(result))
    return result
