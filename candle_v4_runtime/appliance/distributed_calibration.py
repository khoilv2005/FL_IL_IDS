"""Current-task, client-local guard calibration using aggregate messages.

No historical dataset API, pooled raw input, validation/test labels, or donor
model at inference. The result certifies CURRENT calibration support only.
Historical protection and post-drift survival remain separate requirements;
this module does not silently turn a current-only certificate into a main-method
commit authorization.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .codec import encode,decode
from .config import Rejected
from .guarded_head import HeadContract


MAX_AGGREGATE_BYTES=16*1024*1024
ROLES=('fit','selection','holdout')


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class CalibrationSession:
    receiver:int
    donor:int
    class_id:int
    task:int
    round_id:int
    current_classes:tuple
    receiver_snapshot_sha:str
    capability_sha:str
    role_manifest_sha:str
    nonce:str

    def manifest(self):
        d=dict(vars(self),current_classes=list(self.current_classes),
               protocol='appliance_current_client_calibration_v1')
        if (self.receiver==self.donor or self.class_id not in self.current_classes
                or not 0<=self.task<6 or self.round_id<0 or not self.nonce):raise Rejected('INVALID_CALIBRATION_SESSION')
        return d

    @property
    def session_id(self):return fingerprint(self.manifest())


class LocalCalibrationEndpoint:
    """One client's ephemeral task-local splits. Only aggregates leave the API.

    Signals come from the fixed receiver function executed on this client's own
    inputs. A caller must send/verify that function capsule separately. Labels
    are used locally for counts after prediction, never as predictor inputs.
    """
    def __init__(self,client,session,views):
        self.client=int(client);self.session=session;self._views=views;self.selection_lock=None
        if self.client not in (session.receiver,session.donor):raise Rejected('CALIBRATION_CLIENT_NOT_IN_SESSION')
        session.manifest();seen=set()
        if set(views)!=set(ROLES):raise Rejected('CALIBRATION_SPLIT_REQUIRED')
        for role,v in views.items():
            y=np.asarray(v['y'],np.int64);rows=np.asarray(v['row_id'],np.int64)
            s=v['signals'];n=len(y)
            if (not set(map(int,np.unique(y))).issubset(session.current_classes)
                    or len(rows)!=n or len(np.unique(rows))!=n or set(map(int,rows))&seen):
                raise Rejected('CALIBRATION_TASK_OR_ROLE_OVERLAP')
            if any(np.asarray(s[k]).shape!=(n,) for k in ('signature_score','signature_valid','margin','local_confidence','local_pred')):
                raise Rejected('CALIBRATION_SIGNAL_SHAPE')
            if any(not np.isfinite(s[k]).all() for k in ('signature_score','margin','local_confidence')):
                raise Rejected('NONFINITE_CALIBRATION_SIGNAL')
            if self.client==session.receiver and np.any(y==session.class_id):raise Rejected('RECEIVER_TARGET_NOT_MISSING')
            seen.update(map(int,rows))

    def quantiles(self):
        s=self._views['selection']['signals'];q=np.linspace(0,1,31)
        if len(s['margin'])==0:raise Rejected('EMPTY_CURRENT_SELECTION')
        return dict(session_id=self.session.session_id,client=self.client,role='selection',
            quantiles={k:np.quantile(s[k],q).tolist() for k in ('signature_score','margin','local_confidence')})

    def count_packet(self,grid,role='selection'):
        if role not in ('selection','holdout'):raise Rejected('COUNT_ROLE_FORBIDDEN')
        grid.validate(self.session)
        if role=='holdout' and self.selection_lock!=grid.grid_id:raise Rejected('HOLDOUT_BEFORE_GUARD_LOCK')
        v=self._views[role];s=v['signals'];y=np.asarray(v['y']);n=len(y)
        positive=y==self.session.class_id;negative=~positive
        labels=sorted(map(int,np.unique(y)))
        channels=['positive_activated','negative_activated','broken_correct','rescued_wrong']
        masks=[positive,negative,negative & (s['local_pred']==y),positive & (s['local_pred']!=y)]
        channels += [f'activated_class_{c}' for c in labels]
        masks += [y==c for c in labels]
        order=np.argsort(s['local_confidence'],kind='stable')
        cuts=np.searchsorted(np.asarray(s['local_confidence'])[order],grid.betas,side='left')
        ordered=[m[order] for m in masks];blocks=[]
        for tau in grid.taus:
            hit=np.asarray(s['signature_valid']) & (s['signature_score']>tau)
            for gamma in grid.gammas:
                hit_ordered=(hit & (s['margin']>gamma))[order]
                blocks.append(np.stack([np.r_[0,np.cumsum(hit_ordered&m)][cuts] for m in ordered],axis=1))
        counts=np.concatenate(blocks).astype(np.float32)
        meta=dict(session_id=self.session.session_id,grid_id=grid.grid_id,client=self.client,role=role,
            rows=n,positive_rows=int(positive.sum()),negative_rows=int(negative.sum()),channels=channels,
            class_rows={str(c):int((y==c).sum()) for c in labels},
            coordinate_sha256=hashlib.sha256(np.asarray(v['row_id'],dtype='<i8').tobytes()).hexdigest())
        return encode(meta,{'counts':counts},MAX_AGGREGATE_BYTES)

    def lock_guard(self,grid):
        grid.validate(self.session)
        if len(grid.values)!=1:raise Rejected('SINGLE_FINAL_GUARD_REQUIRED')
        if self.selection_lock is not None:raise Rejected('GUARD_ALREADY_LOCKED')
        self.selection_lock=grid.grid_id


class GuardGrid:
    def __init__(self,session,taus,gammas,betas):
        self.session_id=session.session_id
        self.taus=np.unique(np.asarray(taus,np.float64));self.gammas=np.unique(np.asarray(gammas,np.float64))
        self.betas=np.unique(np.asarray(betas,np.float64))
        self.validate(session)

    @property
    def values(self):return [(float(t),float(g),float(b)) for t in self.taus for g in self.gammas for b in self.betas]

    @property
    def grid_id(self):return fingerprint(dict(session_id=self.session_id,taus=self.taus.tolist(),gammas=self.gammas.tolist(),betas=self.betas.tolist()))

    def validate(self,session):
        if self.session_id!=session.session_id:raise Rejected('GUARD_SESSION_CHANGED')
        if (any(not len(v) or len(v)>64 or not np.isfinite(v).all() for v in (self.taus,self.gammas,self.betas))
                or (self.taus<-1).any() or (self.taus>1).any() or (self.gammas<0).any()
                or (self.betas<0).any() or (self.betas>1).any()):raise Rejected('GUARD_GRID_SCHEMA')

    @classmethod
    def from_quantiles(cls,session,messages,minimum_signature=-1.):
        if len(messages)!=2 or {v['client'] for v in messages}!={session.receiver,session.donor}:
            raise Rejected('BOTH_CALIBRATION_ENDPOINTS_REQUIRED')
        if any(v['session_id']!=session.session_id or v['role']!='selection' for v in messages):raise Rejected('QUANTILE_SCOPE_CHANGED')
        def joined(name):return np.concatenate([np.asarray(v['quantiles'][name],np.float64) for v in messages])
        if not np.isfinite(minimum_signature) or not -1<=minimum_signature<=1:
            raise Rejected('INVALID_FIT_SIGNATURE_FLOOR')
        taus=np.clip(np.r_[-1.,1.,joined('signature_score')],-1,1)
        if not np.isfinite(taus).all():raise Rejected('NONFINITE_SELECTION_QUANTILES')
        # The donor-FIT support envelope is fixed before selection. Selection
        # may tighten it, but cannot expand the imported class outside it.
        taus=np.unique(np.r_[minimum_signature,taus[taus>=minimum_signature]])
        return cls(session,taus,np.maximum(0,np.r_[0.,joined('margin')]),
                   np.clip(np.r_[0.,1.,joined('local_confidence')],0,1))


def read_counts(packet,session,grid,client,role):
    grid.validate(session);m,t=decode(packet,MAX_AGGREGATE_BYTES)
    if any(m.get(k)!=v for k,v in dict(session_id=session.session_id,grid_id=grid.grid_id,client=client,role=role).items()):
        raise Rejected('CALIBRATION_COUNT_SCOPE_CHANGED')
    if set(t)!={'counts'}:raise Rejected('CALIBRATION_COUNT_TENSORS')
    counts=t['counts']
    expected=['positive_activated','negative_activated','broken_correct','rescued_wrong']
    labels=sorted(map(int,m['class_rows']))
    if (m['channels']!=expected+[f'activated_class_{c}' for c in labels]
            or counts.shape!=(len(grid.values),len(m['channels'])) or (counts<0).any()
            or not np.array_equal(counts,np.floor(counts)) or counts.max(initial=0)>m['rows']
            or m['rows']>=2**24 or m['positive_rows']+m['negative_rows']!=m['rows']
            or sum(m['class_rows'].values())!=m['rows'] or not set(labels).issubset(session.current_classes)):
        raise Rejected('CALIBRATION_COUNT_SCHEMA')
    if not np.array_equal(counts[:,4:].sum(1),counts[:,0]+counts[:,1]):raise Rejected('CALIBRATION_COUNTS_INCONSISTENT')
    if (m['positive_rows']!=m['class_rows'].get(str(session.class_id),0)
            or (counts[:,0]>m['positive_rows']).any() or (counts[:,1]>m['negative_rows']).any()
            or any((counts[:,4+i]>m['class_rows'][str(c)]).any() for i,c in enumerate(labels))):
        raise Rejected('CALIBRATION_COUNTS_INCONSISTENT')
    if (counts[:,2]>counts[:,1]).any() or (counts[:,3]>counts[:,0]).any():raise Rejected('CALIBRATION_COUNTS_INCONSISTENT')
    return m,counts.astype(np.int64)


def select_distributed_guard(session,grid,receiver_packet,donor_packet,contract=HeadContract()):
    rm,rc=read_counts(receiver_packet,session,grid,session.receiver,'selection')
    dm,dc=read_counts(donor_packet,session,grid,session.donor,'selection')
    if rm['rows']<contract.min_receiver_rows or dm['positive_rows']<8 or rm['positive_rows']:
        raise Rejected('INSUFFICIENT_CURRENT_SELECTION')
    feasible=(rc[:,1]/rm['rows']<=contract.max_receiver_far) & (rc[:,2]+dc[:,2]<=contract.max_break)
    for pos,label in enumerate(sorted(map(int,rm['class_rows']))):
        feasible &= rc[:,4+pos]/rm['class_rows'][str(label)]<=contract.max_receiver_far
    candidates=np.flatnonzero(feasible)
    if not len(candidates):raise Rejected('NO_CURRENT_SCOPE_GUARD')
    values=grid.values
    index=max(candidates,key=lambda i:(dc[i,0],-(rc[i,1]+dc[i,1]),-values[i][2],values[i][1],values[i][0]))
    tau,gamma,beta=values[index]
    return dict(tau=tau,gamma=gamma,beta=beta,grid_id=grid.grid_id,index=int(index),
        selection_positive_rows=dm['positive_rows'],selection_target_activations=int(dc[index,0]),
        receiver_rows=rm['rows'],receiver_false_activations=int(rc[index,1]),
        pooled_break=int(rc[index,2]+dc[index,2]),selection_locked_before_holdout=True)


def current_acceptance(session,grid,receiver_packet,donor_packet,contract=HeadContract()):
    if len(grid.values)!=1:raise Rejected('SINGLE_FINAL_GUARD_REQUIRED')
    rm,rc=read_counts(receiver_packet,session,grid,session.receiver,'holdout')
    dm,dc=read_counts(donor_packet,session,grid,session.donor,'holdout')
    if rm['rows']<contract.min_receiver_rows or dm['positive_rows']<contract.min_positive_rows:
        raise Rejected('INSUFFICIENT_CURRENT_ACCEPTANCE')
    per_class={str(c):float(rc[0,4+i]/rm['class_rows'][str(c)]) for i,c in enumerate(sorted(map(int,rm['class_rows'])))}
    recall=float(dc[0,0]/dm['positive_rows']);far=float(rc[0,1]/rm['rows'])
    broken=int(rc[0,2]+dc[0,2]);rescue=int(rc[0,3]+dc[0,3])
    passed=(recall>=contract.min_recall and far<=contract.max_receiver_far
        and all(v<=contract.max_receiver_far for v in per_class.values()) and broken<=contract.max_break and rescue>0)
    return dict(passed_current_scope=bool(passed),recall=recall,receiver_far=far,
        receiver_far_by_class=per_class,break_count=broken,rescue=rescue,positive_rows=dm['positive_rows'],
        receiver_rows=rm['rows'],historical_classes_certified=False,main_install_authorized=False,
        session_id=session.session_id,guard_id=grid.grid_id,thresholds_retuned=False,
        limitation='current task only; cannot certify all old classes or positive function after backbone drift')
