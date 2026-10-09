"""Task-wise routing summaries; no historical rows in the runtime state.

Each class is covered by a fixed budget of balls in the shared sketch space.
Vetoing every stored ball guarantees rejection of the observations enclosed at
construction, independent of later backbone/router drift. This is a finite
support guarantee, not a population FAR certificate or a positive-recall proof.
"""
import numpy as np
from sklearn.cluster import KMeans

from .config import Rejected
from .portable_route import SharedSketch


class SketchBallMemory:
    VERSION='appliance_streaming_sketch_balls_v1'
    SLACK=1e-5

    def __init__(self,sketch,balls_per_class=8,seed=20261009,representation="unit_sketch",min_group_rows=1):
        if not 1<=int(balls_per_class)<=8:raise ValueError('Ball budget outside locked range')
        if representation not in ('unit_sketch','preprocessed_input'):raise ValueError('Unknown fixed memory representation')
        if min_group_rows not in (1,8) or representation!='unit_sketch' and min_group_rows!=1:raise ValueError('Unsupported aggregate group policy')
        self.min_group_rows=min_group_rows
        self.sketch=sketch;self.budget=int(balls_per_class);self.seed=int(seed)
        self.representation=representation
        self.width=sketch.dimension if representation=='unit_sketch' else int(np.prod(sketch.input_shape))
        self.task=-1;self.entries={};self.events=set()

    def _features(self,x):
        if self.representation=='unit_sketch':return self.sketch.features(x)
        x=np.asarray(x,np.float32)
        if tuple(x.shape[1:])!=tuple(self.sketch.input_shape) or not np.isfinite(x).all():
            raise Rejected('SHARED_INPUT_MEMORY_MISMATCH')
        return x.reshape(len(x),self.width),np.ones(len(x),bool)

    def observe(self,task,classes,x,y,event_id):
        task=int(task);classes=set(map(int,classes));y=np.asarray(y,np.int64)
        if task<self.task:raise Rejected('HISTORICAL_CALIBRATION_REOPENED')
        if str(event_id) in self.events:raise Rejected('CALIBRATION_EVENT_REPLAYED')
        if len(x)!=len(y) or not set(map(int,np.unique(y))).issubset(classes):raise Rejected('CALIBRATION_TASK_SCOPE_CHANGED')
        if any(c<0 or c>=34 for c in classes):raise Rejected('CALIBRATION_CLASS_RANGE')
        z,valid=self._features(x)
        if not valid.all():raise Rejected('UNBOUNDED_ZERO_NORM_MEMORY_OBSERVATION')
        if any(int(c) in self.entries for c in np.unique(y)):raise Rejected('CLASS_SUMMARY_ALREADY_SEALED')
        pending={}
        for c in np.unique(y):
            values=z[y==c]
            # Each centroid summarizes a group, never a retained support row.
            k=min(self.budget,max(1,len(values)//8),len(np.unique(values,axis=0)))
            if k==1:labels=np.zeros(len(values),np.int64)
            else:labels=KMeans(k,n_init=1,random_state=self.seed+int(c),max_iter=100).fit_predict(values)
            if self.min_group_rows==8:
                # Merge small KMeans cells; do not retain one feature per outlier.
                while len(np.unique(labels))>1:
                    ids,amounts=np.unique(labels,return_counts=True)
                    small=ids[amounts<8]
                    if not len(small):break
                    chosen=int(small[0])
                    center=values[labels==chosen].mean(0).astype(np.float64)
                    others=[int(v) for v in ids if v!=chosen]
                    neighbor=min(others,key=lambda v:(float(np.linalg.norm(values[labels==v].mean(0)-center)),v))
                    labels[labels==chosen]=neighbor
            centers=[];radii=[];counts=[]
            for label in np.unique(labels):
                members=values[labels==label];center=members.mean(0).astype(np.float32)
                radius=float(np.linalg.norm(members.astype(np.float64)-center,axis=1).max())+self.SLACK
                centers.append(center);radii.append(radius);counts.append(len(members))
            pending[int(c)]=dict(task=task,count=len(values),centers=np.asarray(centers,np.float32),
                radii=np.asarray(radii,np.float64),cluster_counts=counts)
        self.entries.update(pending)
        self.events.add(str(event_id));self.task=task
        return dict(task=task,rows=len(y),classes=sorted(map(int,np.unique(y))))

    def veto(self,x,classes=None):
        z,valid=self._features(x)
        rejected=~valid
        for c in (sorted(self.entries) if classes is None else sorted(set(map(int,classes)))):
            if c not in self.entries:raise Rejected('MISSING_OLD_CLASS_ROUTING_SUMMARY',str(c))
            e=self.entries[c]
            for center,radius in zip(e['centers'],e['radii']):
                rejected|=np.linalg.norm(z.astype(np.float64)-center,axis=1)<=radius
        return rejected


    def projection_bounds(self,prototype,tau,classes):
        """Worst-case signature-hit counts over enclosed finite observations.

        No Gaussian assumption, per-example history or backbone evaluation.
        Radius + unit-sphere intersection constrains each summary ball. A ball
        whose conservative score upper bound exceeds tau remains ambiguous;
        all its observations are counted. Missing class evidence stays missing.
        """
        if self.representation!='unit_sketch':
            raise Rejected('UNIT_SKETCH_BALL_QUERY_REQUIRED')
        w=np.asarray(prototype)
        if (w.dtype!=np.float32 or w.shape!=(self.width,) or not np.isfinite(w).all() or
                not np.isclose(np.linalg.norm(w.astype(np.float64)),1,rtol=1e-5,atol=1e-6) or
                not np.isfinite(tau) or not -1<=tau<=1 or
                any(type(c) is not int or not 0<=c<34 for c in classes)):
            raise Rejected('INVALID_BALL_PROJECTION_QUERY')
        d=self.width;wf=w.astype(np.float64);wn=float(np.linalg.norm(wf))
        eps=np.finfo(np.float32).eps
        gamma=(2*d+2)*eps/(1-(2*d+2)*eps)
        fp_error=gamma*1.0001*float(np.abs(wf).sum())
        result={};missing=[]
        for c in sorted(set(classes)):
            if c not in self.entries:
                missing.append(c);continue
            entry=self.entries[c];balls=[];upper_count=0
            for center,radius,n in zip(entry['centers'],entry['radii'],entry['cluster_counts']):
                upper=_unit_ball_score_upper(center,float(radius),wf)+fp_error
                ambiguous=bool(tau<1 and upper>tau)
                upper_count+=n if ambiguous else 0
                balls.append(dict(rows=n,score_upper=float(min(1.,upper)),ambiguous=ambiguous))
            result[str(c)]=dict(rows=entry['count'],activation_count_upper=upper_count,
                far_upper=upper_count/entry['count'],balls=balls)
        return dict(per_class=result,missing_classes=missing,
            fp32_dot_error_allowance=fp_error,
            scope='finite observations enclosed by the stored balls only',
            unseen_population_far_certified=False,main_install_authorized=False)

    def state(self):
        result=dict(version=self.VERSION,signature=self.sketch.manifest(),balls_per_class=self.budget,
            seed=self.seed,task=self.task,consumed_events=sorted(self.events),
            entries={str(c):dict(v,centers=v['centers'].tolist(),radii=v['radii'].tolist()) for c,v in self.entries.items()},
            retained_raw_examples=0,retained_per_sample_features=0,
            guarantee='veto on enclosed observations only; no population FAR or positive-recall guarantee')
        if self.representation!='unit_sketch':
            result.update(version='appliance_streaming_input_balls_v2',representation=self.representation)
        elif self.min_group_rows==8:
            result.update(version='appliance_streaming_sketch_balls_min_group8_v3',min_group_rows=8)
        return result

    @classmethod
    def restore(cls,state):
        if state['version'] not in (cls.VERSION,'appliance_streaming_input_balls_v2','appliance_streaming_sketch_balls_min_group8_v3'):raise Rejected('SKETCH_BALL_VERSION')
        expected_fields={'version','signature','balls_per_class','seed','task','consumed_events','entries','retained_raw_examples','retained_per_sample_features','guarantee'}
        if state['version']=='appliance_streaming_input_balls_v2':expected_fields.add('representation')
        if state['version']=='appliance_streaming_sketch_balls_min_group8_v3':expected_fields.add('min_group_rows')
        if set(state)!=expected_fields or state['retained_raw_examples']!=0 or state['retained_per_sample_features']!=0:
            raise Rejected('SKETCH_MEMORY_STATE_SCHEMA_CHANGED')
        representation=state.get('representation','unit_sketch')
        if (state['version']=='appliance_streaming_input_balls_v2')!=(representation=='preprocessed_input'):
            raise Rejected('SKETCH_BALL_REPRESENTATION_VERSION')
        if (state['version']=='appliance_streaming_sketch_balls_min_group8_v3')!=(state.get('min_group_rows',1)==8):raise Rejected('SKETCH_MEMORY_GROUP_VERSION')
        s=state['signature']
        result=cls(SharedSketch(tuple(s['input_shape']),s['dimension'],s['preprocessing_sha256'],s['seed']),
                   state['balls_per_class'],state['seed'],representation,state.get('min_group_rows',1))
        if result.sketch.manifest()!=s:raise Rejected('SKETCH_MEMORY_ENCODER_CHANGED')
        result.task=int(state['task']);result.events=set(state['consumed_events'])
        if type(result.task) is not int or not -1<=result.task<=5 or len(result.events)!=len(state['consumed_events']):raise Rejected('SKETCH_MEMORY_HISTORY_CHANGED')
        for c,e in state['entries'].items():
            if (set(e)!={'task','count','centers','radii','cluster_counts'} or str(int(c))!=c or not 0<=int(c)<34 or type(e['task']) is not int or not 0<=e['task']<=result.task or type(e['count']) is not int or e['count']<1):raise Rejected('SKETCH_MEMORY_ENTRY_SCHEMA_CHANGED')
            centers=np.asarray(e['centers'],np.float32);radii=np.asarray(e['radii'],np.float64)
            if (centers.ndim!=2 or centers.shape[1]!=result.width or not 1<=len(centers)<=result.budget
                    or radii.shape!=(len(centers),) or len(e['cluster_counts'])!=len(centers) or any(type(n) is not int or n<1 for n in e['cluster_counts']) or not np.isfinite(centers).all()
                    or not np.isfinite(radii).all() or (radii<0).any() or sum(e['cluster_counts'])!=e['count']):
                raise Rejected('INVALID_SKETCH_BALL_STATE')
            if representation=='unit_sketch' and np.linalg.norm(centers.astype(np.float64),axis=1).max()>1.0001:raise Rejected('NONUNIT_SKETCH_MEMORY_CENTER')
            if result.min_group_rows==8 and e['count']>=8 and any(n<8 for n in e['cluster_counts']):
                raise Rejected('SKETCH_MEMORY_GROUP_TOO_SMALL')
            result.entries[int(c)]=dict(e,centers=centers,radii=radii)
        return result


def _unit_ball_score_upper(center,radius,weights):
    """Max w.z on intersection ||z-center||<=R and ||z||<=1.0001.

    Returns a conservative FP64 geometry bound; the caller adds the FP32 dot
    allowance. Unsupported/numerically ambiguous geometry falls back to the
    looser ball-only bound, never an optimistic zero certificate.
    """
    c=np.asarray(center,np.float64);w=np.asarray(weights,np.float64)
    if c.shape!=w.shape or not np.isfinite(c).all() or not np.isfinite(w).all() or not np.isfinite(radius) or radius<0:
        raise Rejected('INVALID_UNIT_BALL_GEOMETRY')
    a=1.0001;wn=float(np.linalg.norm(w));cn=float(np.linalg.norm(c))
    if not wn:return 0.
    dot=float(c@w);simple=min(dot+radius*wn,a*wn)
    tolerance=1e-8*(1+abs(simple)+cn+radius)
    if cn>a+1e-8:return simple+tolerance
    direction=w/wn
    # Ball-only optimum lies inside the outer unit sphere.
    if np.linalg.norm(c+radius*direction)<=a:
        return simple+tolerance
    # Unit-sphere optimum lies inside the summary ball.
    if np.linalg.norm(a*direction-c)<=radius:
        return a*wn+tolerance
    if cn<=1e-12:return simple+tolerance
    k=(a*a+cn*cn-radius*radius)/2.
    disc=a*a-k*k/(cn*cn)
    transverse=wn*wn-dot*dot/(cn*cn)
    if disc<0 or transverse< -1e-10:return simple+tolerance
    value=k*dot/(cn*cn)+np.sqrt(max(0.,disc))*np.sqrt(max(0.,transverse))
    return min(simple,float(value))+tolerance
