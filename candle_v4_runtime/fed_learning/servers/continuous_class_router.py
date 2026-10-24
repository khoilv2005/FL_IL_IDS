"""Class-aware continuous activation memory. No raw input retention."""
from copy import deepcopy
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

VERSION='continuous_class_v1'
LAYERS=('conv1','conv2','conv3','gru')


class ContinuousClassMemory:
    def __init__(self,per_class):
        self.per_class=int(per_class)
        if self.per_class<1:raise ValueError('Positive memory cap required')
        self.indices=None;self.full_width=None;self.latest_task=-1
        self.memory={};self.class_to_task={};self.scaler=None;self.classifier=None;self.fitted_classes=None

    @staticmethod
    def flatten(activations):
        arrays=[np.asarray(activations[k],dtype=np.float32) for k in LAYERS]
        if any(a.ndim!=2 for a in arrays) or len({len(a) for a in arrays})!=1:
            raise ValueError('Invalid layer activation layout')
        x=np.concatenate(arrays,axis=1)
        if not np.isfinite(x).all():raise ValueError('Nonfinite continuous activations')
        return x

    @staticmethod
    def selected(mask,width):
        if mask is None:raise ValueError('Continuous router requires a stable feature mask')
        mask=np.asarray(mask)
        if mask.shape!=(width,) or not np.isin(mask,[0,1]).all() or not mask.any():
            raise ValueError('Invalid stable routing feature mask')
        return np.flatnonzero(mask)

    def encode(self,activations,mask):
        x=self.flatten(activations);indices=self.selected(mask,x.shape[1])
        if self.indices is None:raise ValueError('Continuous memory uninitialized; fresh stream required')
        if x.shape[1]!=self.full_width or not np.array_equal(indices,self.indices):
            raise ValueError('Continuous routing feature coordinates changed')
        return x[:,self.indices].copy()

    def push(self,activations,labels,task,mask):
        task=int(task);y=np.asarray(labels)
        x=self.flatten(activations);indices=self.selected(mask,x.shape[1])
        if not 0<=task<=5 or task<self.latest_task:
            raise ValueError('Historical task writes are forbidden')
        if y.shape!=(len(x),) or y.dtype.kind not in 'iu' or not len(y) or np.any((y<0)|(y>=34)) or np.any(y//6!=task):
            raise ValueError('Current-task class labels required for continuous memory')
        if self.indices is not None and (self.full_width!=x.shape[1] or not np.array_equal(indices,self.indices)):
            raise ValueError('Continuous routing feature coordinates changed')
        current={int(c):x[y==c][:self.per_class,indices].copy() for c in np.unique(y)}
        # Replace current-task samples only. Historical feature matrices are untouched.
        self.memory={c:v for c,v in self.memory.items() if self.class_to_task[c]!=task}
        self.memory.update(current);self.class_to_task={c:c//6 for c in self.memory}
        self.indices=indices.copy();self.full_width=x.shape[1];self.latest_task=task
        self.scaler=None;self.classifier=None;self.fitted_classes=None

    def fit(self,task):
        if int(task)!=self.latest_task or not self.memory:raise ValueError('No current continuous memory to fit')
        classes=np.array(sorted(self.memory),dtype=np.int64)
        x=np.concatenate([self.memory[int(c)] for c in classes]);y=np.concatenate([np.full(len(self.memory[int(c)]),c,dtype=np.int64) for c in classes])
        self.scaler=StandardScaler().fit(x)
        if len(classes)>1:
            self.classifier=LogisticRegression(C=1.,max_iter=1000,solver='lbfgs',class_weight='balanced',random_state=0)
            self.classifier.fit(self.scaler.transform(x),y)
        self.fitted_classes=classes

    def predict(self,features):
        x=np.asarray(features,dtype=np.float32)
        if self.fitted_classes is None or self.scaler is None:raise RuntimeError('Continuous router is not fitted')
        if x.ndim!=2 or x.shape[1]!=len(self.indices) or not np.isfinite(x).all():raise ValueError('Invalid continuous routing input')
        if not len(x):return np.empty(0,dtype=np.int64),np.zeros((0,self.latest_task+1))
        if len(self.fitted_classes)==1:
            pred=np.full(len(x),self.class_to_task[int(self.fitted_classes[0])],dtype=np.int64)
            scores=np.zeros((len(x),self.latest_task+1));scores[:,pred[0]]=1.;return pred,scores
        transformed=self.scaler.transform(x);p=self.classifier.predict_proba(transformed)
        classes=self.classifier.classes_;pred=self.classifier.predict(transformed)//6
        scores=np.zeros((len(x),self.latest_task+1),dtype=np.float64)
        for j,c in enumerate(classes):scores[:,self.class_to_task[int(c)]]=np.maximum(scores[:,self.class_to_task[int(c)]],p[:,j])
        # Compatibility scores, NOT calibrated task posterior probabilities.
        scores/=scores.sum(1,keepdims=True)
        return pred.astype(np.int64),scores

    def stats(self):
        return dict(classes=len(self.memory),rows=sum(len(v) for v in self.memory.values()),
                    selected_features=0 if self.indices is None else len(self.indices),
                    feature_bytes=sum(v.nbytes for v in self.memory.values()))

    def state_dict(self):
        scaler=None if self.scaler is None else {k:deepcopy(getattr(self.scaler,k)) for k in ['mean_','var_','scale_','n_features_in_','n_samples_seen_']}
        clf=None if self.classifier is None else {k:deepcopy(getattr(self.classifier,k)) for k in ['classes_','coef_','intercept_','n_iter_','n_features_in_']}
        return deepcopy(dict(version=VERSION,per_class=self.per_class,indices=self.indices,full_width=self.full_width,
            latest_task=self.latest_task,memory=self.memory,class_to_task=self.class_to_task,
            fitted_classes=self.fitted_classes,scaler=scaler,classifier=clf,stats=self.stats()))

    @classmethod
    def restore(cls,state):
        if state.get('version')!=VERSION:raise ValueError('Unknown continuous memory version')
        obj=cls(state['per_class'])
        for k in ['indices','full_width','latest_task','memory','class_to_task','fitted_classes']:setattr(obj,k,deepcopy(state[k]))
        if not -1<=obj.latest_task<=5:raise ValueError('Invalid latest task')
        if obj.indices is not None:
            if obj.indices.ndim!=1 or obj.indices.dtype.kind not in 'iu' or not len(obj.indices) or np.any(obj.indices<0) or np.any(obj.indices>=obj.full_width) or not np.array_equal(obj.indices,np.unique(obj.indices)):
                raise ValueError('Invalid continuous feature coordinates')
        if set(obj.memory)!=set(obj.class_to_task):raise ValueError('Invalid class/task mapping')
        for c,v in obj.memory.items():
            if obj.indices is None or not 0<=c<34 or obj.class_to_task[c]!=c//6 or c//6>obj.latest_task or v.ndim!=2 or v.shape[1]!=len(obj.indices) or not 0<len(v)<=obj.per_class or not np.isfinite(v).all():
                raise ValueError('Invalid continuous memory rows')
        if state['scaler'] is not None:
            obj.scaler=StandardScaler()
            for k,v in state['scaler'].items():setattr(obj.scaler,k,deepcopy(v))
        if state['classifier'] is not None:
            obj.classifier=LogisticRegression(C=1.,max_iter=1000,solver='lbfgs',class_weight='balanced',random_state=0)
            for k,v in state['classifier'].items():setattr(obj.classifier,k,deepcopy(v))
        if obj.fitted_classes is not None:
            if not np.array_equal(obj.fitted_classes,sorted(obj.memory)) or obj.scaler is None or (len(obj.fitted_classes)>1 and obj.classifier is None):raise ValueError('Incomplete continuous fitted state')
            width=len(obj.indices)
            if obj.scaler.n_features_in_!=width or any(np.asarray(getattr(obj.scaler,k)).shape!=(width,) or not np.isfinite(getattr(obj.scaler,k)).all() for k in ['mean_','var_','scale_']) or np.any(obj.scaler.scale_<=0):
                raise ValueError('Invalid continuous scaler state')
            if obj.classifier is not None:
                rows=1 if len(obj.fitted_classes)==2 else len(obj.fitted_classes)
                if (not np.array_equal(obj.classifier.classes_,obj.fitted_classes) or obj.classifier.n_features_in_!=width or
                        obj.classifier.coef_.shape!=(rows,width) or obj.classifier.intercept_.shape!=(rows,) or
                        not np.isfinite(obj.classifier.coef_).all() or not np.isfinite(obj.classifier.intercept_).all()):
                    raise ValueError('Invalid continuous classifier state')
        elif obj.scaler is not None or obj.classifier is not None:raise ValueError('Stale continuous fitted state')
        return obj
