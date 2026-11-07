"""Task/OOD-inspired discriminative routing on saved TRAIN features, no raw replay.

Not a reproduction of MORE, which uses transformer heads and exemplar replay.
Binary scores are uncalibrated and only compared inside original Top2.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from pilot_candle_v4_five_router_methods import main

METHODS=('ovr_task','ovr_class_max')

class OVRScorers:
    def __init__(self,memory):
        self.scaler=memory.scaler;self.classes=np.array(sorted(memory.memory))
        x=self.scaler.transform(np.concatenate([memory.memory[c] for c in self.classes]))
        y=np.concatenate([np.full(len(memory.memory[c]),c) for c in self.classes])
        self.models={}
        for method in METHODS:
            labels=y//6 if method=='ovr_task' else y
            models={}
            for label in np.unique(labels):
                target=(labels==label).astype(np.int64)
                if target.min()==target.max():models[int(label)]=None
                else:
                    clf=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,class_weight='balanced',random_state=0)
                    clf.fit(x,target);models[int(label)]=clf
            self.models[method]=models
    def score(self,z):
        x=self.scaler.transform(z);result={}
        for method,models in self.models.items():
            tasks=np.full((len(x),6),-np.inf)
            for label,clf in models.items():
                s=np.ones(len(x)) if clf is None else clf.decision_function(x)
                task=label if method=='ovr_task' else label//6
                tasks[:,task]=np.maximum(tasks[:,task],s)
            result[method]=tasks
        return result

if __name__=='__main__':
    main(scorer_type=OVRScorers,methods=METHODS,output_name='candle_v4_ovr_task_router_20261107')
