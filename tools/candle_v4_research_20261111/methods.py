"""Six frozen-feature component probes; not full paper reproductions.

No raw examples, TEST labels, live-model writes, or parameter search.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.svm import LinearSVC

METHODS=('fetril_translation','pass_isotropic','il2a_diagonal_loss','slca_gaussian_alignment','acil_recursive','dsal_residual')

def recursive_ridge(h,labels,tasks,regularization=1.):
    """Task-batched RLS, fixed features; equivalent to joint ridge numerically."""
    precision=np.eye(h.shape[1])/regularization
    weights=np.zeros((h.shape[1],labels.shape[1]))
    for task in np.unique(tasks):
        x=h[tasks==task];y=labels[tasks==task]
        gain=np.linalg.solve(np.eye(len(x))+x@precision@x.T,x@precision).T
        weights+=gain@(y-x@weights)
        precision-=gain@x@precision
        precision=(precision+precision.T)/2
    return weights

def fit_head(x,y,classes,steps=100,variance=None):
    with torch.random.fork_rng():
        torch.manual_seed(42);model=torch.nn.Linear(x.shape[1],classes)
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        xx=torch.as_tensor(x,dtype=torch.float32);yy=torch.as_tensor(y,dtype=torch.long)
        var=None if variance is None else torch.as_tensor(variance,dtype=torch.float32)
        for _ in range(steps):
            optimizer.zero_grad();logits=model(xx)
            if var is not None:
                # Diagonal semantic augmentation bound, gamma=1, no classAug.
                difference=model.weight[None,:,:]-model.weight[yy,None,:]
                logits=logits+.5*torch.sum(difference.square()*var[yy,None,:],dim=2)
            loss=torch.nn.functional.cross_entropy(logits,yy);loss.backward();optimizer.step()
        return model.weight.detach().numpy().T.copy(),model.bias.detach().numpy().copy()

class ComponentScorers:
    def __init__(self,memory):
        self.scaler=memory.scaler;self.classes=np.array(sorted(memory.memory))
        blocks=[self.scaler.transform(memory.memory[c]).astype(np.float64) for c in self.classes]
        self.means=np.stack([x.mean(0) for x in blocks]);width=blocks[0].shape[1]
        rng=np.random.default_rng(42);count=32;targets=np.repeat(np.arange(len(blocks)),count)
        latest=max(c//6 for c in self.classes);new=[i for i,c in enumerate(self.classes) if c//6==latest]
        translated=[]
        shared_variance=float(np.mean([np.var(x,axis=0).mean() for x in blocks]))
        isotropic=[];gaussian=[];diagonal=[]
        for i,x in enumerate(blocks):
            near=min(new,key=lambda j:np.sum((self.means[i]-self.means[j])**2))
            source=blocks[near] if self.classes[i]//6<latest else x
            residual=source[rng.integers(len(source),size=count)]-source.mean(0)
            translated.append(self.means[i]+residual)
            isotropic.append(self.means[i]+rng.normal(size=(count,width))*np.sqrt(shared_variance))
            cov=LedoitWolf().fit(x).covariance_ if len(x)>1 else np.eye(width)*shared_variance
            cov=cov+np.eye(width)*1e-6
            gaussian.append(self.means[i]+rng.normal(size=(count,width))@np.linalg.cholesky(cov).T)
            diagonal.append(np.diag(cov))
        svc=LinearSVC(C=1.,dual='auto',max_iter=5000,random_state=0)
        if len(blocks)>1:svc.fit(np.concatenate(translated),targets)
        self.svc=svc if len(blocks)>1 else None
        self.heads={
            'pass_isotropic':fit_head(np.concatenate(isotropic),targets,len(blocks)),
            'slca_gaussian_alignment':fit_head(np.concatenate(gaussian),targets,len(blocks)),
            'il2a_diagonal_loss':fit_head(self.means,np.arange(len(blocks)),len(blocks),variance=np.stack(diagonal)),
        }
        self.projection=rng.normal(size=(width,256))/np.sqrt(width)
        x=np.concatenate(blocks);y=np.concatenate([np.full(len(b),i) for i,b in enumerate(blocks)])
        labels=np.eye(len(blocks))[y];tasks=self.classes[y]//6
        h=np.maximum(x@self.projection,0)
        self.main=recursive_ridge(h,labels,tasks)
        residual_labels=labels-h@self.main
        # Dual-activation residual probe: omits DS-AL null-space module.
        self.compensation=recursive_ridge(np.tanh(x@self.projection),residual_labels,tasks)

    def score(self,z):
        x=self.scaler.transform(z)
        if self.svc is None:translation=np.ones((len(x),1))
        else:
            translation=self.svc.decision_function(x)
            if translation.ndim==1:translation=np.column_stack([-translation,translation])
        h=np.maximum(x@self.projection,0);main=h@self.main
        raw=dict(fetril_translation=translation,acil_recursive=main,
                 dsal_residual=main+np.tanh(x@self.projection)@self.compensation)
        raw.update({m:x@w+b for m,(w,b) in self.heads.items()})
        result={}
        for method,scores in raw.items():
            if not np.isfinite(scores).all():raise ValueError('Nonfinite component score')
            tasks=np.full((len(x),6),-np.inf)
            for i,c in enumerate(self.classes):tasks[:,c//6]=np.maximum(tasks[:,c//6],scores[:,i])
            result[method]=tasks
        return result
