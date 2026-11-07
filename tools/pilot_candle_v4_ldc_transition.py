"""Bounded LDC-inspired compensation at T4->T5; current-task TRAIN pairs only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import copy,io,json,sys,warnings,zipfile,time
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
from candle_v4_historical_repair import member,copy_member,sha256
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.candle_v4_original_train import OriginalTrainService
from fed_learning.servers.continuous_class_router import ContinuousClassMemory
from fed_learning.training.candle_v4_shadow import digest
from fed_learning.training.candle_v4_terminal_test import statistics
from fed_learning.training.candle_v4_continuous_runtime import predictions
from fed_learning.training.candle_v4_bank import base_predictions
from candle_v4_top2_task_select import FixedChoiceView
from pilot_candle_v4_receiver_routing_features import encode

class Compensation:
    def __init__(self,x,y):
        self.sx=StandardScaler().fit(x);self.sy=StandardScaler().fit(y)
        a=self.sx.transform(x);b=self.sy.transform(y);design=np.column_stack([a,np.ones(len(a))])
        penalty=np.eye(design.shape[1]);penalty[-1,-1]=0
        self.ridge=np.linalg.solve(design.T@design+penalty,design.T@b)
        with torch.random.fork_rng():
            torch.manual_seed(42)
            self.net=torch.nn.Sequential(torch.nn.Linear(x.shape[1],128),torch.nn.ReLU(),torch.nn.Linear(128,y.shape[1]))
            opt=torch.optim.Adam(self.net.parameters(),lr=.001)
            xx=torch.as_tensor(a,dtype=torch.float32);yy=torch.as_tensor(b,dtype=torch.float32)
            for _ in range(100):
                opt.zero_grad();loss=torch.mean((self.net(xx)-yy)**2);loss.backward();opt.step()
        self.net.eval()
    def transform(self,x,method):
        a=self.sx.transform(x)
        if method=='ridge':b=np.column_stack([a,np.ones(len(a))])@self.ridge
        else:
            with torch.no_grad():b=self.net(torch.as_tensor(a,dtype=torch.float32)).numpy()
        return self.sy.inverse_transform(b).astype(np.float32)

def main():
    warnings.filterwarnings('ignore',message='Trying to unpickle estimator');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'audit_denice/candle_v4_ldc_transition_20261107';out.mkdir(parents=True,exist_ok=True)
    oldpath=out/'continuation_state_task_4.pt'
    if not oldpath.exists():
        nested=out/'task4_input.zip'
        with zipfile.ZipFile(r'C:/Users/khoak/Downloads/results (19).zip') as z:
            copy_member(z,member(z,'checkpoint_task_4_all_rounds.zip'),nested)
        with zipfile.ZipFile(nested) as z:
            manifest=json.loads(z.read('checkpoint_archive_manifest.json'));name='continuation_state_task_4.pt'
            assert manifest['task_id']==4 and manifest['completed_rounds']==list(range(20))
            copy_member(z,name,oldpath,manifest['checksums'][name])
        # Only the temporary archive created by this script, within its audit directory.
        assert nested.resolve().parent==out.resolve();nested.unlink()
    print('Verified fresh-run Task4 continuation',flush=True)
    old=torch.load(oldpath,map_location='cpu',weights_only=False,mmap=True)
    final=torch.load(ROOT/'audit_denice/candle_v4_results20_20261027/continuation_state_latest.pt',map_location='cpu',weights_only=False,mmap=True)
    assert old['meta']['completed_task']==4 and final['meta']['completed_task']==5
    service=OriginalTrainService(final['candle_v4_shadow_state']['spec'],final['candle_v4_shadow_state']);frozen=digest(service.state_dict())
    oldmem=old['candle_v4_shadow_state']['continuous_routers']
    # A Task5 memory can arrive by bootstrap; require actual local current TRAIN.
    local_current=set()
    for row in final['debug_history']:
        if row.get('task')==5:
            local_current.update(int(cid) for cid,v in row.get('client_data',{}).items() if v.get('num_samples',0)>1280)
    selected=[]
    for cid,mem in sorted(service.continuous_routers.items()):
        if cid not in local_current:continue
        if cid not in oldmem:continue
        prev=ContinuousClassMemory.restore(oldmem[cid])
        if 5 not in mem.class_to_task.values() or len(set(prev.class_to_task.values()))<2:continue
        if not np.array_equal(prev.indices,mem.indices):continue
        if any(c not in mem.memory or not np.array_equal(x,mem.memory[c]) for c,x in prev.memory.items()):continue
        selected.append(cid)
        if len(selected)==3:break
    assert len(selected)==3
    (out/'fixture.json').write_text(json.dumps(dict(clients=selected,selection='first3 metadata-compatible receivers with historical memory and Task5',train_seed=42,fit_per_current_class=256,assess_per_current_class=64,test_opened=False),indent=2),encoding='utf-8')
    print('Locked clients',selected,flush=True)
    trained={};models={};result=dict(completed=False,clients={},fit_source='Only current Task5 original TRAIN; historical saved features for compensation',
        scope='LDC-inspired ridge and MLP; not exact paper reproduction; repeated TEST development',source_task4_sha256=sha256(oldpath))
    with zipfile.ZipFile(r'C:/Users/khoak/Downloads/archive (1).zip') as archive:
        for cid in selected:
            before,bd=_make_denice_client_model(old,cid,'cuda');before.eval()
            after,ad=_make_denice_client_model(final,cid,'cuda');after.eval();models[cid]=(after,ad,digest(after.state_dict()))
            with np.load(io.BytesIO(archive.read(f'100-clients/client_{cid}_train.npz'))) as data:
                y=data['y_train'];rng=np.random.default_rng(42);fit=[];assess=[]
                for c in range(30,34):
                    ids=rng.permutation(np.flatnonzero(y==c));fit.extend(ids[:256]);assess.extend(ids[256:320])
                assert len(fit)>0 and len(assess)>0
                inputs=torch.from_numpy(data['X_train'][fit+assess].astype(np.float32))
            prev=ContinuousClassMemory.restore(oldmem[cid]);mem=service.continuous_routers[cid]
            a=encode(before,inputs,prev.indices);b=encode(after,inputs,mem.indices);n=len(fit)
            mapper=Compensation(a[:n],b[:n]);denom=np.maximum(b[n:].std(0),1e-6)
            error=lambda z:float(np.mean(((z-b[n:])/denom)**2))
            variants={};drift=dict(train_pairs=n,assess_pairs=len(assess),raw_mse=float(np.mean((a[n:]-b[n:])**2)),standardized_mse_identity=error(a[n:]),
                current_train_route_disagreement=float(np.mean(prev.predict(a[n:])[0]!=mem.predict(b[n:])[0])))
            control=ContinuousClassMemory.restore(mem.state_dict());control.fit(5);variants['refit_control']=control
            for method in ('ridge','mlp'):
                modified=ContinuousClassMemory.restore(mem.state_dict())
                for c,x in prev.memory.items():modified.memory[c]=mapper.transform(x,method)
                modified.fit(5);variants[method]=modified
                drift['standardized_mse_'+method]=error(mapper.transform(a[n:],method))
            result['clients'][cid]=dict(drift=drift)
            trained[cid]=variants
            del before,bd
            print('Fit/assess TRAIN complete client',cid,'MSE',drift['raw_mse'],flush=True)
    # No TEST inputs or labels have been opened before all maps/routers are fitted.
    with np.load(r'C:/Users/khoak/Downloads/candle_v1_multiclass_audit/global_test_data.npz') as data:
        truth=data['y_test'];rng=np.random.default_rng(20261107)
        ix=np.concatenate([rng.choice(np.flatnonzero(truth==c),128,replace=False) for c in range(34)])
        xx=torch.from_numpy(data['X_test'][ix].astype(np.float32));yy=truth[ix].astype(np.int64)
    total={m:np.zeros((34,34),dtype=np.int64) for m in ('baseline','refit_control','ridge','mlp')}
    for cid in selected:
        model,det,hash_before=models[cid];mem=service.continuous_routers[cid];cms={m:np.zeros((34,34),dtype=np.int64) for m in total}
        counts={m:dict(route_correct=0,rescue=0,break_count=0) for m in total}
        for lo in range(0,len(yy),512):
            inputs=xx[lo:lo+512].reshape(-1,39);y=yy[lo:lo+512]
            native,first=predictions(service,cid,model,det,inputs,list(range(34)))
            baseline=service.infer(cid,'coverage',native,inputs,predicted_tasks=first)[0]
            z=encode(model,inputs,mem.indices)
            routes={m:v.predict(z)[0] for m,v in trained[cid].items()};routes['baseline']=first
            for method,chosen in routes.items():
                if method=='baseline':pred=baseline
                else:
                    native,check=base_predictions(model,FixedChoiceView(det,mem,chosen),inputs,list(range(34)),True)
                    assert np.array_equal(chosen,check)
                    pred=service.infer(cid,'coverage',native,inputs,predicted_tasks=chosen)[0]
                np.add.at(cms[method],(y,pred),1);v=counts[method]
                v['route_correct']+=int(np.sum(chosen==y//6));v['rescue']+=int(np.sum((pred==y)&(baseline!=y)));v['break_count']+=int(np.sum((pred!=y)&(baseline==y)))
        assert hash_before==digest(model.state_dict())
        result['clients'][cid]['metrics']={m:dict(**statistics(cm),**counts[m]) for m,cm in cms.items()}
        for m in total:total[m]+=cms[m]
    assert frozen==digest(service.state_dict())
    result.update(completed=True,rows_per_client=len(yy),metrics={m:statistics(cm) for m,cm in total.items()},model_and_service_unchanged=True,
        limitations='Historical memory was already stale before T4 potentially; only T4-to-T5 map, current-domain validation cannot prove old-class transport; no live training integration')
    (out/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({m:{k:v[k] for k in ('accuracy','macro_f1')} for m,v in result['metrics'].items()},indent=2),flush=True)

if __name__=='__main__':main()
