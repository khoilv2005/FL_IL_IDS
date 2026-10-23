"""Train class-level stacking on V2 caches; lock before reading test records."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from tools.denice_competence_gate import feature_matrix,gate_decision
from tools.denice_peer_voting import vote
from tools.denice_class_meta import class_features,masked_class_decision,balanced_resample_indices


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf-8')


class GateArchive:
    def __init__(self,path):
        self.zip=zipfile.ZipFile(path)
        self.completion=json.loads(self.zip.read('gate_completion.json'))
        self.protocol=json.loads(self.zip.read('protocol.json'))
        self.lock=json.loads(self.zip.read('gate_lock.json'))
        if not self.completion.get('completed') or self.completion.get('variant')!='v2_peer_supported':
            raise ValueError('Completed Gate V2 archive required')
        payload=self.zip.read('frozen_gate.joblib')
        if hashlib.sha256(payload).hexdigest()!=self.lock['gate_sha256']:
            raise ValueError('V2 gate archive checksum mismatch')
        if self.lock.get('sklearn_version')!=sklearn.__version__:
            raise RuntimeError(f'Install scikit-learn=={self.lock.get("sklearn_version")} before importing this evaluator')
        self.bundle=joblib.load(io.BytesIO(payload));self.ids=self.protocol['client_ids']
        self.classes={int(t):list(map(int,v)) for t,v in self.protocol['task_classes'].items()}
        self.labels=sorted({c for v in self.classes.values() for c in v})
        if not str(self.protocol['training_commit']).startswith('03b9b53'):
            raise ValueError('Expected frozen checkpoint 03b9b53')
        self.edges=pd.read_csv(self.zip.open('peer_edges.csv'))
        self.alphas={cid:self.edges[self.edges.receiver==cid].set_index('donor').alpha.to_dict() for cid in self.ids}

    def role(self,role,k):
        """Only requested role is read. Test records are inaccessible until caller requests them."""
        result={}
        for cid in self.ids:
            chosen=[cid]+self.bundle['orders'][cid][:k];records={}
            folder='test_expert_features' if role=='test' else 'gate_train_expert_features'
            for donor in chosen:
                with np.load(io.BytesIO(self.zip.read(f'{folder}/{role}_receiver_{cid}_donor_{donor}.npz'))) as data:
                    records[donor]={name:data[name] for name in data.files}
            expert,pred,names=feature_matrix(cid,chosen,records,self.alphas[cid],self.bundle['priors'],
                len(self.classes),max(self.labels)+1)
            selected=self.bundle['selected'][k];family,top=selected.split('_top')
            gate=self.bundle['gates'][k,family]
            scores=gate.predict_proba(expert)[:,int(np.flatnonzero(gate.classes_==1)[0])].reshape(pred.shape)
            matrix,available,schema=class_features(expert,pred,scores,names,self.labels)
            majority=vote(pred.T,np.ones(len(chosen)),self.labels,pred[:,0])
            baseline=gate_decision(scores,pred,chosen,self.labels,int(top))[0]
            all_vote=vote(pred.T,scores.T,self.labels,pred[:,0])
            if role=='test':
                reference=pd.read_csv(self.zip.open(f'predictions/client_{cid}.csv'))
                if (not np.array_equal(baseline,reference[f'k{k}_ValidationSelected'])
                    or not np.array_equal(majority,reference[f'k{k}_majority'])):
                    raise RuntimeError('V2/majority baseline cache reproduction failed')
                y=reference.y_true.to_numpy();identities=reference.global_test_row.to_numpy()
                covered=None
            else:
                with np.load(io.BytesIO(self.zip.read(f'gate_role_targets/{role}_receiver_{cid}.npz'))) as targets:
                    y=targets['y_true'];identities=targets['role_row_index'];covered=targets.get('receiver_local_covered')
            if len(y)!=len(matrix) or len(identities)!=len(matrix):raise ValueError('Cache/target row alignment mismatch')
            result[cid]=dict(X=matrix,available=available,y=y,sample_ids=identities,self=pred[:,0],
                majority=majority,GateV2=baseline,GateAllVote=all_vote,
                oracle=(pred==y[:,None]).any(1),correct_votes=(pred==y[:,None]).sum(1),
                receiver_local_covered=covered,schema=schema)
        return result


def class_prediction(model,record,labels):
    return masked_class_decision(model.predict_proba(record['X']),model.classes_,labels,
        record['available'],record['self'],record['majority'])


def validation_audit(records,classes,out,k):
    label_task={label:t for t,labels in classes.items() for label in labels}
    frame=pd.concat([pd.DataFrame(dict(client_id=cid,y_true=r['y'],
        true_task=[label_task[int(y)] for y in r['y']],majority=r['majority']==r['y'],
        GateV2=r['GateV2']==r['y'],actual_routed_oracle=r['oracle'],correct_votes=r['correct_votes']))
        for cid,r in records.items()],ignore_index=True)
    for grouping,name in [('y_true','class'),('true_task','task')]:
        report=frame.groupby(grouping).agg(rows=('GateV2','size'),majority=('majority','mean'),
            GateV2=('GateV2','mean'),actual_routed_oracle=('actual_routed_oracle','mean'),
            mean_correct_votes=('correct_votes','mean'))
        report.to_csv(out/f'validation_{name}_audit_k{k}.csv')


def candidate_specs():
    return {
        'ClassLR_C0.1':('LR',.1,False),'ClassLR_C1':('LR',1.,False),
        'ClassLR_C1_balanced':('LR',1.,True),
        'ClassMLP':('MLP',.001,False),'ClassMLP_balanced':('MLP',.001,True)}


def run_class_meta(archive,out,budgets=(4,8,16)):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh output directory to avoid mixing runs')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'class_meta_completion.json',dict(completed=False))
    source=GateArchive(archive);ids=source.ids;labels=source.labels
    if tuple(budgets)!=tuple(source.bundle['budgets']):raise ValueError('Keep the original V2 budgets')
    specs=candidate_specs()
    write_json(out/'class_meta_protocol.json',dict(candidate_specs=specs,budgets=budgets,
        source_gate_sha256=source.lock['gate_sha256'],sklearn_version=sklearn.__version__,
        training_commit=source.protocol['training_commit'],
        action_set='Only normal argmax classes already predicted by queried V2 experts',
        train_unreachable_policy='Exclude unreachable fitting targets; retain every validation/test sample',
        hyperparameters_selected_on_validation=True,panel_role='diagnostic/development',
        backbone_inference=False,raw_data_loaded=False))
    models={};validation=[];eligibility=[];schema=None
    valdir=out/'validation_predictions';valdir.mkdir(exist_ok=True)
    for k in budgets:
        fitting=source.role('fit',k);held=source.role('validation',k)
        validation_audit(held,source.classes,out,k)
        x=np.concatenate([fitting[cid]['X'] for cid in ids]);y=np.concatenate([fitting[cid]['y'] for cid in ids])
        reachable=np.concatenate([fitting[cid]['oracle'] for cid in ids]);current_schema=fitting[ids[0]]['schema']
        if schema is not None and schema!=current_schema:raise ValueError('Feature schema changed across budgets')
        schema=current_schema
        if any(r['schema']!=schema for r in list(fitting.values())+list(held.values())):
            raise ValueError('Feature schema changed across receivers/roles')
        for label in labels:
            rows=y==label;eligibility.append(dict(k=k,label=label,total_fit=int(rows.sum()),
                reachable_fit=int((rows&reachable).sum())))
        x=x[reachable];y=y[reachable]
        if len(x)<32 or len(np.unique(y))<2:raise ValueError('Insufficient reachable fitting labels')
        outputs={cid:{name:held[cid][name] for name in ('self','majority','GateV2','GateAllVote')} for cid in ids}
        for name,(family,strength,balanced) in specs.items():
            if family=='LR':
                estimator=LogisticRegression(C=strength,class_weight='balanced' if balanced else None,
                    max_iter=1000,solver='lbfgs',random_state=20261005)
                fit_x,fit_y=x,y
            else:
                estimator=MLPClassifier(hidden_layer_sizes=(64,32),alpha=strength,max_iter=50,
                    batch_size=1024,early_stopping=False,random_state=20261005)
                if balanced:
                    indices=balanced_resample_indices(y);fit_x,fit_y=x[indices],y[indices]
                else:fit_x,fit_y=x,y
            model=make_pipeline(StandardScaler(),estimator);model.fit(fit_x,fit_y);models[k,name]=model
            for cid in ids:outputs[cid][name]=class_prediction(model,held[cid],labels)
            print(f'Class meta fit k={k} model={name}, reachable fit rows={len(y)}',flush=True)
        truth=np.concatenate([held[cid]['y'] for cid in ids])
        for name in outputs[ids[0]]:
            pred=np.concatenate([outputs[cid][name] for cid in ids])
            validation.append(dict(k=k,policy=name,accuracy=float((pred==truth).mean()),
                macro_f1=float(f1_score(truth,pred,labels=labels,average='macro',zero_division=0))))
        for cid in ids:
            frame=dict(client_id=np.full(len(held[cid]['y']),cid),role_row_index=held[cid]['sample_ids'],
                y_true=held[cid]['y'],OracleActualRouted=held[cid]['oracle'],**outputs[cid])
            pd.DataFrame(frame).to_csv(valdir/f'k{k}_client_{cid}.csv',index=False)
        del fitting,held,x,y,outputs,fit_x,fit_y
    validation=pd.DataFrame(validation);validation.to_csv(out/'validation_metrics.csv',index=False)
    pd.DataFrame(eligibility).to_csv(out/'fit_candidate_coverage.csv',index=False)
    choices=validation[validation.policy.isin(list(specs)+['GateAllVote'])]
    selected={k:choices[choices.k==k].sort_values(['accuracy','macro_f1','policy'],ascending=[False,False,True]).iloc[0].policy for k in budgets}
    joblib.dump(dict(models=models,selected=selected,feature_schema=schema,labels=labels,specs=specs,
        source_gate_sha256=source.lock['gate_sha256']),out/'frozen_class_meta.joblib')
    digest=hashlib.sha256((out/'frozen_class_meta.joblib').read_bytes()).hexdigest()
    write_json(out/'class_meta_lock.json',dict(sha256=digest,selected_by_validation=selected,
        locked_before_reading_test_role=True,test_labels_used_for_fit=False,
        test_labels_used_for_selection=False,feature_schema=schema))
    restored=joblib.load(out/'frozen_class_meta.joblib');models=restored['models'];selected=restored['selected']
    restore_mismatches=0
    for k in budgets:
        held=source.role('validation',k)
        for cid in ids:
            reference=pd.read_csv(valdir/f'k{k}_client_{cid}.csv')
            for name in specs:
                restore_mismatches+=int((class_prediction(models[k,name],held[cid],labels)!=reference[name].to_numpy()).sum())
        del held
    if restore_mismatches:raise RuntimeError('Class meta save/restore predictions changed')
    lock=json.loads((out/'class_meta_lock.json').read_text(encoding='utf-8'))
    lock['validation_restore_mismatches']=restore_mismatches
    write_json(out/'class_meta_lock.json',lock)
    metrics=[];testdir=out/'predictions';testdir.mkdir(exist_ok=True);saved={}
    # First test-role access occurs only after the frozen meta artifact/lock exist.
    for k in budgets:
        records=source.role('test',k);saved[k]={}
        for cid in ids:
            r=records[cid];outputs={name:r[name] for name in ('self','majority','GateV2','GateAllVote')}
            for name in specs:outputs[name]=class_prediction(models[k,name],r,labels)
            outputs['ValidationSelected']=outputs[selected[k]]
            frame=dict(client_id=np.full(len(r['y']),cid),global_test_row=r['sample_ids'],y_true=r['y'],
                OracleActualRouted=r['oracle'],**outputs)
            pd.DataFrame(frame).to_csv(testdir/f'k{k}_client_{cid}.csv',index=False);saved[k][cid]=frame
            for policy,pred in outputs.items():
                correct=pred==r['y']
                if np.any(correct&~r['oracle']):raise RuntimeError('Meta prediction exceeds fixed routed action set')
                metrics.append(dict(k=k,client_id=cid,policy=policy,samples=len(r['y']),correct=int(correct.sum()),
                    accuracy=float(correct.mean()),oracle_correct=int(r['oracle'].sum())))
        del records
    metrics=pd.DataFrame(metrics);metrics.to_csv(out/'per_client_metrics.csv',index=False)
    summary=[];rng=np.random.default_rng(20261005)
    for (k,policy),group in metrics.groupby(['k','policy']):
        group=group.set_index('client_id').loc[ids]
        baseline=metrics[(metrics.k==k)&(metrics.policy=='GateV2')].set_index('client_id').loc[ids]
        delta=group.accuracy.to_numpy()-baseline.accuracy.to_numpy()
        draw=rng.integers(0,len(ids),(2000,len(ids)));ci=np.quantile(delta[draw].mean(1),[.025,.975])*100
        truth=np.concatenate([saved[k][cid]['y_true'] for cid in ids]);pred=np.concatenate([saved[k][cid][policy] for cid in ids])
        summary.append(dict(k=int(k),policy=policy,pooled_accuracy=float(group.correct.sum()/group.samples.sum()),
            client_mean_accuracy=float(group.accuracy.mean()),pooled_macro_f1=float(f1_score(truth,pred,labels=labels,average='macro',zero_division=0)),
            delta_vs_gate_v2_pp=float(delta.mean()*100),paired_ci_low_pp=float(ci[0]),paired_ci_high_pp=float(ci[1]),
            actual_routed_oracle=float(group.oracle_correct.sum()/group.samples.sum()),queried_models=int(k+1)))
    summary=pd.DataFrame(summary);summary.to_csv(out/'summary.csv',index=False)
    if hashlib.sha256((out/'frozen_class_meta.joblib').read_bytes()).hexdigest()!=digest:raise RuntimeError('Meta artifact changed')
    plot_meta(summary,out)
    write_json(out/'class_meta_completion.json',dict(completed=True,selected_by_validation=selected,
        action_set_matches_actual_routed_oracle=True,final_untouched_test=False,
        shared_retrospective_meta=True,streaming_replay_free_claim=False))
    source.zip.close()
    return summary


def plot_meta(summary,out):
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for name in ('majority','GateV2','GateAllVote','ValidationSelected'):
        values=summary[summary.policy==name].sort_values('k')
        ax.plot(values.k,100*values.pooled_accuracy,marker='o',label=name)
    values=summary[summary.policy=='GateV2'].sort_values('k')
    ax.plot(values.k,100*values.actual_routed_oracle,linestyle='--',label='Actual-routed oracle')
    ax.axhline(50,color='grey',linestyle=':',label='50% reference')
    ax.set(xlabel='Candidate peers k (self additional)',ylabel='Pooled accuracy (%)',ylim=(0,100),
        title='Class-level meta-ensemble on frozen V2 expert caches')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(out/'class_meta_vs_budget.png',dpi=180);fig.savefig(out/'class_meta_vs_budget.pdf');plt.close(fig)
