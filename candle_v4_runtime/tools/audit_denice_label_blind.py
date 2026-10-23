"""Replay frozen inference in an isolated process forbidden to read test targets.

This audits cached expert outputs, not GPU inference from raw inputs. The parent
reads prediction references only after the label-free worker has exited.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits
from tools.denice_competence_gate import feature_matrix, gate_decision, content_role
from tools.denice_class_meta import class_features, masked_class_decision
from tools.denice_peer_voting import vote


def read_allowed(archive, member, allowed):
    if not allowed(member):
        raise RuntimeError(f'Worker attempted forbidden archive access: {member}')
    return archive.read(member)


def worker(args):
    # No prediction CSV, manifest, target NPZ, protocol or provenance is read.
    # Payloads contain frozen training-derived parameters and class vocabularies.
    warnings.filterwarnings('ignore', category=sklearn.exceptions.InconsistentVersionWarning)
    fit_calls=[]
    def refuse_fit(*unused, **kwargs):
        fit_calls.append('fit'); raise RuntimeError('Fitting forbidden in inference audit')
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    for cls in (LogisticRegression, MLPClassifier, Pipeline, StandardScaler):
        cls.fit=refuse_fit
    args.out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.panel) as panel, zipfile.ZipFile(args.gate) as gatezip, zipfile.ZipFile(args.meta) as metazip:
        gate_bytes=read_allowed(gatezip,'frozen_gate.joblib',lambda n:n=='frozen_gate.joblib')
        meta_bytes=read_allowed(metazip,'frozen_class_meta.joblib',lambda n:n=='frozen_class_meta.joblib')
        bundle=joblib.load(io.BytesIO(gate_bytes)); meta=joblib.load(io.BytesIO(meta_bytes))
        edges=pd.read_csv(io.BytesIO(read_allowed(gatezip,'peer_edges.csv',lambda n:n=='peer_edges.csv')))
        labels=meta['labels']; gate=bundle['gates'][16,'MLP']; model=meta['models'][16,'ClassLR_C0.1']
        fields={'pred','task','router_confidence','class_confidence','class_margin',
                'class_entropy','mask_count','task_supported','class_supported'}
        rows=0
        with threadpool_limits(limits=1):
            for position,cid in enumerate(bundle['orders'],1):
                chosen=[cid]+bundle['orders'][cid][:16]; records={}
                for donor in chosen:
                    member=f'fresh_expert_features/test_receiver_{cid}_donor_{donor}.npz'
                    data_bytes=read_allowed(panel,member,lambda n:n.startswith('fresh_expert_features/') and n.endswith('.npz'))
                    with np.load(io.BytesIO(data_bytes),allow_pickle=False) as data:
                        if set(data.files)!=fields:raise RuntimeError('Unexpected expert fields; refuse possible target input')
                        records[donor]={name:data[name] for name in fields}
                alpha=edges[edges.receiver==cid].set_index('donor').alpha.to_dict()
                expert,pred,names=feature_matrix(cid,chosen,records,alpha,bundle['priors'],6,len(labels))
                if names!=bundle['feature_names']:raise RuntimeError('Gate schema changed')
                scores=gate.predict_proba(expert)[:,int(np.flatnonzero(gate.classes_==1)[0])].reshape(pred.shape)
                matrix,available,schema=class_features(expert,pred,scores,names,labels)
                if schema!=meta['feature_schema']:raise RuntimeError('Meta schema changed')
                majority=vote(pred.T,np.ones(len(chosen)),labels,pred[:,0])
                output=dict(self=pred[:,0],majority=majority,
                    GateV2=gate_decision(scores,pred,chosen,labels,1)[0],
                    FrozenClassMeta=masked_class_decision(model.predict_proba(matrix),model.classes_,labels,available,pred[:,0],majority))
                np.savez_compressed(args.out/f'client_{cid}.npz',**output)
                rows+=len(pred)
                if position%20==0:print(f'Label-free worker: {position} receivers',flush=True)
        result=dict(rows=rows,clients=len(bundle['orders']),fit_calls=len(fit_calls),
            test_targets_read=False,reference_predictions_read=False,
            archive_access_policy='expert NPZs + frozen model payloads + recorded graph edges only',
            sklearn_runtime=sklearn.__version__,artifact_sklearn=bundle['sklearn_version'],
            gate_sha256=hashlib.sha256(gate_bytes).hexdigest(),meta_sha256=hashlib.sha256(meta_bytes).hexdigest())
        (args.out/'label_free_worker.json').write_text(json.dumps(result,indent=2)+'\n')


def audit(args):
    if args.out.exists() and any(args.out.iterdir()):raise ValueError('Use a fresh output directory')
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',
        '--panel',str(args.panel),'--gate',str(args.gate),'--meta',str(args.meta),'--out',str(args.out)],check=True)
    # First reference/target access: worker has exited and saved its predictions.
    result=json.loads((args.out/'label_free_worker.json').read_text())
    mismatches={p:0 for p in ('self','majority','GateV2','FrozenClassMeta')}
    correct={p:0 for p in mismatches}; ids=[]
    with zipfile.ZipFile(args.panel) as z:
        protocol=json.loads(z.read('frozen_panel_protocol.json'))
        assert result['gate_sha256']==protocol['gate_sha256'] and result['meta_sha256']==protocol['meta_sha256']
        for cid in protocol['client_ids']:
            reference=pd.read_csv(z.open(f'predictions/client_{cid}.csv')); ids.append(cid)
            with np.load(args.out/f'client_{cid}.npz') as predictions:
                for policy in mismatches:
                    mismatches[policy]+=int((predictions[policy]!=reference[policy]).sum())
                    correct[policy]+=int((predictions[policy]==reference.y_true).sum())
        manifest=pd.read_csv(z.open('panel_manifest.csv'))
    with zipfile.ZipFile(args.gate) as z:
        provenance=pd.read_csv(z.open('gate_data_provenance.csv'))
        role_sets={role:set(group.input_sha256) for role,group in provenance.groupby('role')}
        if set(role_sets)!={'calibration','fit','validation'}:raise RuntimeError('Missing data roles')
        for role,values in role_sets.items():
            if any(content_role(digest)!=role for digest in values):raise RuntimeError('Content role mismatch')
            for other,other_values in role_sets.items():
                if role!=other and values&other_values:raise RuntimeError('Role contamination')
        split=json.loads(z.read('gate_split_audit.json'))
        role_counts={role:len(values) for role,values in role_sets.items()}
        assert role_counts==split['unique_content_by_role']
        target_overlap=len(set(manifest.input_sha256)&set(provenance.input_sha256))
        assert target_overlap==0
        gate_protocol=json.loads(z.read('gate_protocol.json'))
    result.update(prediction_mismatches=mismatches,
        accuracy={p:correct[p]/result['rows'] for p in mismatches},
        unique_content_by_role=role_counts,new_panel_gate_role_hash_overlap=target_overlap,
        original_backbone_holdout=split['original_backbone_holdout'],
        historical_revisit=split['historical_revisit'],
        centralized_gate_fit=gate_protocol['centralized_gate_fit'],
        privacy_deployment_claim=gate_protocol['privacy_deployment_claim'],
        scope='cached expert-to-final inference; raw-input GPU path inspected separately')
    if any(mismatches.values()):raise RuntimeError(f'Label-blind replay changed predictions: {mismatches}')
    (args.out/'label_blind_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel',required=True,type=Path)
    parser.add_argument('--gate',required=True,type=Path)
    parser.add_argument('--meta',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    (worker if args.worker else audit)(args)
