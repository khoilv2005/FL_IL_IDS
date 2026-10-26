"""Kaggle evaluation only: restore V4 history, then frozen terminal TEST."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
from candle_v4_historical_repair import prepare_archives,repair_memories,sha256,SOURCE


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results16',required=True);p.add_argument('--results17',required=True)
    p.add_argument('--data-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--cache',default='/kaggle/temp/candle_v4_historical_repair_cache')
    a=p.parse_args()
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Full readout is Kaggle-only; local tests are separate')
    import torch
    import sklearn
    if not torch.cuda.is_available():raise RuntimeError('Enable Kaggle GPU')
    if sklearn.__version__!='1.6.1':raise RuntimeError('Use the original sklearn1.6.1 requirements before resume/readout')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from eval_checkpoint import _make_denice_client_model
    from tools.train_candle_v1 import resolve_data
    from fed_learning.training.candle_v4_original_train import OriginalTrainService
    from fed_learning.training.candle_v4_shadow import digest
    from fed_learning.training.candle_v4_terminal_test import evaluate_terminal
    out=Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use a fresh separate output directory')
    out.mkdir(parents=True,exist_ok=True)
    paths,old_path=prepare_archives(a.results16,a.results17,a.cache)
    original=json.loads(old_path.read_text(encoding='utf-8'))
    snapshots={t:torch.load(path,map_location='cpu',weights_only=False,mmap=True) for t,path in paths.items()}
    for t,state in snapshots.items():
        if state['meta']['completed_task']!=t or state.get('round_state') is not None:raise ValueError('Not a task boundary')
    final=snapshots[5];saved=final['candle_v4_shadow_state']
    if final['meta']['resume_from_task']!=6:raise ValueError('Expected completed T5, not a training resume')
    data=resolve_data(a.data_dir)
    if sha256(data/'metadata.json')!=saved['spec']['input_sha256']:raise ValueError('Wrong original dataset metadata')
    code=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    plan=dict(protocol='candle_v4_historical_memory_counterfactual_v1',source_training_commit=SOURCE,
        readout_source_commit=code,base_checkpoint_sha256=sha256(paths[5]),
        backbone_training=False,expert_training=False,raw_TRAIN_opened=False,TEST_used_for_fitting=False,
        fixed_threshold=.5,validation_or_acceptance_gate=False,sklearn=sklearn.__version__,
        evaluation='Unchanged terminal evaluator; original TEST disjoint shards; seed523687; pooled metrics',
        scientific_status='Development counterfactual after TEST-driven bug audit; not an independent benchmark or fresh V8 training')
    (out/'repair_run_lock.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    staged,provenance=repair_memories(final,snapshots)
    if set(staged)!=set(map(int,provenance)):raise RuntimeError('Incomplete staged repair')
    receipt=dict(clients=len(final['client_model_states']),repaired_clients=len(staged),repaired_ids=sorted(staged),
        unaffected_estimators_preserved=True,provenance=provenance,fit_data='Saved TRAIN feature memory only')
    (out/'repair_provenance.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    torch.save(dict(**plan,routers={cid:m.state_dict() for cid,m in staged.items()},provenance=provenance,
                    resumable_training_checkpoint=False),out/'historical_router_repair_bundle.pt')
    print(f'Historical memory repaired: {len(staged)}/{len(final["client_model_states"])} clients. No backbone/expert training.',flush=True)
    # Restore certificates under original v7 spec, then change output/data locations
    # only. The alternative readout is never written into the original run.
    service=OriginalTrainService(saved['spec'],saved)
    def expert_hash():
        return digest({arm:{key:e.branches['joint_class_balanced'][0].state_dict()
                            for key,e in engines.items()} for arm,engines in service.engines.items()})
    expert_before=expert_hash()
    service.continuous_routers.update(staged)
    service.spec['output_dir']=str(out);service.spec['data_dir']=str(data)
    models={};detectors={}
    for i,cid in enumerate(sorted(final['client_model_states']),1):
        models[cid],detectors[cid]=_make_denice_client_model(final,cid,'cuda')
        if i%20==0:print(f'Restored frozen predictors {i}/100',flush=True)
    del snapshots,staged
    result=evaluate_terminal(service,models,detectors,5,19)
    if expert_before!=expert_hash():raise RuntimeError('Readout changed expert weights')
    # The evaluator already asserts frozen native/service hashes before/after.
    result['scientific_status']=plan['scientific_status']
    result['repair_protocol']=plan['protocol'];result['repaired_clients']=receipt['repaired_clients']
    result['source_original_metrics']=original['metrics']
    result['native_control_exact_matrix_match']=result['metrics']['V1']['confusion_matrix']==original['metrics']['V1']['confusion_matrix']
    result['accuracy_delta_vs_original_v4']=result['metrics']['coverage']['accuracy']-original['metrics']['coverage']['accuracy']
    result['macro_f1_delta_vs_original_v4']=result['metrics']['coverage']['macro_f1']-original['metrics']['coverage']['macro_f1']
    target=out/'candle_v4_final_test.json';tmp=target.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2),encoding='utf-8');tmp.replace(target)
    completion=dict(completed=True,**plan,TEST_opened=True,repaired_clients=receipt['repaired_clients'],
        test_rows=result['test_rows'],evaluated_rows=result['evaluated_rows'],
        native_control_exact_matrix_match=result['native_control_exact_matrix_match'],
        metrics={name:{k:v[k] for k in ('accuracy','macro_f1')} for name,v in result['metrics'].items()},
        old_v4_metrics={k:original['metrics']['coverage'][k] for k in ('accuracy','macro_f1')})
    (out/'repair_completion.json').write_text(json.dumps(completion,indent=2),encoding='utf-8')
    print('COMPLETE:',json.dumps(completion['metrics']),flush=True)


if __name__=='__main__':main()
