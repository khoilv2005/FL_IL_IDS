"""Kaggle CANDLE V4 seed42; full TEST only after terminal consolidation."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--variant',choices=('coverage','native-only'),default='coverage')
    parser.add_argument('--resume',help='Original-TRAIN continuation only; role-based checkpoints are incompatible')
    args=parser.parse_args()
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Full training is Kaggle-only')
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    import torch
    from tools.train_candle_v1 import resolve_data
    from fed_learning.training.candle_v4_original_train import VERSION
    import hashlib
    if not torch.cuda.is_available():raise RuntimeError('Enable Kaggle GPU')
    data=resolve_data(args.data_dir);output=Path(args.output).resolve()
    metadata_sha256=hashlib.sha256((data/'metadata.json').read_bytes()).hexdigest()
    for owner in range(100):
        if not (data/f'client_{owner}_train.npz').is_file():raise FileNotFoundError(f'Original TRAIN missing for client {owner}')
    if not args.resume and output.exists() and any(output.iterdir()):raise ValueError('Fresh output required')
    config=json.loads((root/'configs/candle_v1.json').read_text())
    spec=dict(version=VERSION,seed=42,input_sha256=metadata_sha256,
              data_dir=str(data),output_dir=str(output),arms=['coverage'] if args.variant=='coverage' else [],
              data_protocol='100% original current-task TRAIN; no validation',fixed_threshold=.5,
              activation_policy='training_only_unverified',
              per_expert_bytes=128*1024**2,per_arm_bytes=64*1024**3)
    config.update(data_dir=str(data),output_dir=str(output),resume_output_dir=str(output),
                  denice_clean_roles_dir=None,candle_v4_full_service=spec,candle_v4_original_train=True,
                  denice_validation_fraction=0.0,
                  denice_post_task_eval_tasks=[],denice_eval_final_round=False,
                  denice_evaluation_data_role='test',
                  candle_v4_terminal_full_test=True,
                  resume_state_path=str(Path(args.resume).resolve()) if args.resume else None)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    output.mkdir(parents=True,exist_ok=True)
    lock=dict(version=VERSION,source_commit=source,seed=42,config=config,
              protocol='100% current-task TRAIN; no validation, roles or shadow native model',
              evaluation='Per-task validation NOT_MEASURED; full 34-class TEST only after T5/R19 consolidation',
              controls=['native V1 counterfactual from same backbone; historical 80/2/8/10 V1 is not a matched baseline'],
              risk='Fixed threshold .5; training-qualified provisional ACTIVE, quality UNVERIFIED; recall/FAR/break/net rescue NOT_MEASURED before TEST',
              runtime=dict(python=sys.version,torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0)))
    lock['resource_preflight']=dict(max_expert_count=100*34*len(spec['arms']),
        max_expert_parameter_bytes=100*34*len(spec['arms'])*2625*4,
        max_adam_moment_bytes=100*34*len(spec['arms'])*2625*8,
        serialized_wire_cap_per_arm=spec['per_arm_bytes'],
        total_expert_wire_cap_all_controls=len(spec['arms'])*spec['per_arm_bytes'],
        native_output_guard_gib=config['denice_checkpoint_storage_budget_gib'],
        policy='Caps are enforced, not a prediction of measured traffic; native traffic additional')
    lock_path=output/'candle_v4_run_lock.json'
    if args.resume:
        if not lock_path.exists():raise ValueError('Restore the same v3 run lock and activation journals with continuation')
        previous=json.loads(lock_path.read_text())
        if previous['source_commit']!=source or previous['config']['candle_v4_full_service']!=spec:
            raise ValueError('Resume source/config changed')
    else:
        lock_path.write_text(json.dumps(lock,indent=2),encoding='utf-8')
        (output/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    os.environ.update(DENICE_RUN_PROFILE='legacy',DENICE_VARIANT='legacy',DENICE_TRAIN_PHASE='5',
                      DENICE_CODE_DIR=str(root),DENICE_SEED='42',DENICE_OUTPUT_DIR=str(output),
                      DENICE_CONFIG_OVERRIDES=json.dumps(config))
    print('CANDLE V1/V4: 100% CURRENT-TASK TRAIN; NO VALIDATION, ROLES OR SHADOW NATIVE MODEL.',flush=True)
    print(f'Seed42 T0..T5 x20 rounds; variant={args.variant}; active expert arms={spec["arms"]}.',flush=True)
    print('One GPU, native batch2048 AMP; experts CPU; per-round continuation; task ZIP. Final full TEST after T5/R19 consolidation only.',flush=True)
    started=time.perf_counter()
    runpy.run_path(str(root/'train_incremental_kaggle.py'),run_name='__main__')
    report=output/'candle_v4_development_report.json'
    if not report.exists():raise RuntimeError('Missing V4 final development report')
    result=json.loads(report.read_text())
    if len(result['evaluations'])!=6:raise RuntimeError('Incomplete task evaluations')
    test_report=output/'candle_v4_final_test.json'
    if not test_report.exists():raise RuntimeError('Missing terminal full TEST report')
    test=json.loads(test_report.read_text())
    if not test.get('completed') or test['evaluated_rows']!=test['test_rows']:
        raise RuntimeError('Incomplete terminal full TEST')
    (output/'candle_v4_completion.json').write_text(json.dumps(dict(completed=True,source_commit=source,
        seed=42,development_only=True,TEST_opened=True,report=str(report),test_report=str(test_report),
        invocation_wall_seconds=time.perf_counter()-started,validation_metrics_status='NOT_MEASURED',
        activation_policy='training_only_unverified',experiment_variant=args.variant,final_validation=result['evaluations'][-1],
        final_test=test['metrics']),indent=2),encoding='utf-8')
    if args.variant=='native-only':
        (output/'candle_v1_completion.json').write_bytes((output/'candle_v4_completion.json').read_bytes())
    print('Completed CANDLE V4 development:',output/'candle_v4_completion.json',flush=True)


if __name__=='__main__':main()
