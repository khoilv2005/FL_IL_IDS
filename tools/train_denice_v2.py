"""Fresh DeNICE V2 Conv2 main, or resume at a verified round boundary."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
import os
import shutil

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fed_learning.data.denice_clean_roles import file_sha256
from fed_learning.strategies.decentralized.denice_v2.data import TaskData
from fed_learning.strategies.decentralized.denice_v2.evaluation import atomic_json
from fed_learning.strategies.decentralized.denice_v2.runner import Runner, RunConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--worker-devices',default='cuda:0,cuda:1')
    parser.add_argument('--batch-size',type=int,default=4086)
    parser.add_argument('--defer-evaluation',action='store_true',help='Never load TEST; evaluate separately with eval_selective')
    parser.add_argument('--server-only-storage',action='store_true',help='Keep only cluster states; omit post-round client broadcast')
    parser.add_argument('--prediction-mode',choices=('cnn_coordinate_router_v1','routerless_v1'),default='cnn_coordinate_router_v1')
    parser.add_argument('--server-exchange',action='store_true')
    parser.add_argument('--resume-task',type=int)
    parser.add_argument('--resume-round',type=int)
    parser.add_argument('--resume-root')
    parser.add_argument('--output-budget-gib',type=float,default=18)
    parser.add_argument('--eval-batch-size',type=int,default=8192)
    args=parser.parse_args()
    worker_devices=tuple(d.strip() for d in args.worker_devices.split(',') if d.strip())
    if worker_devices and (len(worker_devices)!=2 or len(set(worker_devices))!=2):
        raise ValueError('Main training needs two distinct GPU worker devices')
    for name in worker_devices:
        device=torch.device(name)
        if device.type!='cuda' or device.index is None or device.index>=torch.cuda.device_count():
            raise RuntimeError('Enable Kaggle T4 x2; both requested GPU workers must be available')
    output=Path(args.output).resolve()
    if args.resume_task is None and output.exists() and any(output.iterdir()):
        raise ValueError('Fresh training requires an empty output; use --resume-task for continuation')
    output.mkdir(parents=True,exist_ok=True)
    bundle=os.environ.get('DENICE_SOURCE_ZIP')
    if bundle and Path(bundle).is_file():
        target=output/'denice_cluster_server_source.zip'
        if Path(bundle).resolve()!=target.resolve():shutil.copyfile(bundle,target)
    root=Path(__file__).resolve().parents[1]
    sources=[Path(__file__),*sorted((root/'fed_learning').rglob('*.py'))]
    source_hashes={p.relative_to(root).as_posix():file_sha256(p) for p in sources}
    provenance=Path(args.resume_root)/'execution_provenance.json' if args.resume_root else output/'execution_provenance.json'
    if args.resume_root and (args.resume_task is None or not provenance.exists()):
        raise ValueError('Mounted resume requires --resume-task and original execution_provenance.json')
    if provenance.exists():
        old=json.loads(provenance.read_text(encoding='utf-8'))
        if old['source_sha256']!=source_hashes:
            raise ValueError('Resume requires the exact original source; source hash changed')
        if old['torch']!=torch.__version__:
            raise ValueError('Resume requires the original torch version/runtime')
    else:
        try:
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True,stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            commit=None  # Packaged source is identified by the hashes above.
        atomic_json(provenance,dict(commit=commit,source_sha256=source_hashes,
            torch=torch.__version__,python=sys.version,original_train_all_rows=True,
            training_data_protocol='original_train_all_rows_current_task_v1',heldout_train_roles=False,
            evaluation_cadence='external_eval_selective' if args.defer_evaluation else 'last_round_final_task',
            persistent_model_policy='one_per_cluster',post_round_broadcast=not args.server_only_storage,
            router='routerless_v1' if args.prediction_mode=='routerless_v1' else 'binary_cosine',feature_policy=args.prediction_mode,
            architecture='legacy_plain_no_CGoFed_no_CME_no_APPLIANCE',local_budget='one_phase_one_epoch_every_current_TRAIN_row_per_round',
            worker_devices=worker_devices,batch_size=args.batch_size,parallel_execution_policy='spawn_process_per_gpu_lpt_v1' if worker_devices else 'sequential_ephemeral_workers',
            numerical_policy='FP32_no_AMP_round_phase_optimizer_reset'))
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    data=TaskData(args.data_dir,output/'current_data_cache')
    cfg=RunConfig(device=args.device,worker_devices=worker_devices,batch_size=args.batch_size,
        output_budget_gib=args.output_budget_gib,eval_batch_size=args.eval_batch_size,
        defer_evaluation=args.defer_evaluation,server_only_storage=args.server_only_storage,
        policy=args.prediction_mode,server_exchange=args.server_exchange)
    if cfg.server_exchange:
        from fed_learning.strategies.decentralized.denice_v2.hierarchical import HierarchicalRunner
        runner=HierarchicalRunner(data,output,cfg)
    else:
        runner=Runner(data,output,cfg)
    if args.resume_task is not None:
        runner.resume(args.resume_task,args.resume_round,args.resume_root)
        if args.resume_root:
            atomic_json(output/'execution_provenance.json',old)
            atomic_json(output/'resume_origin.json',dict(source=str(Path(args.resume_root).resolve()),
                task=args.resume_task,round=args.resume_round,source_archives={p.name:file_sha256(p)
                    for p in Path(args.resume_root).glob('checkpoint_task_*_*.zip')}))
    atomic_json(output/'protocol_lock.json',dict(lock=runner.lock,config=cfg.__dict__,data_binding=data.binding,
        original_train_all_rows=True,heldout_train_roles=False,test_selection_for_training=False,quality_is_report_only=True))
    print(f'DeNICE V2 MAIN: seed42, 6 x 20 rounds, auto K, xi=.8, prediction={cfg.policy}.',flush=True)
    print(f'Parallel GPU workers={worker_devices}; batch/client={cfg.batch_size}; output={output}',flush=True)
    try:
        runner.run()
    except Exception as exc:
        atomic_json(output/'failure.json',dict(completed=False,error=repr(exc),traceback=traceback.format_exc(),
            next_task=runner.next_task,next_round=runner.next_round,task_end_pending=runner.task_end_pending,
            recovery='Resume the most recent verified checkpoint with the same source/data/config'))
        raise
    print(f'COMPLETED: {output}/completion.json',flush=True)


if __name__=='__main__':main()
