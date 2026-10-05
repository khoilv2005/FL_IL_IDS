"""Clean-role, six-task backbone run. Upload this file or its generated notebook."""
import json
import os
from pathlib import Path
import runpy
import re
import subprocess
import sys
import tempfile

# Main protocol: keep xi=0.5. Run seed 42 first, before the replication campaign.
TRAIN_SEED=int(os.environ.get('DENICE_SEED','42'))
ROLE_SPLIT_SEED=20261006  # Same holdout split across independent training seeds.
DATA_DIR=os.environ.get('DENICE_DATA_DIR','/kaggle/input/datasets/khoilv2005/100-clients/100-clients')
ROLE_DIR=Path(os.environ.get('DENICE_CLEAN_ROLES_DIR','/kaggle/working/denice_clean_roles'))
OUTPUT_DIR=Path(os.environ.get('DENICE_OUTPUT_DIR',f'/kaggle/working/results_denice_cgofed_clean_seed_{TRAIN_SEED}'))
CHECKPOINT_BUDGET_GIB=12.0  # Checkpoints/output budget, not a claim about Kaggle's quota.
RESUME_ARCHIVE=os.environ.get('DENICE_CLEAN_RESUME_ARCHIVE')  # Optional clean task-boundary ZIP.
TASK_START=0
if RESUME_ARCHIVE:
    match=re.fullmatch(r'checkpoint_task_(\d+)_all_rounds.zip',Path(RESUME_ARCHIVE).name)
    if not match or not Path(RESUME_ARCHIVE).is_file():raise ValueError('Clean resume requires a completed task archive')
    TASK_START=int(match.group(1))+1
    if TASK_START>5:raise ValueError('All six tasks already complete')

code=os.environ.get('DENICE_CODE_DIR')
if not code:
    candidate=Path(globals().get('__file__','')).resolve().parent
    if (candidate/'fed_learning').is_dir():code=str(candidate)
if not code:
    code=tempfile.mkdtemp(prefix='denice-clean-source-')
    subprocess.run(['git','clone','--depth','1','https://github.com/khoilv2005/FL_IL_IDS.git',code],
        env={**os.environ,'GIT_LFS_SKIP_SMUDGE':'1'},check=True)
sys.path.insert(0,code)
if not Path(DATA_DIR,'metadata.json').is_file():
    candidates=list(Path('/kaggle/input').rglob('metadata.json')) if Path('/kaggle/input').exists() else []
    candidates=[p.parent for p in candidates if next(p.parent.glob('client_*_train.npz'),None) is not None]
    if len(candidates)!=1:raise FileNotFoundError('Set DENICE_DATA_DIR to the original 100-client dataset')
    DATA_DIR=str(candidates[0])
if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise ValueError('Output already contains a run. Choose a new DENICE_OUTPUT_DIR; never overwrite checkpoints.')
from tools.prepare_denice_clean_roles import prepare
from fed_learning.data.denice_clean_roles import CleanRoleData
if (ROLE_DIR/'role_lock.json').exists():
    roles=CleanRoleData(ROLE_DIR)
    if roles.source.resolve()!=Path(DATA_DIR).resolve() or roles.manifest['split_seed']!=ROLE_SPLIT_SEED:
        raise ValueError('Existing role split belongs to a different dataset or split seed')
else:
    prepare(DATA_DIR,ROLE_DIR,ROLE_SPLIT_SEED)

overrides=dict(
    data_dir=DATA_DIR,denice_clean_roles_dir=str(ROLE_DIR),output_dir=str(OUTPUT_DIR),
    seed=TRAIN_SEED,random_seed=TRAIN_SEED,task_start=TASK_START,task_end=5,
    resume_state_path=RESUME_ARCHIVE,save_resume_after_task=None,save_continuation_every_task=True,
    denice_clustering_mode='paper',denice_similarity_threshold=0.5,
    denice_cgofed_peer_projection=False,denice_amp_enabled=True,
    denice_cluster_edge_top_k=0,denice_collab_use_context_edges=True,
    denice_require_label_overlap=True,denice_max_clients=100,
    denice_max_train_samples_per_client=None,
    denice_checkpoint_format='delta',round_checkpoint_every=1,
    denice_archive_checkpoints=True,denice_checkpoint_storage_budget_gib=CHECKPOINT_BUDGET_GIB,
    denice_save_round_artifacts=False,
    denice_eval_last_round_only=True,denice_eval_final_task_only=True,
    denice_eval_terminal_state_only=True,
    denice_eval_final_round=True,denice_post_task_eval_tasks=[5],eval_every=999999,
    denice_eval_local_validation=False,denice_eval_max_clients=100,
    denice_eval_max_samples=50000,denice_eval_report_nomask=False,
    denice_eval_representative_ensemble=False,
    meta_peer_budget=16,meta_peer_budget_selection='pending clean validation',
)
if os.environ.get('DENICE_CONFIG_OVERRIDES'):
    raise ValueError('This clean launcher owns DENICE_CONFIG_OVERRIDES; edit its explicit protocol if needed')
os.environ.update(DENICE_VARIANT='cgofed',DENICE_TRAIN_PHASE='5',DENICE_CODE_DIR=code,
                  DENICE_SEED=str(TRAIN_SEED),DENICE_OUTPUT_DIR=str(OUTPUT_DIR),
                  DENICE_CONFIG_OVERRIDES=json.dumps(overrides))
print(f'CLEAN MAIN TRAINING: seed={TRAIN_SEED}, paper/xi=0.5, BASE only, tasks {TASK_START}..5.',flush=True)
print('Checkpoints: every round, compressed immediately; one verified ZIP per completed task.',flush=True)
print('Evaluation: task 5 final round only, on clean VALIDATION. FINAL TEST and Meta fitting remain locked.',flush=True)
runpy.run_path(str(Path(code)/'train_denice_cgofed_kaggle.py'),run_name='__main__')
