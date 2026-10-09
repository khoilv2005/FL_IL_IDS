"""Full DeNICE legacy training, xi=.8, with frozen balanced multiclass self routing."""
import json
import os
from pathlib import Path
import runpy
import re
import subprocess
import sys
import tempfile
from importlib.metadata import version, PackageNotFoundError

# Change xi here, or set DENICE_SIMILARITY_THRESHOLD before running the cell.
SIMILARITY_THRESHOLD=float(os.environ.get('DENICE_SIMILARITY_THRESHOLD','0.8'))
# Run seed 42 first, before the replication campaign.
TRAIN_SEED=int(os.environ.get('DENICE_SEED','42'))
ROLE_SPLIT_SEED=20261006  # Same holdout split across independent training seeds.
DATA_DIR=os.environ.get('DENICE_DATA_DIR','/kaggle/input/datasets/khoilv2005/100-clients/100-clients')
ROLE_DIR=Path(os.environ.get('DENICE_CLEAN_ROLES_DIR','/kaggle/working/denice_clean_roles'))
XI_TAG=format(SIMILARITY_THRESHOLD,'.8g').replace('.','p')
OUTPUT_DIR=Path(os.environ.get('DENICE_OUTPUT_DIR',f'/kaggle/working/results_denice_legacy_multiclass_xi_{XI_TAG}_seed_{TRAIN_SEED}'))
CHECKPOINT_BUDGET_GIB=12.0  # Checkpoints/output budget, not a claim about Kaggle's quota.
START_FRESH=True  # Fresh legacy experiment starts from task 0 with new weights.
RESUME_ARCHIVE=None if START_FRESH else os.environ.get('DENICE_CLEAN_RESUME_ARCHIVE')
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
if 'EMBEDDED_LEGACY_SOURCES' in globals():
    for relative, source in EMBEDDED_LEGACY_SOURCES.items():
        Path(code,relative).write_text(source,encoding='utf-8')
# Match the historical Gate/Class Meta implementation; pin before importing sklearn.
try:sklearn_version=version('scikit-learn')
except PackageNotFoundError:sklearn_version=None
if sklearn_version!='1.6.1':
    subprocess.run([sys.executable,'-m','pip','install','--quiet','scikit-learn==1.6.1'],check=True)
if not Path(DATA_DIR,'metadata.json').is_file():
    candidates=list(Path('/kaggle/input').rglob('metadata.json')) if Path('/kaggle/input').exists() else []
    candidates=[p.parent for p in candidates if next(p.parent.glob('client_*_train.npz'),None) is not None]
    if len(candidates)!=1:raise FileNotFoundError('Set DENICE_DATA_DIR to the original 100-client dataset')
    DATA_DIR=str(candidates[0])
if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise ValueError('Output already contains a run. Choose a new DENICE_OUTPUT_DIR; never overwrite checkpoints.')
from tools.prepare_denice_clean_roles import prepare
from fed_learning.data.denice_clean_roles import CleanRoleData, validate_similarity_threshold
SIMILARITY_THRESHOLD=validate_similarity_threshold(SIMILARITY_THRESHOLD)
if (ROLE_DIR/'role_lock.json').exists():
    roles=CleanRoleData(ROLE_DIR)
    if roles.source.resolve()!=Path(DATA_DIR).resolve() or roles.manifest['split_seed']!=ROLE_SPLIT_SEED:
        raise ValueError('Existing role split belongs to a different dataset or split seed')
else:
    prepare(DATA_DIR,ROLE_DIR,ROLE_SPLIT_SEED)

overrides=dict(
    data_dir=DATA_DIR,denice_clean_roles_dir=str(ROLE_DIR),output_dir=str(OUTPUT_DIR),
    seed=TRAIN_SEED,random_seed=TRAIN_SEED,task_start=TASK_START,task_end=5,
    rounds_per_task=20,
    resume_state_path=RESUME_ARCHIVE,save_resume_after_task=None,save_continuation_every_task=True,
    denice_clustering_mode='paper',denice_similarity_threshold=SIMILARITY_THRESHOLD,
    denice_cgofed_peer_projection=False,denice_amp_enabled=True,
    denice_cluster_edge_top_k=0,denice_collab_use_context_edges=True,
    denice_require_label_overlap=True,denice_max_clients=100,
    denice_max_train_samples_per_client=None,
    denice_checkpoint_format='delta',round_checkpoint_every=1,
    denice_archive_checkpoints=True,denice_checkpoint_storage_budget_gib=CHECKPOINT_BUDGET_GIB,
    denice_save_round_artifacts=False,
    denice_eval_last_round_only=True,denice_eval_final_task_only=True,
    denice_eval_terminal_state_only=True,
    denice_eval_final_round=False,denice_post_task_eval=False,denice_post_task_eval_tasks=[],eval_every=999999,
    denice_eval_local_validation=False,denice_eval_max_clients=100,
    denice_evaluation_data_role='test',denice_eval_max_samples=None,
    denice_eval_lazy_client_shards=True,denice_eval_report_nomask=False,
    denice_eval_representative_ensemble=False,
    denice_cme_after_each_task=False,denice_cme_tasks=[],
    denice_clean_protocol='legacy_multiclass',denice_cl_method='legacy',
    denice_router_mode='binary_cosine',denice_eval_route_mode='hard',
    denice_replay_capacity=0,denice_router_replay_enabled=False,denice_memory_policy='sketches',
    denice_plasticity_enabled=False,denice_continual_width=0,
    denice_classifier_enabled=False,denice_transfer_enabled=False,
    denice_shared_context_eval=False,denice_calibrate_plastic_bn=True,
)
if os.environ.get('DENICE_CONFIG_OVERRIDES'):
    raise ValueError('This clean launcher owns DENICE_CONFIG_OVERRIDES; edit its explicit protocol if needed')
os.environ.update(DENICE_VARIANT='legacy',DENICE_TRAIN_PHASE='5',DENICE_CODE_DIR=code,
                  DENICE_SEED=str(TRAIN_SEED),DENICE_OUTPUT_DIR=str(OUTPUT_DIR),
                  DENICE_CONFIG_OVERRIDES=json.dumps(overrides))
print(f'CLEAN MAIN TRAINING: seed={TRAIN_SEED}, paper/xi={SIMILARITY_THRESHOLD}, BASE only, tasks {TASK_START}..5.',flush=True)
print(f'Fresh initialization={START_FRESH}; output={OUTPUT_DIR}',flush=True)
print('Checkpoints: every round, compressed immediately; one verified ZIP per completed task.',flush=True)
print('Legacy training: no CGoFed, replay, Competence or CME; binary context router during training.',flush=True)
print('After Task 5 / Round 19: freeze backbone, refit multiclass router from context memory, self-only full test.',flush=True)
print('Report BinarySelf and MulticlassSelf on identical disjoint receiver shards of ALL 34-class test rows.',flush=True)
print('Missing class support, including class 28, is report-only; data/checkpoint integrity remains enforced.',flush=True)
runpy.run_path(str(Path(code)/'train_incremental_kaggle.py'),run_name='__main__')

import gc
import shutil
import torch
from tools.eval_denice_legacy_self import run_legacy_self
gc.collect()
if torch.cuda.is_available():torch.cuda.empty_cache()
summary=run_legacy_self(OUTPUT_DIR/'checkpoint_task_5_all_rounds.zip',ROLE_DIR,
    OUTPUT_DIR/'legacy_self_task_5',DATA_DIR,
    device='cuda' if torch.cuda.is_available() else 'cpu',batch_size=512,expected_xi=SIMILARITY_THRESHOLD)
print('FULL TEST: legacy BinarySelf vs legacy MulticlassSelf:',json.dumps(summary['metrics'],indent=2),flush=True)
artifact=shutil.make_archive(str(OUTPUT_DIR/'legacy_self_task_5'),'zip',root_dir=OUTPUT_DIR/'legacy_self_task_5')
print('Legacy self evaluation ZIP:',artifact,flush=True)
