"""Fresh DeNICE + automatic APPLIANCE, six tasks x 20 rounds, xi=.8."""
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
METHOD_TAG='appliance' if os.environ.get('DENICE_APPLIANCE_ENABLED','1')=='1' else 'legacy_control'
XI_TAG=format(SIMILARITY_THRESHOLD,'.8g').replace('.','p')
OUTPUT_DIR=Path(os.environ.get('DENICE_OUTPUT_DIR',f'/kaggle/working/results_denice_{METHOD_TAG}_xi_{XI_TAG}_seed_{TRAIN_SEED}'))
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
os.environ['DENICE_CODE_DIR']=str(code)
revision=subprocess.run(['git','-C',code,'rev-parse','HEAD'],capture_output=True,text=True,check=False)
print('Training source:',code,'commit:',revision.stdout.strip() or 'local directory',flush=True)
sys.path.insert(0,code)
if not Path(code,'appliance/training_service.py').is_file():
    raise FileNotFoundError('GitHub source lacks automatic APPLIANCE; push the complete training integration first.')
ENABLED=os.environ.get('DENICE_APPLIANCE_ENABLED','1')=='1'
if SIMILARITY_THRESHOLD!=0.8 or TRAIN_SEED!=42:
    raise ValueError('First locked campaign uses xi=.8, seed42; replication follows separately')
REPORT=Path(code,'artifacts/appliance_production_integration_smoke.json')
if ENABLED:
    if not REPORT.is_file():raise FileNotFoundError('Production smoke certificate not yet available; do not launch full training')
    certificate=json.loads(REPORT.read_text(encoding='utf-8'))
    if not certificate.get('completed'):raise ValueError('Production smoke has not passed')
    if (certificate.get('scope_mode')!='appliance_empirical_current_CAL_v2' or
            not certificate.get('imported_route_activation_verified')):
        raise ValueError('The empirical deployment/activation smoke lock is required; the older strict-scope lock is insufficient')
    import hashlib
    for relative,expected in certificate['source_sha256'].items():
        if hashlib.sha256(Path(code,relative).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Source changed after smoke lock: {relative}')
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

STORE=Path(os.environ.get('APPLIANCE_RUNTIME_DATA_DIR','/kaggle/working/appliance_current_runtime_data'))
if ENABLED and not (STORE/'completion.json').is_file():
    from types import SimpleNamespace
    from tools.prepare_appliance_current_runtime_data import run as prepare_runtime
    prepare_runtime(SimpleNamespace(out=STORE,roles=ROLE_DIR,data_dir=Path(DATA_DIR)))
if ENABLED:
    stage=json.loads((STORE/'completion.json').read_text(encoding='utf-8'))
    if not stage.get('completed') or stage['client_count']!=100:
        raise ValueError('Locked current BASE/CAL stores must contain all 100 owners')

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
    appliance_enabled=ENABLED,appliance_integration_smoke=False,
    appliance_base_store=str(STORE/'base_store'),
    appliance_calibration_store=str(STORE/'calibration_store'),
    appliance_application_domain='cumulative_dataset',appliance_batch_size=512,
    appliance_scope_mode='appliance_empirical_current_CAL_v2',
    appliance_discovery_max_transactions=16,appliance_discovery_max_receivers_per_class=8,
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
print(f'FRESH FULL: legacy DeNICE + APPLIANCE={ENABLED}; seed42; xi=.8; T0..T5; 20 rounds/task.',flush=True)
print('Automatic discovery each round and after native maturation; current CAL only; no CGoFed/CME.',flush=True)
print('Delta checkpoint every round + atomic latest full continuation; seal task ZIP at task end.',flush=True)
print('Empirical deployment: accepted fixed guards may run outside finite CAL evidence; no population safety claim.',flush=True)
print('Observed FAR/break conflict or head/guard drift suspends a patch; initial CAL evidence stays immutable.',flush=True)
runpy.run_path(str(Path(code)/'train_incremental_kaggle.py'),run_name='__main__')
import gc
import shutil
import torch
from tools.eval_denice_legacy_self import run_legacy_self
gc.collect()
if torch.cuda.is_available():torch.cuda.empty_cache()
summary=run_legacy_self(OUTPUT_DIR/'checkpoint_task_5_all_rounds.zip',ROLE_DIR,
    OUTPUT_DIR/'full_test_task_5',DATA_DIR,device='cpu' if ENABLED else ('cuda' if torch.cuda.is_available() else 'cpu'),
    batch_size=512,expected_xi=.8,include_appliance=ENABLED)
print('Frozen full test, one disjoint receiver per row:',json.dumps(summary['metrics'],indent=2),flush=True)
print('Evaluation ZIP:',shutil.make_archive(str(OUTPUT_DIR/'full_test_task_5'),'zip',root_dir=OUTPUT_DIR/'full_test_task_5'),flush=True)
