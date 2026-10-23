"""Build a small GitHub-cloning notebook; no embedded source/artifact maps."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def build():
    old = (ROOT / 'train_denice_legacy_multiclass_kaggle.py').read_text(encoding='utf-8')
    code = old[:old.index('\nimport gc\n')]
    code = code.replace('"""Full DeNICE legacy training, xi=.8, with frozen balanced multiclass self routing."""',
        '"""Fresh DeNICE + automatic APPLIANCE, six tasks x 20 rounds, xi=.8."""')
    code = code.replace("f'/kaggle/working/results_denice_legacy_multiclass_xi_{XI_TAG}_seed_{TRAIN_SEED}'",
        "f'/kaggle/working/results_denice_{METHOD_TAG}_xi_{XI_TAG}_seed_{TRAIN_SEED}'")
    code = code.replace("XI_TAG=format(",
        "METHOD_TAG='appliance' if os.environ.get('DENICE_APPLIANCE_ENABLED','1')=='1' else 'legacy_control'\nXI_TAG=format(", 1)
    start = code.index("if 'EMBEDDED_LEGACY_SOURCES' in globals():")
    end = code.index('# Match the historical', start)
    code = code[:start] + '''if not Path(code,'appliance/training_service.py').is_file():
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
''' + code[end:]
    pos = code.index('\noverrides=dict(')
    preparation = '''
STORE=Path(os.environ.get('APPLIANCE_RUNTIME_DATA_DIR','/kaggle/working/appliance_current_runtime_data'))
if ENABLED and not (STORE/'completion.json').is_file():
    from types import SimpleNamespace
    from tools.prepare_appliance_current_runtime_data import run as prepare_runtime
    prepare_runtime(SimpleNamespace(out=STORE,roles=ROLE_DIR,data_dir=Path(DATA_DIR)))
if ENABLED:
    stage=json.loads((STORE/'completion.json').read_text(encoding='utf-8'))
    if not stage.get('completed') or stage['client_count']!=100:
        raise ValueError('Locked current BASE/CAL stores must contain all 100 owners')
'''
    code = code[:pos] + preparation + code[pos:]
    code = code.replace("    denice_cme_after_each_task=False,denice_cme_tasks=[],", """    denice_cme_after_each_task=False,denice_cme_tasks=[],
    appliance_enabled=ENABLED,appliance_integration_smoke=False,
    appliance_base_store=str(STORE/'base_store'),
    appliance_calibration_store=str(STORE/'calibration_store'),
    appliance_application_domain='cumulative_dataset',appliance_batch_size=512,""")
    code = code.replace("appliance_application_domain='cumulative_dataset',appliance_batch_size=512,",
        """appliance_application_domain='cumulative_dataset',appliance_batch_size=512,
    appliance_scope_mode='appliance_empirical_current_CAL_v2',
    appliance_discovery_max_transactions=16,appliance_discovery_max_receivers_per_class=8,""")
    tail = code.index("print(f'CLEAN MAIN TRAINING:")
    code = code[:tail] + '''print(f'FRESH FULL: legacy DeNICE + APPLIANCE={ENABLED}; seed42; xi=.8; T0..T5; 20 rounds/task.',flush=True)
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
    OUTPUT_DIR/'full_test_task_5',DATA_DIR,device='auto',
    batch_size=512,expected_xi=.8,include_appliance=ENABLED)
print('Frozen full test, one disjoint receiver per row:',json.dumps(summary['metrics'],indent=2),flush=True)
print('Evaluation ZIP:',shutil.make_archive(str(OUTPUT_DIR/'full_test_task_5'),'zip',root_dir=OUTPUT_DIR/'full_test_task_5'),flush=True)
'''
    # The inherited launcher clones GitHub main when no local code directory is supplied.
    code = code.replace("sys.path.insert(0,code)", """os.environ['DENICE_CODE_DIR']=str(code)
revision=subprocess.run(['git','-C',code,'rev-parse','HEAD'],capture_output=True,text=True,check=False)
print('Training source:',code,'commit:',revision.stdout.strip() or 'local directory',flush=True)
sys.path.insert(0,code)""", 1)
    (ROOT / 'train_denice_appliance_full_kaggle.py').write_text(code, encoding='utf-8')
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        cells=[dict(cell_type='markdown',metadata={},source=[
            '# DeNICE + APPLIANCE — full fresh training\n',
            'Seed 42 · ξ=0.8 · six tasks × 20 rounds. Clones the latest GitHub main; no runtime ZIP or embedded sources required.\n',
            'Full test runs only after Task5 finalization and locking the multiclass routers.\n']),
            dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=code.splitlines(True))])
    (ROOT / 'train_denice_appliance_full_kaggle.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')


if __name__=='__main__':
    build()
