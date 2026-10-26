"""Generate a pinned GitHub-cloning Kaggle full-development notebook."""
import argparse
import json
from pathlib import Path

SOURCE='''# CANDLE V4 coverage-aware full development. Edit paths only.
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

CODE_REF = "__REF__"
DATA_DIR = "/kaggle/input/datasets/khoilv2005/100-clients"
OUTPUT_DIR = "/kaggle/working/results_candle_v4_handoff_multiclass_seed42"
VARIANT = "coverage"  # "native-only" reruns V1 with the SAME original-TRAIN protocol.
ROUTER_MODE = "multiclass_balanced"  # Locked in config; binary_cosine remains an explicit variant.
# Optional mounted round continuation .pt or task archive .zip; empty=fresh T0.
# Resume needs the same handoff-v6 run lock and router configuration.
RESUME_STATE = ""

if not Path('/kaggle/working').is_dir():
    raise RuntimeError('Run this notebook on Kaggle, not locally')
env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
for name in ('DENICE_CONFIG_FILE','DENICE_CONFIG_OVERRIDES','DENICE_RESUME_STATE_PATH',
             'DENICE_CONFIG_ONLY','DENICE_TASK_END'):
    env.pop(name,None)
code = Path('/kaggle/working/candle_v4_handoff_source')
if not code.exists():
    subprocess.check_call(['git','clone','--filter=blob:none','--no-checkout',
        'https://github.com/khoilv2005/FL_IL_IDS.git',str(code)],env=env)
    subprocess.check_call(['git','sparse-checkout','set','--no-cone','/candle_v4_runtime/'],cwd=code,env=env)
    subprocess.check_call(['git','checkout','--detach',CODE_REF],cwd=code,env=env)
actual = subprocess.check_output(['git','rev-parse','HEAD'],cwd=code,text=True).strip()
if actual != CODE_REF:raise ValueError('Source pin mismatch; use a fresh source folder')
runtime = code/'candle_v4_runtime'
subprocess.check_call([sys.executable,'-m','pip','install','-q','-r',
                       str(runtime/'requirements-candle-v1-kaggle.txt')],env=env)
args = [sys.executable,'-u',str(runtime/'tools/train_candle_v4.py'),
        '--data-dir',DATA_DIR,'--output',OUTPUT_DIR,'--variant',VARIANT,'--router-mode',ROUTER_MODE]
if RESUME_STATE:
    source = Path(RESUME_STATE);out=Path(OUTPUT_DIR);out.mkdir(parents=True,exist_ok=True)
    if source.suffix=='.zip':
        with zipfile.ZipFile(source) as archive:
            for name in archive.namelist():
                relative=Path(name)
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe resume archive')
                if name=='candle_v4_run_lock.json' or name.startswith(('v4_threshold_locks/','v4_gate_journal/')):
                    target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(archive.read(name))
    else:
        if source.parent.resolve()!=out.resolve():
            shutil.copy2(source.parent/'candle_v4_run_lock.json',out/'candle_v4_run_lock.json')
        if source.parent.resolve()!=out.resolve() and (source.parent/'v4_threshold_locks').is_dir():
            shutil.copytree(source.parent/'v4_threshold_locks',out/'v4_threshold_locks',dirs_exist_ok=True)
        if source.parent.resolve()!=out.resolve() and (source.parent/'v4_gate_journal').is_dir():
            shutil.copytree(source.parent/'v4_gate_journal',out/'v4_gate_journal',dirs_exist_ok=True)
    args += ['--resume',RESUME_STATE]
print('Pinned source:',actual,flush=True)
print('Full TEST runs only after Task5/Round19 consolidation; retain output with Kaggle Save Version.',flush=True)
subprocess.check_call(args,cwd=runtime,env=env)
import json
completion=json.loads(Path(OUTPUT_DIR,'candle_v1_completion.json' if VARIANT=='native-only' else 'candle_v4_completion.json').read_text())
print('Completed:',completion['completed'],'metrics:',
      {arm:{k:m[k] for k in ('accuracy','macro_f1')} for arm,m in completion['final_test'].items()})
'''


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ref',required=True)
    parser.add_argument('--v4-only',action='store_true')
    parser.add_argument('--continuous-only',action='store_true')
    parser.add_argument('--bootstrap-only',action='store_true');args=parser.parse_args()
    if len(args.ref)!=40 or any(c not in '0123456789abcdef' for c in args.ref):raise ValueError('Full public SHA required')
    root=Path(__file__).resolve().parents[1];source=SOURCE.replace('__REF__',args.ref)
    if args.continuous_only or args.bootstrap_only:
        source=source.replace('ROUTER_MODE = "multiclass_balanced"','ROUTER_MODE = "continuous_class_v1"').replace(
            'results_candle_v4_handoff_multiclass_seed42','results_candle_v4_continuous_class_seed42').replace(
            'candle_v4_handoff_source','candle_v4_continuous_source').replace('same handoff-v6 run lock','same continuous-v7 run lock').replace(
            'VARIANT = "coverage"  # "native-only" reruns V1 with the SAME original-TRAIN protocol.',
            'VARIANT = "coverage"  # V4 only. Keep the separate V1 notebook unchanged.')
        markdown='''# CANDLE V4 continuous_class_v1 — experimental variant

Same original TRAIN, current-task filtering, backbone loss/optimizer/steps, graph,
100 clients, seed42, six tasks x20 rounds and final T5/R19 TEST protocol.
V1 source and notebook remain unchanged. No validation/roles/shadow model.

The V4-only router keeps continuous selected activations and class/task identity
from the SAME native reference samples, capped by the existing per-class budget.
It does not retain raw inputs. Native binary memory/capsules and RNG are preserved.
Scaler and class-balanced LR fit only saved feature memory; class predictions map
to task IDs. No true task at inference. Expert handoff and scoped threshold .5 remain.

Native V1 multiclass remains a separate unchanged evaluation control on identical
TEST rows. Continuous-native and expert outputs are recorded without fitting TEST.
No extra per-round/per-task evaluation. Logging stays compact.

This variant has technical regression only, NO measured benchmark gain.
Fresh current-task memory capture is required: binary checkpoints cannot convert.
Keep RESUME_STATE empty for a new run; old v4/v5/v6 runs cannot resume as v7.
Enable GPU + Internet. Use Kaggle Save Version to retain checkpoint/output.
'''
        stem='train_candle_v4_continuous_router_kaggle'
        if args.bootstrap_only:
            stem='train_candle_v4_bootstrap_fixed_kaggle'
            source=source.replace('results_candle_v4_continuous_class_seed42','results_candle_v4_continuous_bootstrap_seed42').replace(
                'candle_v4_continuous_source','candle_v4_bootstrap_source').replace('same continuous-v7 run lock','same bootstrap-v8 run lock')
            source=source.replace("if RESUME_STATE:\n    source = Path(RESUME_STATE)","args += ['--continuous-bootstrap']\nif RESUME_STATE:\n    source = Path(RESUME_STATE)")
            markdown+='''
## V4 bootstrap state fix (protocol v8)

At an exact representative model clone, copy the matching continuous memory,
scaler and classifier without sharing array references. Preserve inherited tasks;
do not copy routing memory during partial catch-up. Record provenance and bytes.
No change to V1, raw TRAIN, loss, native optimizer/steps, graph, fusion threshold
or final TEST protocol. Existing v7 notebooks/results remain unchanged.

Fresh run: RESUME_STATE="". Do not resume results16/17 as v8: their missing state
is already absent and this fix is not a retroactive checkpoint migration.
Technical regressions only; no measured Accuracy gain from the fix yet.
'''
        notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),
            cells=[dict(cell_type='markdown',metadata={},source=markdown.splitlines(True)),
                   dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))])
        (root/(stem+'.py')).write_text(source,encoding='utf-8')
        (root/(stem+'.ipynb')).write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        print('Created separate continuous V4 notebook pinned to',args.ref)
        return
    markdown='''# CANDLE V4 — native coverage-aware, development seed 42

100 benchmark clients, six tasks ×20 rounds; native V1 batch2048/AMP, one GPU.
Experts use donor positives, receiver/donor negatives and current eligible
auxiliary peers. One coverage-aware bank; no shadow V1 model or parallel control banks.
Experts become ACTIVE directly after each round with authorized finite training updates.
Receipts are training_only_unverified; no recall/FAR/break quality acceptance.

Uses 100% original client TRAIN, filtered by current task; validation fraction0.0.
NO validation split, BASE/FIT/CAL preparation, role manifest or shadow native model.
Threshold .5, native fallback; only experts matching the input-predicted multiclass-router task may override.
When the current donor leaves the graph, retain the expert/Adam/quota and switch
to the smallest currently eligible donor ID. No future graph or historical TRAIN.
No true task labels at inference. Missing router profiles use native fallback.
Validation/quality metrics NOT_MEASURED before final TEST.
Old experts retain frozen weights and stay provisionally ACTIVE through task transitions.
**Full 34-class TEST runs once after Task5/Round19 and consolidation**.
V1 native counterfactual and V4 use identical disjoint shards covering the entire TEST.
Set VARIANT="native-only" and a separate OUTPUT_DIR for a standalone matched V1 rerun.
No validation/acceptance callback. Predicted-task-scoped fusion, threshold .5.
V4 expert console summaries: first round and every 5 rounds. No per-sample expert logs.
No MLP parameter aggregation into the CNN-GRU backbone.
Old checkpoints, including direct-fusion v4 and scoped-v5, cannot resume handoff-v6 training.
Start fresh; never change gate/source or resume under a new protocol mid-run.
No historical raw-data replay. Round continuation and task ZIPs contain expert state.
Enable **GPU and Internet**. Source is cloned at a fixed Git SHA; no embedded runtime.
Keep the output with Kaggle Save Version. A stopped interactive session is not an
automatic saved notebook output. Set RESUME_STATE to resume a retained continuation.
'''
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python')),
        cells=[dict(cell_type='markdown',metadata={},source=markdown.splitlines(True)),
               dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))])
    (root/'train_candle_v4_kaggle.py').write_text(source,encoding='utf-8')
    (root/'train_candle_v4_kaggle.ipynb').write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    if args.v4_only:
        print('Created scoped V4 notebook pinned to',args.ref)
        return
    v1source=source.replace('VARIANT = "coverage"','VARIANT = "native-only"').replace(
        'results_candle_v4_handoff_multiclass_seed42','results_candle_v1_full_train_seed42').replace(
        'candle_v4_handoff_source','candle_v1_full_train_source').replace(
        'ROUTER_MODE = "multiclass_balanced"','ROUTER_MODE = "binary_cosine"').replace(
        '# CANDLE V4 coverage-aware full development. Edit paths only.',
        '# CANDLE V1 full-TRAIN baseline; native-only. Edit paths only.')
    v1notebook=json.loads(json.dumps(notebook))
    v1notebook['cells'][0]['source']=('''# CANDLE V1 — full-TRAIN baseline, seed42
Native V1 only, no experts or shadow model. Same original current-task TRAIN,
validation fraction0.0, six tasks x20 rounds, batch2048/AMP and source as V4.
No roles or intermediate evaluation; full34-class TEST after T5/R19 consolidation.
Fresh run required. Historical V1 notebook/protocol remains available in Git history.
Enable GPU + Internet and retain outputs using Kaggle Save Version.
''').splitlines(True)
    v1notebook['cells'][1]['source']=v1source.splitlines(True)
    (root/'train_candle_v1_kaggle.py').write_text(v1source,encoding='utf-8')
    (root/'train_candle_v1_kaggle.ipynb').write_text(json.dumps(v1notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Created full-TRAIN V1 and V4 notebooks pinned to',args.ref)


if __name__=='__main__':main()
