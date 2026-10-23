"""Build the full Kaggle notebook for receiver-encoder head learning."""
import argparse
import json
from pathlib import Path

from build_candle_v3_notebook import SOURCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref',required=True,help='Published full runtime commit SHA')
    args = parser.parse_args()
    if len(args.ref)!=40 or any(c not in '0123456789abcdef' for c in args.ref):
        raise ValueError('Use a full Git commit SHA')
    root = Path(__file__).resolve().parents[1]
    source = SOURCE.replace('__CODE_REF__',args.ref).replace(
        '# CANDLE V3: historical DeNICE V1 configuration; edit paths only.',
        '# CANDLE V3 receiver-encoder head learning; full engineering baseline.').replace(
        '/kaggle/working/results_candle_v3_xi_0p8_seed_42',
        '/kaggle/working/results_candle_v3_receiver_head_xi_0p8_seed_42').replace(
        '/kaggle/temp/candle_v3_source','/kaggle/temp/candle_v3_receiver_head_source').replace(
        '"--data-dir", DATA_DIR, "--output", OUTPUT_DIR, "--roles", ROLES_DIR]',
        '"--peer-teaching", "--data-dir", DATA_DIR, "--output", OUTPUT_DIR, "--roles", ROLES_DIR]')
    markdown = '''# CANDLE V3 — learn head on the receiver encoder

Variant C: frozen receiver encoder, head-only updates from current donor and receiver BASE.
Donor data stays at its owner. The receiver aggregates compatible head deltas, builds
receiver-space router sketches and commits only after current CAL acceptance.
This notebook explicitly passes `--peer-teaching`; it does not copy donor FC2 rows.

## Kaggle setup

- Enable Internet and GPU. Use a fresh session and an empty OUTPUT_DIR.
- Attach `khoilv2005/100-clients`; nested dataset folders are resolved by the launcher.
- One GPU, AMP, seed 42, 100 clients, 6 tasks x 20 rounds, batch 2048, Adam lr 0.001.
- Same native V1 paper neighborhoods (xi=0.8), binary_cosine router and backbone as Direct FC2.
- Same original V1 roles: BASE 80%, FIT 8%, CAL 2%, validation 10% of original TRAIN.
  Each task uses only current data; no historical raw-data replay.
- After task finalization: at most 2 donors / 3 targets per receiver, 8 head aggregation
  rounds at lr 0.01; a separate donor-only control also runs 8 rounds. Backbone stays frozen
  within each teaching session. Current BASE/FIT/CAL budgets are unchanged.
- Acceptance: recall >=95%, observed FAR <=0.1%, negative break=0, gain over controls,
  at least 32 positive/negative CAL samples. Missing evidence never becomes PASS.
- Communication budget: 128 MiB/task including snapshots, updates, sketches and verification.
- No intermediate TEST. After T5/R19 and terminal consolidation: original full 34-class TEST
  on disjoint receiver shards; pooled BinarySelf primary, MulticlassSelf secondary.
- Round checkpoints compressed, ZIP per task, 12 GiB output guard. This is a fresh-run launcher;
  arbitrary round resume is not exposed. Source is cloned from pinned GitHub; no embedded bundle.

## Smoke evidence and comparison limits

25/25 structural tests and 83/83 native checks passed. Controlled synthetic install=1;
natural installs=0 in the two-client smoke. Native head learning executed all 8 rounds,
but its shadow candidate had insufficient positive CAL and failed quality gates; it was not installed.
Exact task-boundary resume matched. No TEST was read. Full-run effectiveness remains unproven.

This is the original Variant C profile, not a single-component ablation of Direct FC2:
C allows 2 donors/3 targets and announced-current-peer teaching edges, while the published
Direct FC2 profile allows 1 donor/1 target restricted to the actual native neighborhood.
The native training/data/TEST protocols are shared. Report these transfer-policy differences
when comparing accuracy, acceptance rates and bytes. Never infer quality from smoke PASS.

Output: `/kaggle/working/results_candle_v3_receiver_head_xi_0p8_seed_42/`.
Completion: `candle_v3_completion.json`; evaluation ZIP: `candle_v3_self_task_5.zip`.
'''
    notebook = dict(nbformat=4,nbformat_minor=5,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python',version='3.12')),cells=[
            dict(id='protocol',cell_type='markdown',metadata={},source=markdown.splitlines(True)),
            dict(id='full-run',cell_type='code',metadata={},execution_count=None,outputs=[],source=source.splitlines(True))])
    (root/'train_candle_v3_receiver_head_kaggle.py').write_text(source,encoding='utf-8')
    path = root/'train_candle_v3_receiver_head_kaggle.ipynb'
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    if path.stat().st_size>=1_000_000: raise ValueError('Kaggle source must be below 1 MB')
    print(f'{path.name}: {path.stat().st_size:,} bytes; source={args.ref}')


if __name__=='__main__': main()
