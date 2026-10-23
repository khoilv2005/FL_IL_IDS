"""Build the Kaggle full-run notebook for the smoke-checked Direct FC2 variant."""
import argparse
import json
from pathlib import Path

from build_candle_v3_notebook import SOURCE


SMOKED_SOURCE = "88f76a2a986f47bacb924a4bf93fcd393d18d32d"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default=SMOKED_SOURCE)
    args = parser.parse_args()
    if len(args.ref) != 40 or any(c not in "0123456789abcdef" for c in args.ref):
        raise ValueError("Use a full Git commit SHA")
    root = Path(__file__).resolve().parents[1]
    source = SOURCE.replace("__CODE_REF__", args.ref).replace(
        "# CANDLE V3: historical DeNICE V1 configuration; edit paths only.",
        "# CANDLE V3 Direct FC2: full engineering baseline; edit paths only.",
    ).replace(
        "/kaggle/working/results_candle_v3_xi_0p8_seed_42",
        "/kaggle/working/results_candle_v3_direct_fc2_xi_0p8_seed_42",
    ).replace(
        "/kaggle/temp/candle_v3_source",
        "/kaggle/temp/candle_v3_direct_fc2_source",
    ).replace(
        '"--data-dir", DATA_DIR, "--output", OUTPUT_DIR, "--roles", ROLES_DIR]',
        '"--direct-fc2", "--data-dir", DATA_DIR, "--output", OUTPUT_DIR, "--roles", ROLES_DIR]',
    )
    markdown = """# CANDLE V3 Direct FC2 — full engineering baseline

Engineering smoke: 27/27 structural tests and 80/80 native checks passed.
One controlled synthetic install; zero natural installs in the bounded real-data smoke.
Transfer effectiveness and historical safety remain unproven. This is the first full run.

- Enable Internet and GPU on Kaggle; use a fresh session and empty OUTPUT_DIR.
- Attach `/kaggle/input/datasets/khoilv2005/100-clients`.
- Fresh seed 42, 100 clients, T0–T5, 20 rounds/task, batch 2048, one GPU with AMP.
- Native V1 paper neighborhoods, xi=0.8; one local model/router per client, binary_cosine.
- Direct FC2 transfer is enabled explicitly. No extra donor/head optimizer or fallback variant.
- Teaching runs at task end, bounded to one donor/target per receiver and 128 MiB per task.
- Uses the original V1 BASE/CAL/FIT/validation split; BASE is 80% of original TRAIN.
  Each task trains only its current BASE; no historical raw-data replay.
- Current FIT/CAL are used for transfer decisions. Missing evidence or rejected transfers
  do not stop native training; acceptance gates are unchanged.
- No intermediate TEST. After T5/R19 and terminal consolidation, evaluate full original
  34-class TEST on disjoint receiver shards: pooled BinarySelf primary, MulticlassSelf secondary.
- Round checkpoints are compressed and each task seals a ZIP; 12 GiB output guard.
  This fresh-run launcher does not expose arbitrary round resume.
- Source comes from the pinned GitHub revision; no embedded source, datasets or checkpoints.
  Existing mounted V1 roles can be reused via ROLES_DIR; otherwise roles are prepared first.

Results: `results_candle_v3_direct_fc2_xi_0p8_seed_42/` under Kaggle Output.
Completion: `candle_v3_completion.json`; final evaluation: `candle_v3_self_task_5.zip`.
"""
    notebook = dict(
        nbformat=4, nbformat_minor=5,
        metadata=dict(
            kernelspec=dict(display_name="Python 3", language="python", name="python3"),
            language_info=dict(name="python", version="3.12")),
        cells=[
            dict(id="protocol", cell_type="markdown", metadata={}, source=markdown.splitlines(True)),
            dict(id="full-run", cell_type="code", metadata={}, execution_count=None,
                 outputs=[], source=source.splitlines(True)),
        ],
    )
    (root / "train_candle_v3_direct_fc2_kaggle.py").write_text(source, encoding="utf-8")
    destination = root / "train_candle_v3_direct_fc2_kaggle.ipynb"
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if destination.stat().st_size >= 1_000_000:
        raise ValueError("Kaggle notebook exceeds 1 MB source limit")
    print(f"{destination.name}: {destination.stat().st_size:,} bytes; source={args.ref}")


if __name__ == "__main__":
    main()
