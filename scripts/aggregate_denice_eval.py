from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd


ZIP_RE = re.compile(r"^(?P<range>0\s*-\s*3|4\s*-\s*5)\s*seed\s*(?P<seed>\d+)\.zip$", re.I)
EXPECTED_SEEDS = [42, 43, 44, 45, 46]
EXPECTED_TASKS = list(range(6))
EXPECTED_ROUNDS = list(range(20))

METRIC_COLUMNS = [
    "train_loss",
    "test_loss",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
    "route_accuracy",
    "route_coverage",
]

META_COLUMNS = [
    "algorithm",
    "routing_mode",
    "eval_client_count",
    "eval_client_total",
    "eval_client_fraction",
    "eval_client_selection",
    "split_mode",
    "split_count",
    "cluster_K",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate DeNICE eval_5 seed zip outputs.")
    parser.add_argument("--downloads", default=r"C:\Users\khoak\Downloads")
    parser.add_argument("--workspace", default=str(Path.cwd()))
    parser.add_argument("--tmp-dir", default=".tmp_denice_eval_results")
    parser.add_argument("--prefix", default="denice_eval_5seed")
    parser.add_argument("--keep-extracted", action="store_true")
    parser.add_argument("--write-raw", action="store_true")
    return parser.parse_args()


def normalize_task_range(text: str) -> str:
    text = re.sub(r"\s+", "", text)
    return text.replace("-", "-")


def find_zip_files(downloads: Path) -> list[tuple[Path, str, int]]:
    found: list[tuple[Path, str, int]] = []
    for path in downloads.iterdir():
        if not path.is_file():
            continue
        match = ZIP_RE.match(path.name)
        if not match:
            continue
        task_range = normalize_task_range(match.group("range"))
        seed = int(match.group("seed"))
        if seed in EXPECTED_SEEDS:
            found.append((path, task_range, seed))
    return sorted(found, key=lambda item: (item[1], item[2], item[0].name))


def extract_and_read(zip_path: Path, task_range: str, seed: int, extract_root: Path) -> pd.DataFrame:
    target_dir = extract_root / f"{task_range.replace('-', '_')}_seed_{seed}"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir)

    matches = list(target_dir.rglob("results_denice/all_round_seed_metrics.csv"))
    if not matches:
        matches = list(target_dir.rglob("all_round_seed_metrics.csv"))
    if not matches:
        raise FileNotFoundError(f"No all_round_seed_metrics.csv in {zip_path}")

    df = pd.read_csv(matches[0])
    df["source_zip"] = zip_path.name
    df["source_task_range"] = task_range
    df["source_seed"] = seed
    return df


def numericize(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["task_id", "round_id", "split_seed", "source_seed", "seed_index"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in METRIC_COLUMNS + ["eval_client_count", "eval_client_total", "eval_client_fraction", "split_count", "cluster_K"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    group_cols = ["task_id", "round_id"]

    agg_spec: dict[str, tuple[str, str]] = {
        "num_seeds": ("source_seed", "nunique"),
    }
    for col in METRIC_COLUMNS:
        if col in raw.columns:
            agg_spec[f"{col}_mean"] = (col, "mean")
            agg_spec[f"{col}_std"] = (col, "std")
    for col in META_COLUMNS:
        if col in raw.columns:
            agg_spec[col] = (col, "first")

    summary = raw.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
    seeds = (
        raw.groupby(group_cols)["source_seed"]
        .apply(lambda s: ",".join(str(int(x)) for x in sorted(set(s.dropna()))))
        .reset_index(name="seeds")
    )
    zips = (
        raw.groupby(group_cols)["source_zip"]
        .apply(lambda s: "; ".join(sorted(set(str(x) for x in s.dropna()))))
        .reset_index(name="source_zips")
    )
    summary = summary.merge(seeds, on=group_cols, how="left").merge(zips, on=group_cols, how="left")
    return summary.sort_values(group_cols).reset_index(drop=True)


def missing_report(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    present = set(
        (int(r.task_id), int(r.round_id), int(r.source_seed))
        for r in raw[["task_id", "round_id", "source_seed"]].dropna().itertuples(index=False)
    )
    for task_id in EXPECTED_TASKS:
        for round_id in EXPECTED_ROUNDS:
            missing = [seed for seed in EXPECTED_SEEDS if (task_id, round_id, seed) not in present]
            if missing:
                rows.append({"task_id": task_id, "round_id": round_id, "missing_seeds": ",".join(map(str, missing))})
    return pd.DataFrame(rows)


def fmt_pct(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(std):
        std = 0.0
    return f"{mean * 100:.2f} +/- {std * 100:.2f}"


def fmt_num(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(std):
        std = 0.0
    return f"{mean:.4f} +/- {std:.4f}"


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_markdown(raw: pd.DataFrame, summary: pd.DataFrame, missing: pd.DataFrame, path: Path) -> None:
    final = summary[summary["round_id"] == 19].copy()
    final_rows = []
    for _, row in final.iterrows():
        final_rows.append(
            {
                "task": int(row["task_id"]),
                "round": int(row["round_id"]),
                "seeds": row["seeds"],
                "clients": f"{int(row['eval_client_count'])}/{int(row['eval_client_total'])}"
                if not pd.isna(row.get("eval_client_count")) and not pd.isna(row.get("eval_client_total"))
                else "",
                "K": int(row["cluster_K"]) if not pd.isna(row.get("cluster_K")) else "",
                "accuracy": fmt_pct(row.get("accuracy_mean"), row.get("accuracy_std")),
                "f1_weighted": fmt_pct(row.get("f1_weighted_mean"), row.get("f1_weighted_std")),
                "f1_macro": fmt_pct(row.get("f1_macro_mean"), row.get("f1_macro_std")),
                "route_acc": fmt_pct(row.get("route_accuracy_mean"), row.get("route_accuracy_std")),
                "test_loss": fmt_num(row.get("test_loss_mean"), row.get("test_loss_std")),
            }
        )
    final_df = pd.DataFrame(final_rows)

    best_rows = []
    for task_id, group in summary.groupby("task_id"):
        best_f1 = group.sort_values(["f1_weighted_mean", "accuracy_mean"], ascending=False).iloc[0]
        best_acc = group.sort_values(["accuracy_mean", "f1_weighted_mean"], ascending=False).iloc[0]
        best_rows.append(
            {
                "task": int(task_id),
                "best_f1_round": int(best_f1["round_id"]),
                "best_f1_weighted": fmt_pct(best_f1.get("f1_weighted_mean"), best_f1.get("f1_weighted_std")),
                "best_acc_round": int(best_acc["round_id"]),
                "best_accuracy": fmt_pct(best_acc.get("accuracy_mean"), best_acc.get("accuracy_std")),
            }
        )
    best_df = pd.DataFrame(best_rows)

    coverage = (
        raw.groupby(["source_task_range", "source_seed"])
        .agg(rows=("round_id", "count"), tasks=("task_id", lambda s: ",".join(str(int(x)) for x in sorted(set(s.dropna())))))
        .reset_index()
        .rename(columns={"source_task_range": "task_range", "source_seed": "seed"})
    )

    missing_preview = missing.copy()
    if not missing_preview.empty:
        missing_preview = missing_preview.head(30)

    text = f"""# DeNICE 5-Seed Evaluation Summary

Protocol: `eval_5.ipynb` proxy evaluation, global cumulative test split equally across active clients, aggregated over seeds `{','.join(map(str, EXPECTED_SEEDS))}`.

Raw rows: `{len(raw)}`

Summary rows: `{len(summary)}`

Missing task-round-seed entries: `{len(missing)}`

## Final Round Per Task

{markdown_table(final_df, ['task', 'round', 'seeds', 'clients', 'K', 'accuracy', 'f1_weighted', 'f1_macro', 'route_acc', 'test_loss'])}

## Best Round Per Task

{markdown_table(best_df, ['task', 'best_f1_round', 'best_f1_weighted', 'best_acc_round', 'best_accuracy'])}

## Input Coverage

{markdown_table(coverage, ['task_range', 'seed', 'rows', 'tasks'])}

## Missing Entries

{markdown_table(missing_preview, ['task_id', 'round_id', 'missing_seeds'])}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    downloads = Path(args.downloads)
    workspace = Path(args.workspace)
    extract_root = workspace / args.tmp_dir
    extract_root.mkdir(parents=True, exist_ok=True)

    zip_files = find_zip_files(downloads)
    expected_count = len(EXPECTED_SEEDS) * 2
    if len(zip_files) != expected_count:
        print(f"Warning: expected {expected_count} zip files, found {len(zip_files)}")
    for path, task_range, seed in zip_files:
        print(f"Found: {path.name} range={task_range} seed={seed}")

    frames = [extract_and_read(path, task_range, seed, extract_root) for path, task_range, seed in zip_files]
    raw = numericize(pd.concat(frames, ignore_index=True))
    raw = raw.sort_values(["task_id", "round_id", "source_seed"]).reset_index(drop=True)
    summary = aggregate(raw)
    missing = missing_report(raw)

    summary_path = workspace / f"{args.prefix}_summary.csv"
    md_path = workspace / f"{args.prefix}_summary.md"

    summary.to_csv(summary_path, index=False)
    write_markdown(raw, summary, missing, md_path)
    if args.write_raw:
        raw_path = workspace / f"{args.prefix}_raw.csv"
        missing_path = workspace / f"{args.prefix}_missing.csv"
        raw.to_csv(raw_path, index=False)
        missing.to_csv(missing_path, index=False)
        print(f"Wrote raw: {raw_path}")
        print(f"Wrote missing csv: {missing_path}")

    if not args.keep_extracted:
        shutil.rmtree(extract_root, ignore_errors=True)

    print(f"Wrote summary csv: {summary_path}")
    print(f"Wrote markdown: {md_path}")
    print(f"Rows raw={len(raw)} summary={len(summary)} missing={len(missing)}")


if __name__ == "__main__":
    main()
