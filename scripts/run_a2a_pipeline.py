"""
A2a (CHEMBL251) Downstream Grid Search Pipeline
=================================================
Runs ECFP fingerprint + sklearn grid search across dataset sizes and seeds.

Run from the PROJECT ROOT:
    python scripts/run_a2a_pipeline.py
    python scripts/run_a2a_pipeline.py --task regression
    python scripts/run_a2a_pipeline.py --task classification
    python scripts/run_a2a_pipeline.py --seeds 42 123 456
    python scripts/run_a2a_pipeline.py --subsets 0.1 0.5 full
    python scripts/run_a2a_pipeline.py --models xgboost randomforest
    python scripts/run_a2a_pipeline.py --skip-generation
    python scripts/run_a2a_pipeline.py --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLD = 7.5          # pIC50 ≥ 7.5 → active (classification only)
DEFAULT_TASK      = "regression"       # "regression", "classification", or "both"

DEFAULT_SEEDS   = [42, 123, 2137]
DEFAULT_SUBSETS = ["0.2", "0.4", "0.6","0.8", "full"]
DEFAULT_MODELS  = ["mlp", "xgboost", "randomforest", "svm"]  # Matches choices in train_downstream_fp.py

SPLITS_DIR  = Path("data", "splits", "downstream")
SCRIPTS_DIR = Path("scripts")

def results_dir(task: str) -> Path:
    # Matches teammate's directory structure: results/downstream/A2a
    return Path("results", "downstream", "A2a")

# Metric names as returned by downstream/eval.py (capitalized)
PRIMARY_METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def a2a_paths(seed: int, subset: str) -> tuple[Path, Path, Path]:
    base = SPLITS_DIR / str(seed)
    return (
        base / subset / "A2a_train.csv",
        base / "A2a_val.csv",
        base / "A2a_test.csv",
    )


def run(cmd: list[str], dry_run: bool = False) -> str:
    print(f"\n$ {' '.join(cmd)}")
    if dry_run:
        return ""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(f"[STDERR]\n{result.stderr}", file=sys.stderr)
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n{result.stderr[-600:]}"
        )
    return result.stdout


def parse_test_metrics(stdout: str) -> tuple[dict[str, float], dict]:
    metrics: dict[str, float] = {}
    best_params: dict = {}
    in_metrics = False
    in_params  = False
    for line in stdout.splitlines():
        if "TEST RESULTS" in line:
            in_metrics, in_params = True, False
            continue
        if "BEST PARAMETERS" in line:
            in_params, in_metrics = True, False
            continue
        if in_metrics and ":" in line:
            key, _, val = line.partition(":")
            try:
                metrics[key.strip()] = float(val.strip())
            except ValueError:
                pass
        if in_params and line.strip().startswith("{"):
            try:
                import ast
                best_params = ast.literal_eval(line.strip())
            except Exception:
                pass
    return metrics, best_params


def pick_primary(columns) -> str | None:
    return next((m for m in PRIMARY_METRICS if m in columns), None)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – generate missing splits
# ──────────────────────────────────────────────────────────────────────────────

def generate_datasets(seeds: list[int], subsets: list[str], dry_run: bool):
    print("\n" + "=" * 60)
    print("STEP 1: Generating missing downstream splits")
    print("=" * 60)

    any_missing = False
    for seed in seeds:
        for subset in subsets:
            missing = [p for p in a2a_paths(seed, subset) if not p.exists()]
            if not missing:
                print(f"  [OK]   seed={seed}  subset={subset}")
                continue
            any_missing = True
            print(f"  [GEN]  seed={seed}  subset={subset}")
            cmd = [sys.executable, str(SCRIPTS_DIR / "generate_downstream_dataset.py"),
                   "--seed", str(seed)]
            if subset != "full":
                cmd += ["--subset", subset]
            run(cmd, dry_run=dry_run)

    if not any_missing:
        print("  All splits already present.")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – grid search
# ──────────────────────────────────────────────────────────────────────────────

def run_grid_search(
    seeds: list[int],
    subsets: list[str],
    models: list[str],
    task: str,
    threshold: float,
    dry_run: bool,
) -> list[dict]:
    print("\n" + "=" * 60)
    print(f"STEP 2: Grid search  [{task}]")
    print("=" * 60)

    combos  = [(s, sub, m) for s in seeds for sub in subsets for m in models]
    records = []

    for i, (seed, subset, model_name) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}]  seed={seed}  subset={subset}  model={model_name}")
        train_csv, val_csv, test_csv = a2a_paths(seed, subset)

        missing = [p for p in (train_csv, val_csv, test_csv) if not p.exists()]
        if missing and not dry_run:
            print(f"  [SKIP] Missing: {[p.name for p in missing]}")
            records.append({"seed": seed, "subset": subset, "model": model_name,
                            "elapsed_s": 0, "status": "missing_data"})
            continue

        cmd = [
            sys.executable, str(SCRIPTS_DIR / "train_downstream_fp.py"),
            "--model",     model_name,
            "--task",      task,
            "--train-csv", str(train_csv),
            "--val-csv",   str(val_csv),
            "--test-csv",  str(test_csv),
            "--threshold", str(threshold),
        ]

        t0 = time.time()
        try:
            stdout  = run(cmd, dry_run=dry_run)
            elapsed = round(time.time() - t0, 1)
            metrics, best_params = parse_test_metrics(stdout)
            
            # Setup the filename
            txt_filename = f"{model_name}_seed-{seed}_size-{subset}.txt"
            
            # Format metrics with commas to match European CSV formatting
            def fmt_metric(key):
                val = metrics.get(key)
                return str(round(val, 4)).replace(".", ",") if val is not None else ""

            # Append the exact structure your teammate uses
            records.append({
                "target": "A2a",
                "model": model_name,
                "seed": seed,
                "train_size": subset,
                "accuracy": fmt_metric("Accuracy"),
                "precision": fmt_metric("Precision"),
                "recall": fmt_metric("Recall"),
                "f1": fmt_metric("F1"),
                "roc_auc": fmt_metric("ROC_AUC"),
                "best_params": str(best_params),
                "file": txt_filename
            })
            
            # Save individual output text file for this run combination
            if not dry_run:
                out_dir = results_dir(task)
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / txt_filename, "w") as f:
                    f.write(stdout)

            if metrics:
                print("  → " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                      + f"  ({elapsed}s)")
        except RuntimeError as exc:
            records.append({"seed": seed, "subset": subset, "model": model_name,
                            "elapsed_s": round(time.time() - t0, 1), "status": "error"})
            print(f"  [ERROR] {exc}")

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Save & summarise
# ──────────────────────────────────────────────────────────────────────────────

def save_results(records: list[dict], task: str):
    out_dir = results_dir(task)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Saved to a static clean summary.csv file matching your teammate's style
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "summary.csv", index=False)

    ok = df[df["status"] == "ok"]
    if ok.empty:
        print("  No successful runs.")
        return

    primary = pick_primary(ok.columns)
    if not primary:
        return

    agg = (
        ok.groupby(["subset", "model"])[primary]
        .agg(mean="mean", std="std")
        .round(4)
    )
    agg.columns = [f"{primary}_mean", f"{primary}_std"]
    print(f"\n  [{task}]  Primary metric: {primary}\n")
    print(agg.to_string())

    wide = ok.pivot_table(
        index=["subset", "model"], columns="seed",
        values=primary, aggfunc="first",
    ).round(4)
    wide.columns.name = None
    wide["mean"] = wide.mean(axis=1).round(4)
    wide["std"]  = wide.std(axis=1).round(4)
    wide.to_csv(out_dir / f"pivot_{primary}.csv")
    print(f"\n  Saved → {out_dir}/")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="A2a (CHEMBL251) downstream grid search pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS,
                        help="Training-set fractions or 'full'")
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS,
                        default=DEFAULT_MODELS)
    parser.add_argument("--task", choices=["regression", "classification", "both"],
                        default=DEFAULT_TASK)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="pIC50 cut-off for active/inactive (classification only)")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip split generation (assume all CSVs already exist)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    return parser.parse_args()


def main():
    args = parse_args()

    subsets = []
    for s in args.subsets:
        s = str(s).lower()
        subsets.append("full" if s in ("full", "none", "1.0") else s)

    tasks = ["regression", "classification"] if args.task == "both" else [args.task]

    print("=" * 60)
    print("A2a (CHEMBL251) Downstream Grid Search")
    print("=" * 60)
    print(f"  Task(s):   {tasks}")
    print(f"  Seeds:     {args.seeds}")
    print(f"  Subsets:   {subsets}")
    print(f"  Models:    {args.models}")
    print(f"  Threshold: {args.threshold}  (classification only)")
    print(f"  Dry-run:   {args.dry_run}")
    print("=" * 60)

    if not args.skip_generation:
        generate_datasets(args.seeds, subsets, dry_run=args.dry_run)
    else:
        print("\n[SKIP] Step 1: dataset generation")

    for task in tasks:
        records = run_grid_search(
            seeds=args.seeds, subsets=subsets, models=args.models,
            task=task, threshold=args.threshold, dry_run=args.dry_run,
        )
        if records and not args.dry_run:
            save_results(records, task)

    print("\nDone.")


if __name__ == "__main__":
    main()