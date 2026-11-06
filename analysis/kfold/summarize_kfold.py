#!/usr/bin/env python3
"""Summarize seven-fold, five-seed evaluations for Supplementary Table S5.

For every architecture or ray condition, each metric is first averaged across
the five seed-specific models within a fold.  The seven fold-level means are
then summarized using their arithmetic mean and sample standard deviation
(ddof=1), matching the Supplementary Methods and Table S5 footnote.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPERIMENTS = (
    ("Baseline architectures", "baseline", "resnet18", "ResNet18"),
    ("Baseline architectures", "baseline", "resnet50", "ResNet50"),
    ("Baseline architectures", "baseline", "densenet121", "DenseNet121"),
    ("Baseline architectures", "baseline", "efficientnet_b0", "EfficientNet-B0"),
    ("Baseline architectures", "baseline", "mobilenet_v3_large", "MobileNetV3-Large"),
    ("Selective ray masking", "ablation", "ray2_selective", "Second ray masked"),
    ("Selective ray masking", "ablation", "ray3_selective", "Third ray masked"),
    ("Selective ray masking", "ablation", "ray4_selective", "Fourth ray masked"),
    ("Selected-ray image analysis", "ablation", "ray2_selected", "Second ray only"),
    ("Selected-ray image analysis", "ablation", "ray3_selected", "Third ray only"),
    ("Selected-ray image analysis", "ablation", "ray4_selected", "Fourth ray only"),
)

METRICS = {
    "accuracy": "test_acc",
    "auc": "test_auc",
    "sensitivity": "test_sens",
    "specificity": "test_spec",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kfold-root",
        type=Path,
        default=Path("./outputs/kfold"),
        help="Root containing baseline/evaluation and ablation/evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./outputs/kfold/summary"),
        help="Directory for fold-level and Table-S5 summary CSV files.",
    )
    parser.add_argument("--n-folds", type=int, default=7)
    parser.add_argument("--n-seeds", type=int, default=5)
    return parser.parse_args()


def load_fold_means(
    kfold_root: Path,
    source_group: str,
    experiment: str,
    n_folds: int,
    n_seeds: int,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for fold in range(1, n_folds + 1):
        csv_path = (
            kfold_root
            / source_group
            / "evaluation"
            / experiment
            / f"fold_{fold}"
            / "evaluation_results.csv"
        )
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing fold evaluation: {csv_path}")

        frame = pd.read_csv(csv_path)
        required = {"type", *METRICS.values()}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")

        seed_rows = frame.loc[frame["type"].eq("single")].copy()
        if len(seed_rows) != n_seeds:
            raise ValueError(
                f"{csv_path}: expected {n_seeds} type=single rows, found {len(seed_rows)}"
            )

        fold_row: dict[str, float | int] = {"fold": fold, "n_seeds": len(seed_rows)}
        for output_name, source_name in METRICS.items():
            values = pd.to_numeric(seed_rows[source_name], errors="raise").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"{csv_path}: non-finite values in {source_name}")
            fold_row[output_name] = float(values.mean())
        rows.append(fold_row)
    return rows


def main() -> None:
    args = parse_args()
    if args.n_folds < 2:
        raise ValueError("--n-folds must be at least 2 to calculate a sample SD")

    fold_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    display_rows: list[dict[str, object]] = []

    for section, source_group, experiment, display_name in EXPERIMENTS:
        experiment_fold_rows = load_fold_means(
            args.kfold_root, source_group, experiment, args.n_folds, args.n_seeds
        )
        for row in experiment_fold_rows:
            fold_rows.append(
                {
                    "section": section,
                    "experiment": experiment,
                    "display_name": display_name,
                    **row,
                }
            )

        numeric_summary: dict[str, object] = {
            "section": section,
            "experiment": experiment,
            "display_name": display_name,
            "n_folds": args.n_folds,
            "seeds_per_fold": args.n_seeds,
        }
        display_summary: dict[str, object] = {
            "Section": section,
            "Architecture or condition": display_name,
        }
        for metric in METRICS:
            values = np.asarray([row[metric] for row in experiment_fold_rows], dtype=float)
            mean = float(values.mean())
            sample_sd = float(values.std(ddof=1))
            numeric_summary[f"{metric}_mean"] = mean
            numeric_summary[f"{metric}_sample_sd"] = sample_sd
            display_summary[metric.capitalize()] = f"{mean:.3f} ± {sample_sd:.3f}"

        summary_rows.append(numeric_summary)
        display_rows.append(display_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = args.output_dir / "k7_fold_level_seed_mean_metrics.csv"
    numeric_path = args.output_dir / "k7_summary_numeric.csv"
    table_path = args.output_dir / "supplementary_table_s5.csv"

    pd.DataFrame(fold_rows).to_csv(fold_path, index=False)
    pd.DataFrame(summary_rows).to_csv(numeric_path, index=False)
    pd.DataFrame(display_rows).to_csv(table_path, index=False)

    print(f"[OK] Fold-level five-seed means: {fold_path}")
    print(f"[OK] Seven-fold numeric summary: {numeric_path}")
    print(f"[OK] Supplementary Table S5 format: {table_path}")


if __name__ == "__main__":
    main()
