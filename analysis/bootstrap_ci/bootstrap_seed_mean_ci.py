#!/usr/bin/env python
"""Stratified test-patient bootstrap CIs for five-seed mean performance.

This intentionally does not retrain models or average image probabilities across
seeds.  It resamples the fixed held-out test images within class, applies each
same resample to every seed and condition, calculates seed-level metrics, and
then averages the five seed-level metrics for each bootstrap replicate.
"""

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
# Set MENOPAUSE_RESULTS_ROOT to the directory containing
# baseline_ci_evaluation/ and ablation_ci_evaluation/ prediction folders.
RESULTS_ROOT = Path(os.environ.get("MENOPAUSE_RESULTS_ROOT", PACKAGE_ROOT / "outputs"))

METRICS = ("auc", "accuracy", "sensitivity", "specificity")


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    table: str
    display_name: str
    prediction_csv: str


def experiment_definitions():
    baseline_root = RESULTS_ROOT / "baseline_ci_evaluation"
    ablation_root = RESULTS_ROOT / "ablation_ci_evaluation"
    return [
        Experiment("densenet121", "Table 2", "DenseNet121", str(baseline_root / "densenet121_primary" / "per_image_predictions.csv")),
        Experiment("resnet18", "Table 2", "ResNet18", str(baseline_root / "resnet18_primary" / "per_image_predictions.csv")),
        Experiment("resnet50", "Table 2", "ResNet50", str(baseline_root / "resnet50_primary" / "per_image_predictions.csv")),
        Experiment("mobilenet_v3_large", "Table 2", "MobileNetV3-Large", str(baseline_root / "mobilenet_primary_padded" / "per_image_predictions.csv")),
        Experiment("efficientnet_b0", "Table 2", "EfficientNet-B0", str(baseline_root / "efficientnet_primary_padded" / "per_image_predictions.csv")),
        Experiment("ray2_selective", "Table 3", "Ray2 selective", str(ablation_root / "ray2_selective" / "per_image_predictions.csv")),
        Experiment("ray2_selected", "Table 3", "Ray2 selected", str(ablation_root / "ray2_selected" / "per_image_predictions.csv")),
        Experiment("ray3_selective", "Table 3", "Ray3 selective", str(ablation_root / "ray3_selective" / "per_image_predictions.csv")),
        Experiment("ray3_selected", "Table 3", "Ray3 selected", str(ablation_root / "ray3_selected" / "per_image_predictions.csv")),
        Experiment("ray4_selective", "Table 3", "Ray4 selective", str(ablation_root / "ray4_selective" / "per_image_predictions.csv")),
        Experiment("ray4_selected", "Table 3", "Ray4 selected", str(ablation_root / "ray4_selected" / "per_image_predictions.csv")),
    ]


def seed_from_checkpoint(checkpoint):
    match = re.search(r"seed(\d+)", str(checkpoint))
    if not match:
        raise ValueError(f"Could not extract seed from checkpoint name: {checkpoint}")
    return int(match.group(1))


def load_experiment(experiment):
    if not os.path.isfile(experiment.prediction_csv):
        raise FileNotFoundError(experiment.prediction_csv)
    frame = pd.read_csv(experiment.prediction_csv)
    required = {"checkpoint", "split", "image_id", "label", "prob_class1", "prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{experiment.prediction_csv} missing columns: {sorted(missing)}")
    frame = frame.loc[frame["split"] == "test"].copy()
    frame["seed"] = frame["checkpoint"].map(seed_from_checkpoint)
    frame["label"] = frame["label"].astype(int)
    frame["prediction"] = frame["prediction"].astype(int)
    frame["prob_class1"] = frame["prob_class1"].astype(float)

    labels = frame[["image_id", "label"]].drop_duplicates()
    if labels["image_id"].duplicated().any():
        raise ValueError(f"Conflicting labels in {experiment.prediction_csv}")
    labels = labels.sort_values("image_id").reset_index(drop=True)
    seeds = sorted(frame["seed"].unique().tolist())
    if len(seeds) != 5:
        raise ValueError(f"{experiment.experiment_id}: expected five seeds, got {seeds}")

    probabilities, predictions = [], []
    for seed in seeds:
        seed_frame = frame.loc[frame["seed"] == seed, ["image_id", "prob_class1", "prediction"]]
        if seed_frame["image_id"].duplicated().any() or len(seed_frame) != len(labels):
            raise ValueError(f"{experiment.experiment_id}, seed {seed}: invalid test-image rows")
        merged = labels[["image_id"]].merge(seed_frame, on="image_id", how="left", validate="one_to_one")
        if merged[["prob_class1", "prediction"]].isna().any().any():
            raise ValueError(f"{experiment.experiment_id}, seed {seed}: test-image alignment failure")
        probabilities.append(merged["prob_class1"].to_numpy(dtype=float))
        predictions.append(merged["prediction"].to_numpy(dtype=int))

    return {
        "experiment": experiment,
        "image_ids": labels["image_id"].to_numpy(),
        "labels": labels["label"].to_numpy(dtype=int),
        "seeds": np.asarray(seeds, dtype=int),
        "probabilities": np.stack(probabilities),
        "predictions": np.stack(predictions),
    }


def metric_values(probabilities, predictions, labels):
    """Return values in METRICS order for arrays shaped (seed, image)."""
    positive = labels == 1
    negative = labels == 0
    values = np.empty((probabilities.shape[0], len(METRICS)), dtype=float)
    for seed_index in range(probabilities.shape[0]):
        values[seed_index, 0] = roc_auc_score(labels, probabilities[seed_index])
        values[seed_index, 1] = np.mean(predictions[seed_index] == labels)
        values[seed_index, 2] = np.mean(predictions[seed_index, positive] == 1)
        values[seed_index, 3] = np.mean(predictions[seed_index, negative] == 0)
    return values


def bootstrap_metrics(probabilities, predictions, sampled_negative, sampled_positive):
    """Seed-level metrics for one common batch of stratified bootstrap samples."""
    n_seeds = probabilities.shape[0]
    batch_size = sampled_negative.shape[0]
    n_negative = sampled_negative.shape[1]
    n_positive = sampled_positive.shape[1]
    values = np.empty((n_seeds, len(METRICS), batch_size), dtype=float)

    for seed_index in range(n_seeds):
        negative_scores = probabilities[seed_index, sampled_negative]
        positive_scores = probabilities[seed_index, sampled_positive]
        scores = np.concatenate((negative_scores, positive_scores), axis=1)
        ranks = rankdata(scores, axis=1, method="average")
        values[seed_index, 0] = (
            ranks[:, n_negative:].sum(axis=1) - n_positive * (n_positive + 1) / 2.0
        ) / (n_positive * n_negative)

        negative_predictions = predictions[seed_index, sampled_negative]
        positive_predictions = predictions[seed_index, sampled_positive]
        specificity = np.mean(negative_predictions == 0, axis=1)
        sensitivity = np.mean(positive_predictions == 1, axis=1)
        accuracy = (specificity * n_negative + sensitivity * n_positive) / (n_negative + n_positive)
        values[seed_index, 1] = accuracy
        values[seed_index, 2] = sensitivity
        values[seed_index, 3] = specificity
    return values


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-bootstrap", type=int, default=20000)
    parser.add_argument("--random-seed", type=int, default=20260927)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_ROOT / "seed_mean_stratified_bootstrap_ci"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_bootstrap < 1 or args.batch_size < 1:
        raise ValueError("n-bootstrap and batch-size must be positive")
    experiments = experiment_definitions()
    loaded = [load_experiment(experiment) for experiment in experiments]

    reference = loaded[0]
    for item in loaded[1:]:
        if not np.array_equal(item["image_ids"], reference["image_ids"]) or not np.array_equal(item["labels"], reference["labels"]):
            raise ValueError(f"{item['experiment'].experiment_id} does not share the same test image IDs/labels as baseline")
        if not np.array_equal(item["seeds"], reference["seeds"]):
            raise ValueError(f"{item['experiment'].experiment_id} has a different seed set")

    labels = reference["labels"]
    negative_indices = np.flatnonzero(labels == 0)
    positive_indices = np.flatnonzero(labels == 1)
    if len(negative_indices) == 0 or len(positive_indices) == 0:
        raise ValueError("Both classes are required for stratified bootstrap")

    os.makedirs(args.output_dir, exist_ok=True)
    observed_rows = []
    summary_rows = []
    observed_by_experiment = {}
    for item in loaded:
        observed = metric_values(item["probabilities"], item["predictions"], labels)
        observed_by_experiment[item["experiment"].experiment_id] = observed
        for seed, seed_values in zip(item["seeds"], observed):
            observed_rows.append({
                "table": item["experiment"].table,
                "experiment_id": item["experiment"].experiment_id,
                "model_or_condition": item["experiment"].display_name,
                "seed": int(seed),
                **{metric: float(value) for metric, value in zip(METRICS, seed_values)},
            })

    distributions = np.empty((len(loaded), len(METRICS), args.n_bootstrap), dtype=np.float32)
    rng = np.random.default_rng(args.random_seed)
    for start in range(0, args.n_bootstrap, args.batch_size):
        stop = min(start + args.batch_size, args.n_bootstrap)
        current_batch = stop - start
        # The same sampled test patients are used for all 5 seeds, all 11
        # experiments, and every metric in this bootstrap replicate.
        sampled_negative = rng.choice(negative_indices, size=(current_batch, len(negative_indices)), replace=True)
        sampled_positive = rng.choice(positive_indices, size=(current_batch, len(positive_indices)), replace=True)
        for experiment_index, item in enumerate(loaded):
            seed_values = bootstrap_metrics(item["probabilities"], item["predictions"], sampled_negative, sampled_positive)
            distributions[experiment_index, :, start:stop] = seed_values.mean(axis=0).astype(np.float32)
        if stop == args.n_bootstrap or start == 0 or (start // args.batch_size + 1) % 10 == 0:
            print(f"Bootstrap replicates: {stop}/{args.n_bootstrap}")

    for experiment_index, item in enumerate(loaded):
        observed = observed_by_experiment[item["experiment"].experiment_id]
        row = {
            "table": item["experiment"].table,
            "experiment_id": item["experiment"].experiment_id,
            "model_or_condition": item["experiment"].display_name,
            "n_test_images": int(len(labels)),
            "n_class0_younger": int(len(negative_indices)),
            "n_class1_older": int(len(positive_indices)),
            "n_seeds": int(len(item["seeds"])),
            "seed_values": ";".join(str(seed) for seed in item["seeds"]),
            "bootstrap_replicates": int(args.n_bootstrap),
            "bootstrap_random_seed": int(args.random_seed),
        }
        for metric_index, metric in enumerate(METRICS):
            row[f"{metric}_mean"] = float(np.mean(observed[:, metric_index]))
            # Match the existing evaluation_results.csv ensemble_std convention (np.std, ddof=0).
            row[f"{metric}_sd"] = float(np.std(observed[:, metric_index], ddof=0))
            row[f"{metric}_ci95_low"] = float(np.quantile(distributions[experiment_index, metric_index], 0.025))
            row[f"{metric}_ci95_high"] = float(np.quantile(distributions[experiment_index, metric_index], 0.975))
        summary_rows.append(row)

    pd.DataFrame(observed_rows).to_csv(os.path.join(args.output_dir, "seed_level_test_metrics.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(args.output_dir, "seed_mean_stratified_bootstrap_ci95_summary.csv"), index=False)
    np.savez_compressed(
        os.path.join(args.output_dir, "bootstrap_seed_mean_metric_distributions.npz"),
        distributions=distributions,
        experiment_ids=np.asarray([item["experiment"].experiment_id for item in loaded]),
        metric_names=np.asarray(METRICS),
    )
    metadata = {
        "method": "stratified nonparametric percentile bootstrap of held-out test patients",
        "n_bootstrap": args.n_bootstrap,
        "bootstrap_random_seed": args.random_seed,
        "common_resampling": "Each replicate uses the same within-class resampled image IDs for every seed, experiment, and metric.",
        "strata": {"class0_younger": int(len(negative_indices)), "class1_older": int(len(positive_indices))},
        "seed_aggregation": "Each replicate calculates each metric separately for all five fixed seed models, then takes their arithmetic mean. No image-level probability/logit ensemble is used.",
        "auc": "AUC is calculated from prob_class1. Accuracy, sensitivity, and specificity use the stored prediction values generated with each seed checkpoint's fixed validation-derived threshold.",
        "ci": "2.5th and 97.5th percentiles of the bootstrap distribution of five-seed mean performance.",
        "sd": "Observed seed-level SD uses ddof=0 to match existing evaluation_results.csv ensemble_std rows.",
        "experiments": [item["experiment"].experiment_id for item in loaded],
    }
    with open(os.path.join(args.output_dir, "bootstrap_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[OK] Summary: {os.path.join(args.output_dir, 'seed_mean_stratified_bootstrap_ci95_summary.csv')}")


if __name__ == "__main__":
    main()
