"""Evaluate a fold manifest with source-project model/evaluation functions."""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_INFERENCE_DIR = os.environ.get(
    "MENOPAUSE_SOURCE_INFERENCE_DIR", os.path.join(PACKAGE_ROOT, "src", "inference")
)
sys.path.insert(0, SOURCE_INFERENCE_DIR)
import inference as source_inference


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, choices=["resnet18", "resnet50", "densenet121", "mobilenet_v3_large", "efficientnet_b0"])
    parser.add_argument("--model_path", nargs="+", required=True)
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--resize_mode", default="padded", choices=["random", "padded", "zeropad", "reflect"])
    parser.add_argument("--region", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def split_dataframe(manifest_path, split):
    manifest = pd.read_csv(manifest_path)
    subset = manifest.loc[manifest["split"] == split, ["image_path", "label"]].copy()
    if subset.empty or subset["image_path"].duplicated().any() or not subset["image_path"].map(os.path.isfile).all():
        raise ValueError(f"Invalid {split} split in {manifest_path}")
    return subset


def main():
    args = parse_args()
    checkpoints = []
    for path in args.model_path:
        checkpoints.extend(sorted(glob.glob(path)) if any(token in path for token in "*?[") else [path])
    checkpoints = [path for path in checkpoints if os.path.isfile(path)]
    if len(checkpoints) != 5:
        raise ValueError(f"Expected five checkpoints, found {len(checkpoints)}")

    validation_df = split_dataframe(args.split_manifest, "validation")
    test_df = split_dataframe(args.split_manifest, "test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_loader = DataLoader(source_inference.HandXrayDataset(validation_df, args.resize_mode, train=False), batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(source_inference.HandXrayDataset(test_df, args.resize_mode, train=False), batch_size=32, shuffle=False, num_workers=4)

    results = []
    for checkpoint_path in checkpoints:
        model = source_inference.get_model(args.model_name, pretrained=False).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            threshold = checkpoint.get("threshold", 0.5)
        else:
            model.load_state_dict(checkpoint)
            threshold = 0.5
        results.append({
            "name": os.path.basename(checkpoint_path),
            "val": source_inference.evaluate_single_model(model, validation_loader, device, threshold),
            "test": source_inference.evaluate_single_model(model, test_loader, device, threshold),
        })

    metrics = ["acc", "auc", "ece", "sens", "spec"]
    rows = []
    for result in results:
        row = {"model_name": args.model_name, "weight_file": f"{args.region}_{result['name']}", "type": "single"}
        for metric in metrics:
            row[f"val_{metric}"] = round(result["val"][metric], 4)
            row[f"test_{metric}"] = round(result["test"][metric], 4)
        rows.append(row)
    for summary_type, reducer in (("ensemble_mean", np.mean), ("ensemble_std", np.std)):
        row = {"model_name": args.model_name, "weight_file": f"{args.region}_{summary_type} (5 seeds)", "type": summary_type}
        for metric in metrics:
            row[f"val_{metric}"] = round(reducer([result["val"][metric] for result in results]), 4)
            row[f"test_{metric}"] = round(reducer([result["test"][metric] for result in results]), 4)
        rows.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.output_dir, "evaluation_results.csv"), index=False)

    per_image_frames = []
    for result in results:
        for split in ("val", "test"):
            frame = result[split]["per_image"].copy()
            frame.insert(0, "split", split)
            frame.insert(0, "checkpoint", result["name"])
            frame.insert(0, "model_name", args.model_name)
            per_image_frames.append(frame)
    pd.concat(per_image_frames, ignore_index=True).to_csv(os.path.join(args.output_dir, "per_image_predictions.csv"), index=False)

    ensemble_frames = []
    for split in ("val", "test"):
        seed_frames = []
        for result in results:
            frame = result[split]["per_image"][["image_id", "label", "prob_class1"]].copy()
            seed_frames.append(frame.rename(columns={"prob_class1": result["name"]}))
        ensemble = seed_frames[0]
        for frame in seed_frames[1:]:
            ensemble = ensemble.merge(frame, on=["image_id", "label"], how="inner", validate="one_to_one")
        score_columns = [result["name"] for result in results]
        ensemble["mean_prob_class1"] = ensemble[score_columns].mean(axis=1)
        ensemble.insert(0, "split", split)
        ensemble.insert(0, "model_name", args.model_name)
        ensemble_frames.append(ensemble)
    pd.concat(ensemble_frames, ignore_index=True).to_csv(os.path.join(args.output_dir, "ensemble_scores.csv"), index=False)

    confidence_frames = []
    for result in results:
        frame = result["test"]["per_image"][["image_id", "true_class_probability"]].copy()
        confidence_frames.append(frame.rename(columns={"true_class_probability": result["name"]}))
    confidence = confidence_frames[0]
    for frame in confidence_frames[1:]:
        confidence = confidence.merge(frame, on="image_id", how="inner", validate="one_to_one")
    confidence["mean_true_class_probability"] = confidence[[result["name"] for result in results]].mean(axis=1)
    confidence.to_csv(os.path.join(args.output_dir, "true_class_confidence.csv"), index=False)

    print(f"[OK] Fold evaluation written to {args.output_dir}")


if __name__ == "__main__":
    main()
