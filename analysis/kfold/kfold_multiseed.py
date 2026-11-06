"""5-seed training wrapper using a precomputed fold manifest."""

import argparse
import copy
import os
import sys

import pandas as pd

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_TRAIN_DIR = os.environ.get(
    "MENOPAUSE_SOURCE_TRAIN_DIR", os.path.join(PACKAGE_ROOT, "src", "train")
)
sys.path.insert(0, SOURCE_TRAIN_DIR)
import run_train as source_train

SEEDS = [42, 777, 2024, 1004, 8282]


def manifest_dataframe(manifest_path, split):
    manifest = pd.read_csv(manifest_path)
    required = {"split", "image_path", "label"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"{manifest_path} is missing {required - set(manifest.columns)}")
    subset = manifest.loc[manifest["split"] == split, ["image_path", "label"]].copy()
    if subset.empty:
        raise ValueError(f"{manifest_path} has no {split} rows")
    if subset["image_path"].duplicated().any() or not subset["image_path"].map(os.path.isfile).all():
        raise ValueError(f"{manifest_path} has duplicate or missing {split} images")
    return subset


def run_training_from_manifest(args):
    cached = {split: manifest_dataframe(args.split_manifest, split) for split in ("train", "validation", "test")}
    train_ids = set(cached["train"]["image_path"])
    val_ids = set(cached["validation"]["image_path"])
    test_ids = set(cached["test"]["image_path"])
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise ValueError("Manifest has within-fold split overlap.")

    def prepare_from_manifest(*_args, **_kwargs):
        return cached["train"].copy(), cached["validation"].copy(), cached["test"].copy()

    # run_training resolves prepare_dataframe from its own module globals.
    source_train.prepare_dataframe = prepare_from_manifest
    return source_train.run_training(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, choices=["resnet18", "resnet50", "densenet121", "mobilenet_v3_large", "efficientnet_b0"])
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--save_root", required=True)
    parser.add_argument("--save_name", required=True)
    # Compatibility-only fields: source run_training() accesses them before
    # calling the manifest-backed replacement prepare_dataframe().
    parser.add_argument("--data_root", default="")
    parser.add_argument("--removal", nargs="*", default=[])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--resize_mode", default="padded", choices=["random", "padded", "zeropad", "reflect"])
    parser.add_argument("--num_runs", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 1 <= args.num_runs <= len(SEEDS):
        raise ValueError("num_runs must be between 1 and 5")
    os.makedirs(args.save_root, exist_ok=True)
    records = []
    for seed in SEEDS[:args.num_runs]:
        current = copy.deepcopy(args)
        current.seed = seed
        current.save_name = f"{args.save_name}_seed{seed}"
        auc, checkpoint_path = run_training_from_manifest(current)
        records.append({"seed": seed, "best_validation_auc": auc, "checkpoint_path": checkpoint_path})
    pd.DataFrame(records).to_csv(os.path.join(args.save_root, "multiseed_training_summary.csv"), index=False)


if __name__ == "__main__":
    main()
