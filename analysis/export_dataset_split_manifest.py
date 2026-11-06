#!/usr/bin/env python
"""Export the exact fixed split used by training/evaluation without loading a model."""

import argparse
import json
import os
import sys

import pandas as pd


PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE_INFERENCE_DIR = os.environ.get(
    "MENOPAUSE_SOURCE_INFERENCE_DIR", os.path.join(PACKAGE_ROOT, "src", "inference")
)
sys.path.insert(0, SOURCE_INFERENCE_DIR)
from inference import SEED, prepare_dataframe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--removal_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def make_manifest(split_name, dataframe):
    output = dataframe.copy()
    output.insert(0, "split", split_name)
    output.insert(1, "image_id", output["image_path"].map(lambda value: os.path.splitext(os.path.basename(value))[0]))
    output.insert(2, "filename", output["image_path"].map(os.path.basename))
    output["class_group"] = output["label"].map({0: "F_20_45", 1: "F_55_80"})
    return output[["split", "image_id", "filename", "label", "class_group", "image_path"]]


def main():
    args = parse_args()
    train, validation, test = prepare_dataframe(args.data_root, removal_files=[args.removal_csv], split_seed=SEED)
    manifest = pd.concat([
        make_manifest("train", train), make_manifest("validation", validation), make_manifest("test", test)
    ], ignore_index=True).sort_values(["split", "label", "image_id"], kind="stable")
    os.makedirs(args.output_dir, exist_ok=True)
    manifest.to_csv(os.path.join(args.output_dir, "dataset_split_manifest.csv"), index=False)
    summary = manifest.groupby(["split", "label", "class_group"], as_index=False).size().rename(columns={"size": "n_images"})
    summary.to_csv(os.path.join(args.output_dir, "dataset_split_summary.csv"), index=False)
    with open(os.path.join(args.output_dir, "dataset_split_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump({"split_seed": SEED, "sorting": "lexicographic filename sorting within class", "n_images": int(len(manifest))}, handle, indent=2)
    print(f"[OK] Wrote fixed-split manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
