"""Create reproducible 7-fold manifests without modifying the source project."""

import argparse
import glob
import json
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


SEED = 42
N_SPLITS = 7
# Outer test is 1/7 (=14.3%).  This gives validation 15% of the total dataset
# and training 70% when 17.5% of each outer-training set is held out for val.
INNER_VALIDATION_FRACTION = 0.175


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_root", required=True,
                        help="Baseline image root containing F_20_45/ and F_55_80/.")
    parser.add_argument("--ray_root", required=True,
                        help="Root containing ray_2/, ray_3/, and ray_4/ refined_results directories.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--removal_csv", required=True)
    return parser.parse_args()


def experiment_roots(baseline_root, ray_root):
    roots = {"baseline": baseline_root}
    for ray in (2, 3, 4):
        root = os.path.join(ray_root, f"ray_{ray}", "refined_results")
        roots[f"ray{ray}_selective"] = os.path.join(root, f"masked_images_GOLD_ray{ray}")
        roots[f"ray{ray}_selected"] = os.path.join(root, f"masked_images_GOLD_ray{ray}_inverted")
    return roots


def dataframe_from_root(root, removal_ids):
    records = []
    for group, label in (("F_20_45", 0), ("F_55_80", 1)):
        paths = sorted(glob.glob(os.path.join(root, group, "*.*")))
        for path in paths:
            if not path.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            filename = os.path.basename(path)
            if filename in removal_ids:
                continue
            records.append({
                "image_id": os.path.splitext(filename)[0],
                "filename": filename,
                "label": label,
                "class_group": group,
                "image_path": os.path.abspath(path),
            })
    return pd.DataFrame(records).sort_values(["label", "image_id"], kind="stable").reset_index(drop=True)


def make_assignments(master):
    assignments = []
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (outer_train_idx, outer_test_idx) in enumerate(splitter.split(master, master["label"]), start=1):
        outer_train = master.iloc[outer_train_idx]
        outer_test = master.iloc[outer_test_idx]
        train_idx, validation_idx = train_test_split(
            outer_train.index,
            test_size=INNER_VALIDATION_FRACTION,
            stratify=outer_train["label"],
            random_state=SEED + fold,
        )
        for split, indices in (("train", train_idx), ("validation", validation_idx), ("test", outer_test.index)):
            split_df = master.loc[indices, ["image_id", "label"]].copy()
            split_df.insert(0, "split", split)
            split_df.insert(0, "fold", fold)
            assignments.append(split_df)
    assignments = pd.concat(assignments, ignore_index=True)
    return assignments.sort_values(["fold", "split", "label", "image_id"], kind="stable").reset_index(drop=True)


def validate_assignments(assignments, n_images):
    test_counts = assignments.loc[assignments["split"] == "test", "image_id"].value_counts()
    if len(test_counts) != n_images or not (test_counts == 1).all():
        raise RuntimeError("Outer test folds are not a disjoint complete partition.")
    for fold, fold_df in assignments.groupby("fold"):
        groups = {name: set(part["image_id"]) for name, part in fold_df.groupby("split")}
        if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
            raise RuntimeError(f"Fold {fold} contains overlap between train/validation/test.")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    removal_ids = set(pd.read_csv(args.removal_csv)["id"].astype(str))
    roots = experiment_roots(args.baseline_root, args.ray_root)
    master = dataframe_from_root(roots["baseline"], removal_ids)
    if master.empty:
        raise RuntimeError("No baseline images found.")
    assignments = make_assignments(master)
    validate_assignments(assignments, len(master))

    summary_rows = []
    for experiment, root in roots.items():
        experiment_df = dataframe_from_root(root, removal_ids)
        if set(experiment_df["image_id"]) != set(master["image_id"]):
            missing = sorted(set(master["image_id"]) - set(experiment_df["image_id"]))
            extra = sorted(set(experiment_df["image_id"]) - set(master["image_id"]))
            raise RuntimeError(f"{experiment} does not match baseline image IDs (missing={len(missing)}, extra={len(extra)}).")
        merged = assignments.merge(experiment_df, on=["image_id", "label"], how="inner", validate="many_to_one")
        if len(merged) != len(assignments):
            raise RuntimeError(f"{experiment} manifest merge lost rows.")
        experiment_dir = os.path.join(args.output_dir, experiment)
        os.makedirs(experiment_dir, exist_ok=True)
        for fold in range(1, N_SPLITS + 1):
            fold_df = merged.loc[merged["fold"] == fold].sort_values(["split", "label", "image_id"], kind="stable")
            fold_df.to_csv(os.path.join(experiment_dir, f"fold_{fold}_manifest.csv"), index=False)
            for (split, label), part in fold_df.groupby(["split", "label"]):
                summary_rows.append({"experiment": experiment, "fold": fold, "split": split, "label": label, "n_images": len(part)})
        merged.to_csv(os.path.join(experiment_dir, "k7_all_folds_manifest.csv"), index=False)

    summary = pd.DataFrame(summary_rows).sort_values(["experiment", "fold", "split", "label"])
    summary.to_csv(os.path.join(args.output_dir, "k7_split_summary.csv"), index=False)
    metadata = {
        "n_splits": N_SPLITS,
        "outer_split": "StratifiedKFold(shuffle=True, random_state=42)",
        "outer_test_fraction": "1/7 (approximately 14.3%)",
        "inner_split": "stratified validation split from outer training data",
        "inner_validation_fraction_of_outer_train": INNER_VALIDATION_FRACTION,
        "approximate_total_ratios": {"train": "70%", "validation": "15%", "test": "14.3%"},
        "test_fold_overlap": "none; every eligible image occurs in exactly one outer test fold",
        "within_fold_overlap": "none between train, validation, and test",
        "important": "Training sets necessarily overlap across folds in standard k-fold cross-validation; only outer test folds are mutually disjoint.",
        "removal_csv": os.path.abspath(args.removal_csv),
        "eligible_images": int(len(master)),
    }
    with open(os.path.join(args.output_dir, "k7_split_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[OK] Created manifests for {len(roots)} experiments in {args.output_dir}")
    print(summary.groupby(["experiment", "split"])["n_images"].sum())


if __name__ == "__main__":
    main()
