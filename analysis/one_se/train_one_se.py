#!/usr/bin/env python
"""Train one seed using the original training mechanics with 1-SE checkpoint selection.

The source project's train/run_train.py is intentionally not modified.  This
module imports its model, dataset, augmentation, split, loss, and seed helpers
and differs only by retaining model-only epoch snapshots long enough to apply a
validation-AUC one-standard-error selection rule.
"""

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

PACKAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SOURCE_TRAIN_DIR = os.environ.get(
    "MENOPAUSE_SOURCE_TRAIN_DIR", os.path.join(PACKAGE_ROOT, "src", "train")
)
sys.path.insert(0, SOURCE_TRAIN_DIR)
import run_train as source_train


def delong_auc_se(labels, scores):
    """Unpaired DeLong standard error for a binary AUC, including ties."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if len(positives) < 2 or len(negatives) < 2:
        raise ValueError("DeLong AUC SE requires at least two samples per class.")
    comparisons = (positives[:, None] > negatives[None, :]).astype(float)
    comparisons += 0.5 * (positives[:, None] == negatives[None, :])
    v10 = comparisons.mean(axis=1)
    v01 = comparisons.mean(axis=0)
    variance = np.var(v10, ddof=1) / len(positives) + np.var(v01, ddof=1) / len(negatives)
    return float(np.sqrt(max(variance, 0.0)))


def state_dict_on_cpu(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def parse_args():
    parser = argparse.ArgumentParser(description="1-SE AUC checkpoint-selection training")
    parser.add_argument("--model_name", required=True, choices=[
        "resnet18", "resnet50", "densenet121", "mobilenet_v3_large", "efficientnet_b0"
    ])
    parser.add_argument("--save_root", required=True)
    parser.add_argument("--save_name", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--removal", nargs="+", required=True)
    parser.add_argument("--resize_mode", default="padded", choices=["random", "padded", "zeropad", "reflect"])
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--keep_epoch_snapshots", action="store_true",
                        help="Keep temporary all-epoch model-state snapshots after selection.")
    return parser.parse_args()


def evaluate_validation(model, loader, criterion, device):
    """Same validation mechanics as source train/run_train.py, plus AUC SE."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    probs = np.asarray(all_probs, dtype=float)
    labels = np.asarray(all_labels, dtype=int)
    best_acc, best_threshold = 0.0, 0.5
    for threshold in np.arange(0.1, 0.91, 0.01):
        accuracy = np.mean((probs >= threshold).astype(int) == labels)
        if accuracy > best_acc:
            best_acc, best_threshold = float(accuracy), float(threshold)
    auc = float(roc_auc_score(labels, probs))
    return {
        "val_loss": running_loss / len(loader.dataset),
        "val_acc": best_acc,
        "val_auc": auc,
        "val_auc_se_delong": delong_auc_se(labels, probs),
        "threshold": best_threshold,
    }


def main():
    args = parse_args()
    source_train.seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    run_root = os.path.join(args.save_root, args.save_name)
    snapshot_dir = os.path.join(run_root, "epoch_snapshots")
    selected_root = os.path.join(run_root, "selected")
    os.makedirs(snapshot_dir, exist_ok=True)
    os.makedirs(selected_root, exist_ok=True)

    train_df, val_df, _ = source_train.prepare_dataframe(
        args.data_root, args.split_seed, removal_files=args.removal
    )
    train_loader = DataLoader(
        source_train.HandXrayDataset(train_df, args.img_size, args.resize_mode, train=True),
        batch_size=args.batch_size, shuffle=True, num_workers=4,
    )
    val_loader = DataLoader(
        source_train.HandXrayDataset(val_df, args.img_size, args.resize_mode, train=False),
        batch_size=args.batch_size, shuffle=False, num_workers=4,
    )
    model = source_train.get_model(args.model_name).to(device)
    # Original manuscript runs used the wrapper default: CrossEntropyLoss.
    # Keep it fixed here so the 1-SE analysis cannot accidentally diverge.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-8)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Seed {args.seed} - Epoch {epoch}/{args.epochs}", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            predicted = torch.max(outputs, 1)[1]
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loop.set_postfix(loss=loss.item())

        validation = evaluate_validation(model, val_loader, criterion, device)
        scheduler.step()
        record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader.dataset),
            "train_acc": correct / total,
            **validation,
        }
        snapshot_path = os.path.join(snapshot_dir, f"epoch_{epoch:03d}.pth")
        torch.save({
            "model_state_dict": state_dict_on_cpu(model),
            "epoch": epoch,
            "val_auc": validation["val_auc"],
            "val_auc_se_delong": validation["val_auc_se_delong"],
            "threshold": validation["threshold"],
            "selection_method": "pending",
        }, snapshot_path)
        record["snapshot_path"] = snapshot_path
        history.append(record)
        print(
            f"Epoch {epoch:02d}: val_auc={validation['val_auc']:.6f}, "
            f"DeLong_SE={validation['val_auc_se_delong']:.6f}, threshold={validation['threshold']:.2f}"
        )

    history_df = pd.DataFrame(history)
    maximum_row = history_df.loc[history_df["val_auc"].idxmax()]
    max_auc = float(maximum_row["val_auc"])
    max_auc_se = float(maximum_row["val_auc_se_delong"])
    one_se_lower_bound = max_auc - max_auc_se
    eligible = history_df.loc[history_df["val_auc"] >= one_se_lower_bound - 1e-12].sort_values("epoch")
    one_se_row = eligible.iloc[0]

    selection_rows = []
    for method, row in (("max_auc", maximum_row), ("one_se_earliest", one_se_row)):
        source_path = row["snapshot_path"]
        destination = os.path.join(selected_root, f"{method}_{args.save_name}.pth")
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        checkpoint.update({
            "selection_method": method,
            "max_val_auc": max_auc,
            "max_val_auc_delong_se": max_auc_se,
            "one_se_lower_bound": one_se_lower_bound,
            "selected_epoch": int(row["epoch"]),
            "selected_val_auc": float(row["val_auc"]),
        })
        torch.save(checkpoint, destination)
        selection_rows.append({
            "selection_method": method,
            "selected_epoch": int(row["epoch"]),
            "selected_val_auc": float(row["val_auc"]),
            "max_val_auc": max_auc,
            "max_val_auc_delong_se": max_auc_se,
            "one_se_lower_bound": one_se_lower_bound,
            "selected_checkpoint": destination,
        })

    history_df["max_val_auc"] = max_auc
    history_df["max_val_auc_delong_se"] = max_auc_se
    history_df["one_se_lower_bound"] = one_se_lower_bound
    history_df["one_se_eligible"] = history_df["val_auc"] >= one_se_lower_bound - 1e-12
    history_df.to_csv(os.path.join(run_root, "epoch_validation_history.csv"), index=False)
    pd.DataFrame(selection_rows).to_csv(os.path.join(run_root, "selection_summary.csv"), index=False)

    if not args.keep_epoch_snapshots:
        shutil.rmtree(snapshot_dir)
    print(pd.DataFrame(selection_rows).to_string(index=False))
    print(f"[OK] 1-SE selection complete: {run_root}")


if __name__ == "__main__":
    main()
