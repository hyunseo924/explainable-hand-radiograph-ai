#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_dir DATA_ROOT
require_dir RAY_ROOT
require_file REMOVAL_CSV
export MENOPAUSE_SOURCE_TRAIN_DIR="${PACKAGE_ROOT}/src/train"
export MENOPAUSE_SOURCE_INFERENCE_DIR="${PACKAGE_ROOT}/src/inference"

manifest_root=./outputs/kfold/manifests
baseline_checkpoint_root=./outputs/kfold/baseline/checkpoints
baseline_evaluation_root=./outputs/kfold/baseline/evaluation
ablation_checkpoint_root=./outputs/kfold/ablation/checkpoints
ablation_evaluation_root=./outputs/kfold/ablation/evaluation

"${PYTHON_BIN}" ./analysis/kfold/kfold_manifest.py \
    --baseline_root "${DATA_ROOT}" --ray_root "${RAY_ROOT}" \
    --removal_csv "${REMOVAL_CSV}" --output_dir "${manifest_root}"

for fold in 1 2 3 4 5 6 7; do
    for model in resnet18 resnet50 densenet121 mobilenet_v3_large efficientnet_b0; do
        checkpoint_dir="${baseline_checkpoint_root}/${model}/fold_${fold}"
        mkdir -p "${checkpoint_dir}"
        "${PYTHON_BIN}" ./analysis/kfold/kfold_multiseed.py \
            --model_name "${model}" --split_manifest "${manifest_root}/baseline/fold_${fold}_manifest.csv" \
            --save_root "${checkpoint_dir}" \
            --save_name "k7_fold${fold}_${model}_wd5e3_lr2e5_ep30_hf_thres_padded" \
            --resize_mode padded --epochs 30 --lr 2e-5 --weight_decay 5e-3 --num_runs 5
        mapfile -t checkpoints < <(find "${checkpoint_dir}" -maxdepth 1 -type f -name '*.pth' -print | sort)
        [[ "${#checkpoints[@]}" -eq 5 ]] || { echo "ERROR: ${model} fold ${fold}: expected five checkpoints." >&2; exit 1; }
        "${PYTHON_BIN}" ./analysis/kfold/kfold_evaluate.py \
            --model_name "${model}" --model_path "${checkpoints[@]}" \
            --split_manifest "${manifest_root}/baseline/fold_${fold}_manifest.csv" --resize_mode padded \
            --region "baseline_${model}_fold${fold}" \
            --output_dir "${baseline_evaluation_root}/${model}/fold_${fold}"
    done
done

for condition in ray2_selective ray2_selected ray3_selective ray3_selected ray4_selective ray4_selected; do
    for fold in 1 2 3 4 5 6 7; do
        checkpoint_dir="${ablation_checkpoint_root}/${condition}/fold_${fold}"
        mkdir -p "${checkpoint_dir}"
        "${PYTHON_BIN}" ./analysis/kfold/kfold_multiseed.py \
            --model_name densenet121 --split_manifest "${manifest_root}/${condition}/fold_${fold}_manifest.csv" \
            --save_root "${checkpoint_dir}" \
            --save_name "k7_fold${fold}_${condition}_densenet121_wd5e3_lr2e5_ep30_hf_thres_padded" \
            --resize_mode padded --epochs 30 --lr 2e-5 --weight_decay 5e-3 --num_runs 5
        mapfile -t checkpoints < <(find "${checkpoint_dir}" -maxdepth 1 -type f -name '*.pth' -print | sort)
        [[ "${#checkpoints[@]}" -eq 5 ]] || { echo "ERROR: ${condition} fold ${fold}: expected five checkpoints." >&2; exit 1; }
        "${PYTHON_BIN}" ./analysis/kfold/kfold_evaluate.py \
            --model_name densenet121 --model_path "${checkpoints[@]}" \
            --split_manifest "${manifest_root}/${condition}/fold_${fold}_manifest.csv" --resize_mode padded \
            --region "ablation_${condition}_fold${fold}" \
            --output_dir "${ablation_evaluation_root}/${condition}/fold_${fold}"
    done
done

"${PYTHON_BIN}" ./analysis/kfold/summarize_kfold.py \
    --kfold-root ./outputs/kfold \
    --output-dir ./outputs/kfold/summary \
    --n-folds 7 --n-seeds 5
