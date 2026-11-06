#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_dir DATA_ROOT
require_file REMOVAL_CSV
require_dir BASELINE_CHECKPOINT_DIR

run_evaluation() {
    local run_name="$1"
    local model_name="$2"
    local pattern="$3"
    local -a checkpoints=()
    collect_five_checkpoints "${BASELINE_CHECKPOINT_DIR}" "${pattern}" checkpoints
    local run_dir="./outputs/baseline_ci_evaluation/${run_name}"
    mkdir -p "${run_dir}"
    "${PYTHON_BIN}" ./src/inference/inference.py \
        --model_name "${model_name}" --model_path "${checkpoints[@]}" \
        --data_root "${DATA_ROOT}" --removal "${REMOVAL_CSV}" --resize_mode padded \
        --region "${run_name}" \
        --output_csv "${run_dir}/evaluation_results.csv" \
        --prediction_csv "${run_dir}/per_image_predictions.csv" \
        --ensemble_score_csv "${run_dir}/ensemble_scores.csv" \
        --conf_csv "${run_dir}/true_class_confidence.csv" \
        --conf_col_name "${run_name}_true_class_confidence"
}

run_evaluation resnet18_primary resnet18 'best_iter5_resnet18_wd5e3_lr2e5_ep30_hf_thres_padded_seed*.pth'
run_evaluation resnet50_primary resnet50 'best_iter5_resnet50_wd5e3_lr2e5_ep30_hf_thres_padded_seed*.pth'
run_evaluation densenet121_primary densenet121 'best_iter5_densenet121_wd5e3_lr2e5_ep30_hf_thres_padded_seed*.pth'
run_evaluation mobilenet_primary_padded mobilenet_v3_large 'best_mobilenet_v3_large_seed*.pth'
run_evaluation efficientnet_primary_padded efficientnet_b0 'best_efficientnet_b0_seed*.pth'
