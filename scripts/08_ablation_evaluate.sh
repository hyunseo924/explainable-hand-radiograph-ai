#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_dir RAY_ROOT
require_dir ABLATION_CHECKPOINT_DIR
require_file REMOVAL_CSV

run_evaluation() {
    local ray="$1"
    local condition="$2"
    local suffix="$3"
    local data_root="${RAY_ROOT}/ray_${ray}/refined_results/masked_images_GOLD_ray${ray}${suffix}"
    local run_name="ray${ray}_${condition}"
    local pattern="best_GOLD_ray${ray}${suffix}_densenet121_wd5e3_lr2e5_ep30_hf_thres_padded_seed*.pth"
    local -a checkpoints=()
    [[ -d "${data_root}" ]] || { echo "ERROR: missing masked image root: ${data_root}" >&2; exit 1; }
    collect_five_checkpoints "${ABLATION_CHECKPOINT_DIR}" "${pattern}" checkpoints
    run_dir="./outputs/ablation_ci_evaluation/${run_name}"
    mkdir -p "${run_dir}"
    "${PYTHON_BIN}" ./src/inference/inference.py \
        --model_name densenet121 --model_path "${checkpoints[@]}" \
        --data_root "${data_root}" --removal "${REMOVAL_CSV}" --resize_mode padded --region "${run_name}" \
        --output_csv "${run_dir}/evaluation_results.csv" \
        --prediction_csv "${run_dir}/per_image_predictions.csv" \
        --ensemble_score_csv "${run_dir}/ensemble_scores.csv" \
        --conf_csv "${run_dir}/true_class_confidence.csv" \
        --conf_col_name "${run_name}_true_class_confidence"
}

for ray in 2 3 4; do
    run_evaluation "${ray}" selective ""
    run_evaluation "${ray}" selected "_inverted"
done
