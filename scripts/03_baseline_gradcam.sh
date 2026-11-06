#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_dir DATA_ROOT
require_file REMOVAL_CSV
require_dir BASELINE_CHECKPOINT_DIR
require_path TEMPLATE_RELATIVE_PATH
template_path="${DATA_ROOT}/${TEMPLATE_RELATIVE_PATH}"
if [[ ! -f "${template_path}" ]]; then
    echo "ERROR: Grad-CAM template does not exist: ${template_path}" >&2
    exit 1
fi

run_gradcam() {
    local save_name="$1"
    local model_name="$2"
    local pattern="$3"
    local -a checkpoints=()
    collect_five_checkpoints "${BASELINE_CHECKPOINT_DIR}" "${pattern}" checkpoints
    "${PYTHON_BIN}" ./src/analysis/cam_all_in_one.py \
        --model_name "${model_name}" --model_path "${checkpoints[@]}" \
        --save_name "${save_name}" --save_dir ./outputs/baseline_gradcam \
        --data_root "${DATA_ROOT}" --removal "${REMOVAL_CSV}" \
        --template_path "${template_path}" --resize_mode padded --cam_method gradcam --target_class pred \
        --overlay_alpha 0.50 --percentile_low 75 --percentile_high 99 \
        --per_image_noise_threshold 0.15 \
        --group_average_noise_threshold 0.15 --whole_average_noise_threshold 0.15 \
        --save_success_only --save_raw_cam --save_individual_cams
}

run_gradcam resnet18_primary_padded resnet18 'best_iter5_resnet18_wd5e3_lr2e5_ep30_hf_thres_padded_seed*.pth'
run_gradcam resnet50_primary_padded resnet50 'best_iter5_resnet50_wd5e3_lr2e5_ep30_hf_thres_padded_seed*.pth'
run_gradcam densenet121_primary_padded densenet121 'best_iter5_densenet121_wd5e3_lr2e5_ep30_hf_thres_padded_seed*.pth'
run_gradcam mobilenet_primary_padded mobilenet_v3_large 'best_mobilenet_v3_large_seed*.pth'
run_gradcam efficientnet_primary_padded efficientnet_b0 'best_efficientnet_b0_seed*.pth'
