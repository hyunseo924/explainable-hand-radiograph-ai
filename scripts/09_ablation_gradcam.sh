#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_dir RAY_ROOT
require_dir ABLATION_CHECKPOINT_DIR
require_file REMOVAL_CSV
require_path TEMPLATE_RELATIVE_PATH

run_gradcam() {
    local ray="$1"
    local condition="$2"
    local suffix="$3"
    local data_root="${RAY_ROOT}/ray_${ray}/refined_results/masked_images_GOLD_ray${ray}${suffix}"
    local template_path="${data_root}/${TEMPLATE_RELATIVE_PATH}"
    local run_name="ray${ray}_${condition}"
    local pattern="best_GOLD_ray${ray}${suffix}_densenet121_wd5e3_lr2e5_ep30_hf_thres_padded_seed*.pth"
    local -a checkpoints=()
    [[ -f "${template_path}" ]] || { echo "ERROR: missing Grad-CAM template: ${template_path}" >&2; exit 1; }
    collect_five_checkpoints "${ABLATION_CHECKPOINT_DIR}" "${pattern}" checkpoints
    "${PYTHON_BIN}" ./src/analysis/cam_all_in_one.py \
        --model_name densenet121 --model_path "${checkpoints[@]}" \
        --save_name "${run_name}_padded" --save_dir ./outputs/ablation_gradcam \
        --data_root "${data_root}" --removal "${REMOVAL_CSV}" --template_path "${template_path}" \
        --resize_mode padded --cam_method gradcam --target_class pred \
        --overlay_alpha 0.50 --percentile_low 75 --percentile_high 99 \
        --per_image_noise_threshold 0.15 \
        --group_average_noise_threshold 0.15 --whole_average_noise_threshold 0.15 \
        --save_success_only --save_raw_cam --save_individual_cams
}

for ray in 2 3 4; do
    run_gradcam "${ray}" selective ""
    run_gradcam "${ray}" selected "_inverted"
done
