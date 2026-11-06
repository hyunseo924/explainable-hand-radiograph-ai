#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_dir DATA_ROOT
require_dir RAY_ROOT
require_file REMOVAL_CSV
export MENOPAUSE_SOURCE_TRAIN_DIR="${PACKAGE_ROOT}/src/train"

one_se_root=./outputs/one_se_checkpoint_selection
seeds=(42 777 2024 1004 8282)

evaluate_selected() {
    local model="$1"
    local condition="$2"
    local data_root="$3"
    local result_prefix="$4"
    for method in max_auc one_se_earliest; do
        mapfile -t checkpoints < <(find "${one_se_root}/checkpoints/${condition}" -type f -path "*/selected/${method}_*.pth" -print | sort)
        [[ "${#checkpoints[@]}" -eq 5 ]] || { echo "ERROR: ${condition}/${method}: expected five selected checkpoints." >&2; exit 1; }
        result_dir="${one_se_root}/evaluation/${result_prefix}/${condition}/${method}"
        mkdir -p "${result_dir}"
        "${PYTHON_BIN}" ./src/inference/inference.py \
            --model_name "${model}" --model_path "${checkpoints[@]}" \
            --data_root "${data_root}" --removal "${REMOVAL_CSV}" --resize_mode padded \
            --region "${condition}_${method}" \
            --output_csv "${result_dir}/evaluation_results.csv" \
            --prediction_csv "${result_dir}/per_image_predictions.csv" \
            --ensemble_score_csv "${result_dir}/ensemble_scores.csv" \
            --conf_csv "${result_dir}/true_class_confidence.csv" \
            --conf_col_name "${condition}_${method}_true_class_confidence"
    done
}

for model in resnet18 resnet50 densenet121 mobilenet_v3_large efficientnet_b0; do
    condition="baseline_${model}"
    for seed in "${seeds[@]}"; do
        "${PYTHON_BIN}" ./analysis/one_se/train_one_se.py \
            --model_name "${model}" --save_root "${one_se_root}/checkpoints/${condition}" \
            --save_name "${condition}_wd5e3_lr2e5_ep30_hf_thres_padded_seed${seed}" \
            --data_root "${DATA_ROOT}" --removal "${REMOVAL_CSV}" --resize_mode padded \
            --img_size 224 --batch_size 32 --epochs 30 --lr 2e-5 --weight_decay 5e-3 \
            --seed "${seed}" --split_seed 42
    done
    evaluate_selected "${model}" "${condition}" "${DATA_ROOT}" baseline
done

for ray in 2 3 4; do
    for condition_suffix in selective selected; do
        if [[ "${condition_suffix}" == selective ]]; then
            image_suffix=""
        else
            image_suffix="_inverted"
        fi
        condition="ray${ray}_${condition_suffix}"
        data_root="${RAY_ROOT}/ray_${ray}/refined_results/masked_images_GOLD_ray${ray}${image_suffix}"
        [[ -d "${data_root}" ]] || { echo "ERROR: missing masked image root: ${data_root}" >&2; exit 1; }
        for seed in "${seeds[@]}"; do
            "${PYTHON_BIN}" ./analysis/one_se/train_one_se.py \
                --model_name densenet121 --save_root "${one_se_root}/checkpoints/${condition}" \
                --save_name "${condition}_densenet121_wd5e3_lr2e5_ep30_hf_thres_padded_seed${seed}" \
                --data_root "${data_root}" --removal "${REMOVAL_CSV}" --resize_mode padded \
                --img_size 224 --batch_size 32 --epochs 30 --lr 2e-5 --weight_decay 5e-3 \
                --seed "${seed}" --split_seed 42
        done
        evaluate_selected densenet121 "${condition}" "${data_root}" ablation
    done
done
