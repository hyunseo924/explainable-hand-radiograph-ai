#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_dir DATA_ROOT
require_file REMOVAL_CSV
mkdir -p "${BASELINE_CHECKPOINT_DIR}"

for model in resnet18 resnet50 densenet121 mobilenet_v3_large efficientnet_b0; do
    save_name="iter5_${model}_wd5e3_lr2e5_ep30_hf_thres_padded"
    echo "===== baseline training: ${model} ====="
    "${PYTHON_BIN}" ./src/train/run_multi_seed.py \
        --model_name "${model}" \
        --save_name "${save_name}" \
        --save_root "${BASELINE_CHECKPOINT_DIR}" \
        --data_root "${DATA_ROOT}" \
        --removal "${REMOVAL_CSV}" \
        --img_size 224 --batch_size 32 --epochs 30 \
        --lr 2e-5 --weight_decay 5e-3 \
        --resize_mode padded --num_runs 5
done
