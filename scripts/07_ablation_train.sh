#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_dir RAY_ROOT
require_file REMOVAL_CSV
mkdir -p "${ABLATION_CHECKPOINT_DIR}"

run_condition() {
    local ray="$1"
    local suffix="$2"
    local data_root="${RAY_ROOT}/ray_${ray}/refined_results/masked_images_GOLD_ray${ray}${suffix}"
    [[ -d "${data_root}" ]] || { echo "ERROR: missing masked image root: ${data_root}" >&2; exit 1; }
    save_name="GOLD_ray${ray}${suffix}_densenet121_wd5e3_lr2e5_ep30_hf_thres_padded"
    "${PYTHON_BIN}" ./src/train/run_multi_seed.py \
        --model_name densenet121 --save_name "${save_name}" --save_root "${ABLATION_CHECKPOINT_DIR}" \
        --data_root "${data_root}" --removal "${REMOVAL_CSV}" \
        --img_size 224 --batch_size 32 --epochs 30 --lr 2e-5 --weight_decay 5e-3 \
        --resize_mode padded --num_runs 5
}

for ray in 2 3 4; do
    run_condition "${ray}" ""
    run_condition "${ray}" "_inverted"
done
