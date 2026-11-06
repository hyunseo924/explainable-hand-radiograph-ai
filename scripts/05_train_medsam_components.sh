#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_dir DATA_ROOT
require_dir SEGMENTATION_LABEL_ROOT
mkdir -p "${MEDSAM_WEIGHT_ROOT}/metacarpal" "${MEDSAM_WEIGHT_ROOT}/phalanx"

for ray in 2 3 4; do
    echo "===== MedSAM: metacarpal ray ${ray} ====="
    "${PYTHON_BIN}" ./src/segmentation/train_crop_seg.py \
        --img_dir "${DATA_ROOT}" \
        --mask_dir "${SEGMENTATION_LABEL_ROOT}/Metacarpal/Metacarpal_${ray}/masks" \
        --mask_csv "${SEGMENTATION_LABEL_ROOT}/Metacarpal/Metacarpal_${ray}/list/masks.csv" \
        --model_type sam --sam_id wanglab/medsam-vit-base --epoch 25 \
        --save_dir "${MEDSAM_WEIGHT_ROOT}/metacarpal" \
        --save_name "crop_seg_medsam_ep25_meta${ray}"

    echo "===== MedSAM: phalanx ray ${ray} ====="
    "${PYTHON_BIN}" ./src/segmentation/train_crop_seg.py \
        --img_dir "${DATA_ROOT}" \
        --mask_dir "${SEGMENTATION_LABEL_ROOT}/Phalanx/Phalanx_${ray}/masks" \
        --mask_csv "${SEGMENTATION_LABEL_ROOT}/Phalanx/Phalanx_${ray}/list/masks.csv" \
        --model_type sam --sam_id wanglab/medsam-vit-base --epoch 25 \
        --save_dir "${MEDSAM_WEIGHT_ROOT}/phalanx" \
        --save_name "crop_seg_medsam_ep25_phalanx${ray}"
done
