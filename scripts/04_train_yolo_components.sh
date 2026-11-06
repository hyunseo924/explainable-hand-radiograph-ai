#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_dir DATA_ROOT
require_dir SEGMENTATION_LABEL_ROOT
require_file REMOVAL_CSV
mkdir -p "${YOLO_WEIGHT_ROOT}"

for ray in 2 3 4; do
    echo "===== YOLO: metacarpal ray ${ray} ====="
    "${PYTHON_BIN}" ./src/segmentation/YOLO.py \
        --root "${YOLO_WEIGHT_ROOT}/metacarpal/ray${ray}" \
        --img_dir "${DATA_ROOT}" \
        --mask_dir "${SEGMENTATION_LABEL_ROOT}/Metacarpal/Metacarpal_${ray}/masks" \
        --mask_csv "${SEGMENTATION_LABEL_ROOT}/Metacarpal/Metacarpal_${ray}/list/masks.csv" \
        --class_name "metacarpal_ray${ray}" \
        --removal_csv "${REMOVAL_CSV}" --epochs 50 --batch 16 --lr 0.01

    echo "===== YOLO: phalanx ray ${ray} ====="
    "${PYTHON_BIN}" ./src/segmentation/YOLO.py \
        --root "${YOLO_WEIGHT_ROOT}/phalanx/ray${ray}" \
        --img_dir "${DATA_ROOT}" \
        --mask_dir "${SEGMENTATION_LABEL_ROOT}/Phalanx/Phalanx_${ray}/masks" \
        --mask_csv "${SEGMENTATION_LABEL_ROOT}/Phalanx/Phalanx_${ray}/list/masks.csv" \
        --class_name "phalanx_ray${ray}" \
        --removal_csv "${REMOVAL_CSV}" --epochs 50 --batch 16 --lr 0.001
done
