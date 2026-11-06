#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_dir DATA_ROOT
require_file REMOVAL_CSV
require_dir YOLO_WEIGHT_ROOT
require_dir MEDSAM_WEIGHT_ROOT
mkdir -p "${SEGMENTATION_OUTPUT_ROOT}/inference"

for ray in 2 3 4; do
    meta_yolo="${YOLO_WEIGHT_ROOT}/metacarpal/ray${ray}/checkpoints/yolov8n_bone_det/weights/best.pt"
    phal_yolo="${YOLO_WEIGHT_ROOT}/phalanx/ray${ray}/checkpoints/yolov8n_bone_det/weights/best.pt"
    meta_sam="${MEDSAM_WEIGHT_ROOT}/metacarpal/crop_seg_medsam_ep25_meta${ray}.pth"
    phal_sam="${MEDSAM_WEIGHT_ROOT}/phalanx/crop_seg_medsam_ep25_phalanx${ray}.pth"
    for weight in "${meta_yolo}" "${phal_yolo}" "${meta_sam}" "${phal_sam}"; do
        [[ -f "${weight}" ]] || { echo "ERROR: missing segmentation weight: ${weight}" >&2; exit 1; }
    done

    meta_root="${SEGMENTATION_OUTPUT_ROOT}/inference/metacarpal/ray${ray}"
    phal_root="${SEGMENTATION_OUTPUT_ROOT}/inference/phalanx/ray${ray}"
    meta_name="metacarpal_ray${ray}"
    phal_name="phalanx_ray${ray}"

    "${PYTHON_BIN}" ./src/segmentation/infer_crop_seg.py \
        --img_dir "${DATA_ROOT}" --root "${meta_root}" --model_type sam \
        --sam_id wanglab/medsam-vit-base --sam_path "${meta_sam}" --yolo_path "${meta_yolo}" \
        --save_name "${meta_name}" --removal_csv "${REMOVAL_CSV}" \
        --fail_removal "${meta_root}/segmentation_failures.csv"
    "${PYTHON_BIN}" ./src/segmentation/infer_crop_seg.py \
        --img_dir "${DATA_ROOT}" --root "${phal_root}" --model_type sam \
        --sam_id wanglab/medsam-vit-base --sam_path "${phal_sam}" --yolo_path "${phal_yolo}" \
        --save_name "${phal_name}" --removal_csv "${REMOVAL_CSV}" \
        --fail_removal "${phal_root}/segmentation_failures.csv"
done

echo "Component-mask inference completed."
echo "STOP: visually review the metacarpal and phalanx masks and manually correct them where required."
echo "Place the reviewed masks under REFINED_COMPONENT_MASK_ROOT as documented in README.md before running 06b."
