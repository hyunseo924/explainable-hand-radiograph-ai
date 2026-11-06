#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_dir DATA_ROOT
require_dir REFINED_COMPONENT_MASK_ROOT
mkdir -p "${RAY_ROOT}"

for ray in 2 3 4; do
    meta_dir="${REFINED_COMPONENT_MASK_ROOT}/metacarpal/ray${ray}"
    phal_dir="${REFINED_COMPONENT_MASK_ROOT}/phalanx/ray${ray}"
    [[ -d "${meta_dir}" ]] || { echo "ERROR: reviewed metacarpal masks not found: ${meta_dir}" >&2; exit 1; }
    [[ -d "${phal_dir}" ]] || { echo "ERROR: reviewed phalanx masks not found: ${phal_dir}" >&2; exit 1; }

    ray_root="${RAY_ROOT}/ray_${ray}"
    "${PYTHON_BIN}" ./src/segmentation/mask_preprocessing/make_ray_mask.py \
        --root "${ray_root}" \
        --meta_dir "${meta_dir}" \
        --phal_dir "${phal_dir}" \
        --save_name "refined_masks_GOLD_ray${ray}"

    mask_dir="${ray_root}/refined_masks_GOLD_ray${ray}"
    result_dir="${ray_root}/refined_results"
    # selective: remove the specified ray; selected: retain only that ray.
    "${PYTHON_BIN}" ./src/segmentation/mask_preprocessing/apply_mask.py \
        --img_dir "${DATA_ROOT}" --mask_dir "${mask_dir}" --save_dir "${result_dir}" \
        --save_name "GOLD_ray${ray}" --morph_size 0 --dont_save_mask
    "${PYTHON_BIN}" ./src/segmentation/mask_preprocessing/apply_mask.py \
        --img_dir "${DATA_ROOT}" --mask_dir "${mask_dir}" --save_dir "${result_dir}" \
        --save_name "GOLD_ray${ray}_inverted" --morph_size 0 --dont_save_mask --invert
done

echo "Reviewed component masks were merged and the selective/selected ray datasets were created."
