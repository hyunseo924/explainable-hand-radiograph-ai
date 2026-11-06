#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_dir DATA_ROOT
require_file REMOVAL_CSV

"${PYTHON_BIN}" ./analysis/export_dataset_split_manifest.py \
    --data_root "${DATA_ROOT}" \
    --removal_csv "${REMOVAL_CSV}" \
    --output_dir ./outputs/fixed_split_manifest
