#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_common.sh"

require_dir MENOPAUSE_RESULTS_ROOT
export MENOPAUSE_RESULTS_ROOT

"${PYTHON_BIN}" ./analysis/bootstrap_ci/bootstrap_seed_mean_ci.py \
    --n-bootstrap 20000 --random-seed 20260927 \
    --output-dir "${MENOPAUSE_RESULTS_ROOT}/seed_mean_stratified_bootstrap_ci"
