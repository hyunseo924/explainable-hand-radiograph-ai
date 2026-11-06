#!/usr/bin/env bash
# Shared setup for the executable workflow scripts. Run scripts from any directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PACKAGE_ROOT}"

PATHS_FILE="${PATHS_FILE:-./configs/paths.env}"
if [[ ! -f "${PATHS_FILE}" ]]; then
    echo "ERROR: ${PATHS_FILE} is missing. Copy ./configs/example_paths.env to ./configs/paths.env and edit it." >&2
    exit 1
fi
source "${PATHS_FILE}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

require_path() {
    local name="$1"
    local value="${!name:-}"
    if [[ -z "${value}" ]]; then
        echo "ERROR: ${name} is not set in ${PATHS_FILE}." >&2
        exit 1
    fi
}

require_dir() {
    local name="$1"
    require_path "${name}"
    if [[ ! -d "${!name}" ]]; then
        echo "ERROR: ${name} is not a directory: ${!name}" >&2
        exit 1
    fi
}

require_file() {
    local name="$1"
    require_path "${name}"
    if [[ ! -f "${!name}" ]]; then
        echo "ERROR: ${name} is not a file: ${!name}" >&2
        exit 1
    fi
}

collect_five_checkpoints() {
    local directory="$1"
    local pattern="$2"
    local -n result_ref="$3"
    mapfile -t result_ref < <(find "${directory}" -maxdepth 2 -type f -name "${pattern}" -print | sort)
    if [[ "${#result_ref[@]}" -ne 5 ]]; then
        echo "ERROR: expected five checkpoints matching ${pattern} under ${directory}; found ${#result_ref[@]}." >&2
        printf '  %s\n' "${result_ref[@]:-}" >&2
        exit 1
    fi
}
