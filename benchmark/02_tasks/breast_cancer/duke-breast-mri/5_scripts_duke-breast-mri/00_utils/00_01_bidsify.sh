#!/bin/bash
# Build 1_BIDS_duke-breast-mri/onc-breast-dukebreastmri from the MAMA-MIA download (CPU job via run_job).
#   bash 5_scripts_duke-breast-mri/00_utils/00_01_bidsify.sh
# DUKE_RAW_DIR overrides the source (default $SCRATCH/duke_download/raw).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/env.sh"
LOG_DIR="${DATASET_ROOT}/1_BIDS_duke-breast-mri/_not_in_annex/bidsify_logs"; mkdir -p "$LOG_DIR"
export DUKE_RAW_DIR="${DUKE_RAW_DIR:-$SCRATCH/duke_download/raw}"
run_job --name duke_bidsify --gpus 0 --cpus 2 --mem 16G --time 03:00:00 \
    --log "${LOG_DIR}/bidsify_$(date +%Y%m%d_%H%M%S).log" -- \
    bash -c "module load python/3.11 >/dev/null 2>&1 || true; DUKE_RAW_DIR='${DUKE_RAW_DIR}' '${PROJECT_ROOT:-${DATASET_ROOT}/../../../..}/.venv/bin/python' '${HERE}/00_01_bidsify.py'"
