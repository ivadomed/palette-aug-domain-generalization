#!/usr/bin/env bash
# FOV/geometry audit: this companion's test items vs the SOURCE's TRAINING images (shared 00_00_utils/fov_audit.py), via run_job (CPU).
# Run BEFORE choosing a crop and again after cropping. MISMATCH lines mean: crop/resample to the training geometry before predicting.
#   bash 02_02_fov_audit.sh [SOURCE_CONTRAST=t1wce] [ITEM_DIR_SUFFIX=]        (ITEM_DIR_SUFFIX e.g. _crop to audit cropped items)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../00_utils/env.sh"; cd "${PROJECT_ROOT}"
SC="${1:-t1wce}"; SUF="${2:-}"
TRAIN_DIR="$(ls -d "${PANSEG_NNUNET_RAW}"/Dataset*/imagesTr | head -1)"   # TODO if the source has one Dataset per contrast: pick the one trained on ${SC}
mkdir -p "${RESULTS_DIR}/_logs"
run_job --name amos-pancreas_fov_audit --gpus 0 --cpus 2 --mem 8G --time 00:20:00 --log "${RESULTS_DIR}/_logs/fov_audit_$(date +%Y%m%d_%H%M%S).log" --wait -- \
    .venv/bin/python benchmark/00_commun_scripts/00_00_utils/fov_audit.py --train-dir "${TRAIN_DIR}" --test-dir "${nnUNet_raw}/imagesTs_mixed_t1${SUF}"
tail -25 "$(ls -t "${RESULTS_DIR}"/_logs/fov_audit_*.log | head -1)"
