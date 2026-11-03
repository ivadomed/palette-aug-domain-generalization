#!/usr/bin/env bash
# Low-data texture step table (flat vs texture fill per training-set size, next to the full-data runs).
#   bash 06_21_lowdata_texture_step.sh [mri|ct]      (training contrast, default mri)
# Light (a few CSVs): runs inline. Inputs: metrics of 04_35 runs evaluated with
#   TRAINING_CONTRAST=<c> METRICS_SUBDIR=lowdata bash 06_01_evaluate_run.sh <RUN_ID> auglab
# Output: 8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/<c>/lowdata/lowdata_texture_step_summary.md
set -euo pipefail
C="${1:-mri}"
case "${C}" in mri|ct) ;; *) echo "ERROR: contrast must be mri or ct (got '${C}')" >&2; exit 1 ;; esac
source "$(dirname "$0")/../00_utils/env_mri.sh"
cd "${PROJECT_ROOT}"
.venv/bin/python benchmark/00_commun_scripts/00_04_analysis/lowdata/lowdata_texture_step.py \
    "$(dirname "$0")/configs/lowdata_texture_step_${C}.yaml"
