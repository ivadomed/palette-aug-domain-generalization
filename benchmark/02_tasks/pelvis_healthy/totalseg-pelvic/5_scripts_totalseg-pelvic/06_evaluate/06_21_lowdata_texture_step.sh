#!/usr/bin/env bash
# Low-data texture step table (flat vs texture fill per training-set size, next to the full-data runs), MRI-trained.
# Light (a few CSVs): runs inline. Inputs: metrics of 04_35 runs evaluated with
#   TRAINING_CONTRAST=mri METRICS_SUBDIR=lowdata bash 06_01_evaluate_run.sh <RUN_ID> auglab
# Output: 8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/mri/lowdata/lowdata_texture_step_summary.md
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_mri.sh"
cd "${PROJECT_ROOT}"
.venv/bin/python benchmark/00_commun_scripts/00_04_analysis/lowdata/lowdata_texture_step.py \
    "$(dirname "$0")/configs/lowdata_texture_step_mri.yaml"
