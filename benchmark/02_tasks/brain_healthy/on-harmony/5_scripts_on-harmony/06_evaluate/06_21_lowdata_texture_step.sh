#!/usr/bin/env bash
# Low-data texture step table, on-harmony (flat vs texture fill per training-set size, next to the full-data rung-4.5 /
# rung-5 runs). Held-out contrasts = the ladders' (06_10 / 06_11 / 06_12 full pool): every test contrast except the training one; per-image
# case ids reduced to sessions as in the ladder engine (_patient_key strips the contrast suffix).
#   bash 06_21_lowdata_texture_step.sh [T1w|T2w|dwi_ap]      (training contrast, default T1w)
# Inputs: metrics of 04_91 runs evaluated with METRICS_SUBDIR=lowdata bash 06_01_evaluate_testset.sh <RUN_ID> <FOLD>
# (scripts/cluster/lowdata_vulcan/post_run.sh). Light (CSVs): runs inline.
# Output: 8_results_on-harmony/02_metrics/on_harmony_model/<c>/lowdata/lowdata_texture_step_summary.md
set -euo pipefail
C="${1:-T1w}"
case "${C}" in T1w|T2w|dwi_ap) ;; *) echo "ERROR: contrast must be T1w, T2w or dwi_ap (got '${C}')" >&2; exit 1 ;; esac
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
.venv/bin/python benchmark/00_commun_scripts/00_04_analysis/lowdata/lowdata_texture_step.py \
    "$(dirname "$0")/configs/lowdata_texture_step_${C}.yaml"
