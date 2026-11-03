#!/usr/bin/env bash
# Low-data texture step table, chaos (flat vs texture fill per training-set size, next to the full-data rung-4.5 / rung-5 runs).
#   bash 06_38_lowdata_texture_step.sh [t1in|t2spir]      (training contrast, default t1in)
# Held-out pool = the chaos ladders' (06_34 / 06_35) grouped OOD pool: chaos's own held-out contrasts + amos (ct, mri) and
# sliver07 (ct) scored on volumes cropped to the CHAOS field of view BEFORE prediction, pooled by true contrast.
# Inputs: metrics of 04_93 runs, own items via 06_01_evaluate_run.sh with METRICS_SUBDIR=lowdata, amos / sliver07 via
# fov_crop_predict_evaluate.sh with RUNS_FILE metrics_subdir "lowdata" (scripts/cluster/lowdata_vulcan/post_run.sh does both).
# Light (CSVs): runs inline. Output: 8_results_chaos/02_metrics/chaos_model/<c>/lowdata/lowdata_texture_step_summary.md
set -euo pipefail
C="${1:-t1in}"
case "${C}" in t1in|t2spir) ;; *) echo "ERROR: contrast must be t1in or t2spir (got '${C}')" >&2; exit 1 ;; esac
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
.venv/bin/python benchmark/00_commun_scripts/00_04_analysis/lowdata/lowdata_texture_step.py \
    "$(dirname "$0")/configs/lowdata_texture_step_${C}.yaml"
