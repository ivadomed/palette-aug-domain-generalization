#!/usr/bin/env bash
# Low-data texture step table, pansegdata (flat vs texture fill per training-set size, next to the full-data rung-4.5 / rung-5 runs).
#   bash 06_21_lowdata_texture_step.sh [t1wce|t2w]      (training contrast, default t1wce)
# Held-out pool = the ladders' (06_10 / 06_11) grouped OOD pool: own held-out contrast + totalsegmri-pancreas (t2like, t1gre)
# and msd-pancreas (ct), pooled by true contrast. Inputs: metrics of 04_45 runs evaluated with METRICS_SUBDIR=lowdata (own:
# 06_01_evaluate_run.sh; companions: their 06_01_evaluate_run.sh), as scripts/cluster/lowdata_vulcan/post_run.sh does.
# Light (CSVs): runs inline. Output: 8_results_pansegdata/02_metrics/pansegdata_model/<c>/lowdata/lowdata_texture_step_summary.md
set -euo pipefail
C="${1:-t1wce}"
case "${C}" in t1wce|t2w) ;; *) echo "ERROR: contrast must be t1wce or t2w (got '${C}')" >&2; exit 1 ;; esac
source "$(dirname "$0")/../00_utils/env.sh"
cd "${PROJECT_ROOT}"
.venv/bin/python benchmark/00_commun_scripts/00_04_analysis/lowdata/lowdata_texture_step.py \
    "$(dirname "$0")/configs/lowdata_texture_step_${C}.yaml"
