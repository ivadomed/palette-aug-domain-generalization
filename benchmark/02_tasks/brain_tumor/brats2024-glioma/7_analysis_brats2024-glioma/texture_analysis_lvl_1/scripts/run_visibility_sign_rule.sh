#!/usr/bin/env bash
# Vulcan launcher for visibility_sign_rule.py (FINDINGS 21-25, 47). One CPU job.
#   bash run_visibility_sign_rule.sh score     # rung 4 lblvor -> 5, 4.5 flat -> 5 (texture), 4 lblvor -> 4.5 (noise removal), in that order
#   bash run_visibility_sign_rule.sh extract   # visibility features for keys without a vis_label_*.csv
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.."
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
MODE="${1:?score|extract}"
A="benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1"
PY="${A}/scripts/visibility_sign_rule.py"; LOGS="${SCRATCH}/visibility_sign_rule/logs"; mkdir -p "${LOGS}"
ENVS="export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2; cd '${PROJECT_ROOT}'"
TS="$(date +%Y%m%d_%H%M%S)"
case "${MODE}" in
  score)   run_job --name vis_rule_score --gpus 0 --cpus 1 --mem 6G --time 00:15:00 --log "${LOGS}/score_${TS}.log" -- bash -c "${ENVS};
             RUNG4=lblvor .venv/bin/python ${PY} score && RUNG4=flatfill .venv/bin/python ${PY} score && RUNG4=lblvor REAL_ARM=flatfill .venv/bin/python ${PY} score" ;;
  extract) run_job --name vis_rule_extract --gpus 0 --cpus 8 --mem 48G --time 02:00:00 --log "${LOGS}/extract_${TS}.log" -- bash -c "${ENVS}; .venv/bin/python ${PY} extract" ;;
  *) echo "unknown mode ${MODE}" >&2; exit 1 ;;
esac
echo "log: ${LOGS}/${MODE}_${TS}.log"
