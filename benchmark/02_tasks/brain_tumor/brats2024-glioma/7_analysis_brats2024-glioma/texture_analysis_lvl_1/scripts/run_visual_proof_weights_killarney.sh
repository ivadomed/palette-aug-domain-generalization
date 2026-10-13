#!/usr/bin/env bash
# Killarney launcher for visual_proof_weights.py (flat- vs real-fill glioma model weights / representations). Same env pattern as
# run_visual_proof_killarney.sh. Short jobs (<= 30 min). Per-case cache on $SCRATCH (resumable: existing cases are skipped).
#   bash run_visual_proof_weights_killarney.sh weights               # CPU: A1 filter spectra + layer ranks
#   bash run_visual_proof_weights_killarney.sh acts t2f 0:8          # GPU: A2 CKA + A3 selectivity, test contrast t2f, cases 0-7 of the fixed 24
#   bash run_visual_proof_weights_killarney.sh report                # CPU: tables + figures
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.."
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
export RUN_JOB_GPU_TYPE="${RUN_JOB_GPU_TYPE:-l40s}" RUN_JOB_CPUS_PER_GPU=4 RUN_JOB_MEM_PER_GPU=24G RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-kn101}"
MODE="${1:?weights|acts <contrast> <slice>|report}"; CON="${2:-}"; SLICE="${3:-}"; TAG="${SLICE//:/-}"
A="benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1"; T="benchmark/02_tasks"
XPY=$(ls -d "${PROJECT_ROOT}"/${T}/*/*/5_scripts_* | tr '\n' ':')
LOGS_DIR="${SCRATCH}/visual_proof_weights/logs"; mkdir -p "${LOGS_DIR}"
ENVS="export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'; export nnUNet_compile=false; export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2; export PYTHONPATH='${XPY}:${PYTHONPATH:-}';
      export VP8_OUT='${PROJECT_ROOT}/${A}/outputs/plots/visual_proof_erf_tasks'; export VPW_CACHE='${SCRATCH}/visual_proof_weights/cache';
      export VPW_REGION='${VPW_REGION:-brain}'; export VPW_OUT='${PROJECT_ROOT}/${A}/outputs/plots/weights_analysis'; cd '${PROJECT_ROOT}'"
PY=".venv/bin/python ${A}/scripts/visual_proof_weights.py"
case "${MODE}" in
  weights) run_job --name vpw_weights --gpus 0 --cpus 2 --mem 24G --time 00:30:00 --log "${LOGS_DIR}/weights.log" -- bash -c "${ENVS}; ${PY} --weights" ;;
  acts)    run_job --name "vpw_${CON}_${TAG}" --gpus 1 --time 00:30:00 --log "${LOGS_DIR}/acts_${CON}_${TAG}.log" -- bash -c "${ENVS}; ${PY} --acts --contrast ${CON} ${SLICE:+--case-slice ${SLICE}}" ;;
  report)  run_job --name vpw_report --gpus 0 --cpus 2 --mem 24G --time 00:30:00 --log "${LOGS_DIR}/report.log" -- bash -c "${ENVS}; ${PY} --report" ;;
  *) echo "unknown mode ${MODE}" >&2; exit 1 ;;
esac
