#!/usr/bin/env bash
# Killarney (per-GPU, L40S) launcher for the signed-gradient spectral analysis (visual_proof_ig_texture.py; pre-registration in
# outputs/plots/ig_texture/ig_texture.md). One single-GPU 30-min job per (mode, cell, case chunk); resumable per case.
#   bash run_visual_proof_ig_killarney.sh ig t1n_t2f 0:12        # cases 0-11 of the 35-case sample
#   bash run_visual_proof_ig_killarney.sh maps "CASE_A CASE_B"  # dense HF-share maps for the fixed-rule example cases (main cell)
#   bash run_visual_proof_ig_killarney.sh rep                    # CPU report + figure
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.."
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
export RUN_JOB_GPU_TYPE="${RUN_JOB_GPU_TYPE:-l40s}" RUN_JOB_CPUS_PER_GPU=4 RUN_JOB_MEM_PER_GPU=24G RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-kn101}"
MODE="${1:?ig|maps|spoke|rep}"; CELL="${2:-}"; SLICE="${3:-}"; TAG="${SLICE//:/-}"
A="benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1"; T="benchmark/02_tasks"
XPY=$(ls -d "${PROJECT_ROOT}"/${T}/*/*/5_scripts_* | tr '\n' ':')
LOGS_DIR="${SCRATCH}/visual_proof_killarney/ig"; mkdir -p "${LOGS_DIR}"
ENVS="export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'; export nnUNet_compile=false;
      export PYTHONPATH='${PROJECT_ROOT}/${T}/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma:${PROJECT_ROOT}/${T}/brain_ms/open-ms/5_scripts_open-ms:${PROJECT_ROOT}/${T}/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic:${PYTHONPATH:-}';
      export PYTHONPATH=\"${XPY}:\${PYTHONPATH}\";
      export VP_OUT='${PROJECT_ROOT}/${A}/outputs/plots/visual_proof_v2_brats_full'; export VP8_OUT='${PROJECT_ROOT}/${A}/outputs/plots/visual_proof_erf_tasks';
      export IG_OUT='${PROJECT_ROOT}/${A}/outputs/plots/ig_texture'; export IG_K='${IG_K:-8}' IG_BATCH='${IG_BATCH:-4}'; cd '${PROJECT_ROOT}'"
PY=".venv/bin/python ${A}/scripts/visual_proof_ig_texture.py"
case "${MODE}" in
  ig)    run_job --name "ig_${CELL}_${TAG}" --gpus 1 --time 00:30:00 --log "${LOGS_DIR}/ig_${CELL}_${TAG}.log" -- bash -c "${ENVS}; ${PY} --cells ${CELL} ${SLICE:+--case-slice ${SLICE}}" ;;
  maps)  run_job --name ig_maps --gpus 1 --time 00:30:00 --log "${LOGS_DIR}/maps_${CELL// /_}.log" -- bash -c "${ENVS}; ${PY} --maps --cases ${CELL}" ;;
  spoke) run_job --name "igspk_${CELL}_${TAG}" --gpus 1 --time 00:30:00 --log "${LOGS_DIR}/spoke_${CELL}_${TAG}.log" -- bash -c "${ENVS}; ${PY} --spoke --cells ${CELL} ${SLICE:+--case-slice ${SLICE}}" ;;
  rep)   run_job --name ig_report --gpus 0 --cpus 2 --mem 24G --time 00:30:00 --log "${LOGS_DIR}/report.log" -- bash -c "${ENVS}; ${PY} --report" ;;
  *) echo "unknown mode ${MODE}" >&2; exit 1 ;;
esac
