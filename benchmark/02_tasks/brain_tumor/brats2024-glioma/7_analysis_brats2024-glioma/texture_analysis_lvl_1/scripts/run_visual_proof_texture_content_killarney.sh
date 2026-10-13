#!/usr/bin/env bash
# Killarney launcher for visual_proof_texture_content.py (FINDINGS section 31: texture content of each stage + readout alignment, flat vs real).
# Activations -> ${SCRATCH}/texture_content/feat ; md/png -> outputs/plots/visual_proof_erf_tasks/texture_content. Jobs <= 30 min, resumable per case.
#   bash run_visual_proof_texture_content_killarney.sh extract t1n t2f 0:15
#   bash run_visual_proof_texture_content_killarney.sh report
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.."
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
export RUN_JOB_GPU_TYPE="${RUN_JOB_GPU_TYPE:-l40s}" RUN_JOB_CPUS_PER_GPU=4 RUN_JOB_MEM_PER_GPU=24G RUN_JOB_EXCLUDE_NODES=kn101
MODE="${1:?extract|report}"
A="benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1"; T="benchmark/02_tasks"
XPY=$(ls -d "${PROJECT_ROOT}"/${T}/*/*/5_scripts_* | tr '\n' ':')
W="${SCRATCH}/texture_content"; LOGS="${W}/logs"; mkdir -p "${LOGS}"
ENVS="export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'; export nnUNet_compile=false; export PYTHONPATH='${XPY}':\${PYTHONPATH:-};
      export BLOSC2_NTHREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; export TC_FEAT='${W}/feat${TC_TAG:-}' TC_TAG='${TC_TAG:-}' TC_NCASES='${TC_NCASES:-30}';
      export VP_OUT='${W}/_tmp/v2' VP7_OUT='${W}/_tmp/v7' VP8_OUT='${PROJECT_ROOT}/${A}/outputs/plots/visual_proof_erf_tasks'; cd '${PROJECT_ROOT}'"
PY="${A}/scripts/visual_proof_texture_content.py"
case "${MODE}" in
  extract) G="${2:?t1n|t2w}"; TE="${3:?t2f|own}"; S="${4:-}"; TAG="${S//:/-}"
           run_job --name "tc_${G}_${TE}${TAG:+_${TAG}}" --gpus 1 --time 00:30:00 --log "${LOGS}/${G}_${TE}${TAG:+_${TAG}}.log" -- \
             bash -c "${ENVS}; .venv/bin/python ${PY} --extract --group ${G} --test ${TE} ${S:+--case-slice ${S}}" ;;
  report)  run_job --name tc_report --gpus 0 --cpus 4 --mem 48G --time 00:30:00 --log "${LOGS}/report.log" -- \
             bash -c "${ENVS}; .venv/bin/python ${PY} --report" ;;
  *) echo "unknown mode ${MODE}" >&2; exit 1 ;;
esac
