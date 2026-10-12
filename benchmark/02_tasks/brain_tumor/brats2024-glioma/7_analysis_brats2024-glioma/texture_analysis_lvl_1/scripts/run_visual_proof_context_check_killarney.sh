#!/usr/bin/env bash
# Killarney CPU job: stress test of bg_context.md (visual_proof_context_check.py) -> outputs/plots/visual_proof_erf_tasks/bg/context_check/
#   bash run_visual_proof_context_check_killarney.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.."
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
export RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-kn101}"
A="benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1"; T="benchmark/02_tasks"
LOGS_DIR="${SCRATCH}/visual_proof_killarney"; mkdir -p "${LOGS_DIR}"
ENVS="export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'; export nnUNet_compile=false;
      export PYTHONPATH='${PROJECT_ROOT}/${T}/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma:${PROJECT_ROOT}/${T}/brain_ms/open-ms/5_scripts_open-ms:${PROJECT_ROOT}/${T}/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic:${PYTHONPATH:-}';
      export VP8_OUT='${PROJECT_ROOT}/${A}/outputs/plots/visual_proof_erf_tasks'; cd '${PROJECT_ROOT}'"
run_job --name vp8_ctxcheck --gpus 0 --cpus 2 --mem 24G --time 00:30:00 --log "${LOGS_DIR}/v8_ctxcheck.log" -- bash -c "${ENVS}; .venv/bin/python ${A}/scripts/visual_proof_context_check.py"
