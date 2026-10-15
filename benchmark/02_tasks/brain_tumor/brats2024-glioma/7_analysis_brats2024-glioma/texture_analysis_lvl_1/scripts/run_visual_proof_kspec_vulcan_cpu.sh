#!/usr/bin/env bash
# Vulcan CPU launcher for visual_proof_kspec.py Part A (FINDINGS 46): DCT kernel spectra of every conv layer, 15 training settings x 3 arms x folds 0-2
# + 3 random inits per setting (all checkpoints live on Vulcan). Also runs the Part A/B report and figures on CPU when asked.
#   bash run_visual_proof_kspec_vulcan_cpu.sh spectra     # Part A extraction (loads state dicts only)
#   bash run_visual_proof_kspec_vulcan_cpu.sh report      # kspec.md + csv + summary figures (needs Part B csv rsynced from TamIA for section B)
#   bash run_visual_proof_kspec_vulcan_cpu.sh figure      # example figures (patient slices: never commit)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.."
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
MODE="${1:?spectra|report|figure}"
A="benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1"; T="benchmark/02_tasks"
XPY=$(ls -d "${PROJECT_ROOT}"/${T}/*/*/5_scripts_* | tr '\n' ':')
LOGS="${SCRATCH}/vp_kspec_logs"; mkdir -p "${LOGS}"
ENVS="export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'; export nnUNet_compile=false; export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8;
      export PYTHONPATH='${XPY}${PYTHONPATH:-}'; export VP8_OUT='${SCRATCH}/vp_kspec_tmp/erf_tasks'; export VP_ERFO_OUT='${SCRATCH}/vp_kspec_tmp/erf_outcome';
      export VP_OUT='${SCRATCH}/vp_kspec_tmp/v2'; export VP7_OUT='${SCRATCH}/vp_kspec_tmp/erf'; export VP_KSPEC_OUT='${PROJECT_ROOT}/${A}/outputs/plots/kspec'; cd '${PROJECT_ROOT}'"
PY=".venv/bin/python ${A}/scripts/visual_proof_kspec.py"
case "${MODE}" in
  spectra) run_job --name kspec_A --gpus 0 --cpus 8 --mem 32G --time 02:00:00 ${WAIT:+--wait} --log "${LOGS}/kspec_A.log" -- bash -c "${ENVS}; ${PY} --spectra" ;;
  report)  run_job --name kspec_report --gpus 0 --cpus 4 --mem 32G --time 00:45:00 ${WAIT:+--wait} --log "${LOGS}/kspec_report.log" -- bash -c "${ENVS}; ${PY} --report" ;;
  figure)  run_job --name kspec_figure --gpus 0 --cpus 4 --mem 32G --time 00:45:00 ${WAIT:+--wait} --log "${LOGS}/kspec_figure.log" -- bash -c "${ENVS}; ${PY} --figure" ;;
  *) echo "unknown mode ${MODE}" >&2; exit 1 ;;
esac
