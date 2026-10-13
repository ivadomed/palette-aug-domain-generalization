#!/usr/bin/env bash
# Killarney launcher for visual_proof_domain_probe.py (domain-flag representation probe, rung 4.5 flat vs rung 5 real fill, BraTS glioma).
# Inputs: staged training patches + verified AugLab snapshot in ${SCRATCH}/train_synth_vis (see run_train_synth_visibility_killarney.sh),
# ${SCRATCH}/domain_probe/fold_train_cases.json (staged cases per fold's TRAINING set, from Vulcan's splits_final.json).
# Features -> ${SCRATCH}/domain_probe/feat/<group>_f<fold>/<part>/<case>.npz ; report -> ${SCRATCH}/domain_probe/report (rsync to Vulcan
# outputs/plots/domain_probe). GPU extraction jobs: 1 L40S, 4 CPUs, 24G, <= 30 min, resumable per case. Report: CPU only.
#   bash run_visual_proof_domain_probe_killarney.sh extract t1n train 0
#   bash run_visual_proof_domain_probe_killarney.sh extract t1n test_t2f 0 [0:35]
#   bash run_visual_proof_domain_probe_killarney.sh report
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.."
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
export RUN_JOB_GPU_TYPE="${RUN_JOB_GPU_TYPE:-l40s}" RUN_JOB_CPUS_PER_GPU=4 RUN_JOB_MEM_PER_GPU=24G RUN_JOB_EXCLUDE_NODES=kn101
MODE="${1:?extract|report}"
A="benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1"; T="benchmark/02_tasks"
XPY=$(ls -d "${PROJECT_ROOT}"/${T}/*/*/5_scripts_* | tr '\n' ':')
W="${SCRATCH}/domain_probe"; LOGS="${W}/logs"; mkdir -p "${LOGS}"
ENVS="export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'; export nnUNet_compile=false;
      export PYTHONPATH='${SCRATCH}/train_synth_vis/auglab_snapshot:${XPY}':\${PYTHONPATH:-};
      export BLOSC2_NTHREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4;
      export DP_WORK='${W}' DP_STAGED='${SCRATCH}/train_synth_vis' VP_OUT='${W}/_tmp/v2' VP7_OUT='${W}/_tmp/v7' VP8_OUT='${W}/_tmp/v8'; cd '${PROJECT_ROOT}'"
PY="${A}/scripts/visual_proof_domain_probe.py"
case "${MODE}" in
  extract) G="${2:?t1n|t2w}"; P="${3:?train|test_<contrast>}"; F="${4:-0}"; S="${5:-}"; TAG="${S//:/-}"
           run_job --name "dp_${G}_f${F}_${P}${TAG:+_${TAG}}" --gpus 1 --time 00:30:00 --log "${LOGS}/${G}_f${F}_${P}${TAG:+_${TAG}}.log" -- \
             bash -c "${ENVS}; .venv/bin/python ${PY} --extract --group ${G} --fold ${F} --part ${P} ${S:+--case-slice ${S}}" ;;
  report)  run_job --name dp_report --gpus 0 --cpus 2 --mem 24G --time 00:30:00 --log "${LOGS}/report.log" -- \
             bash -c "${ENVS}; .venv/bin/python ${PY} --report --folds ${FOLDS:-0} --out '${W}/report'" ;;
  *) echo "unknown mode ${MODE}" >&2; exit 1 ;;
esac
