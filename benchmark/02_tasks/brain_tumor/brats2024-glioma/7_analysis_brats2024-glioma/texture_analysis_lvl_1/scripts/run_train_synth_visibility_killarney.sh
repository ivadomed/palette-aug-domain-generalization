#!/usr/bin/env bash
# Killarney launcher for train_synth_edema_visibility.py (training-side edema-visibility test, rung 4.5 flat vs rung 5 real fill).
# Inputs are STAGED (from Vulcan, by rsync) into ${SCRATCH}/train_synth_vis/ -- never into the Killarney repo's own AugLab (older commit,
# used by other jobs):  data/<DatasetXXX>/nnUNetPlans_3d_fullres/<case>{.b2nd,_seg.b2nd,.pkl}, auglab_snapshot/auglab (Vulcan AugLab package
# snapshot, prepended to PYTHONPATH), cases.txt. Outputs: ${SCRATCH}/train_synth_vis/out (rsync back to Vulcan
# outputs/plots/train_synth_visibility/). CPU-only jobs, <= 30 min each.
#   bash run_train_synth_visibility_killarney.sh verify
#   bash run_train_synth_visibility_killarney.sh run t1n 0:4 [DRAWS]
#   bash run_train_synth_visibility_killarney.sh grid [CASE]
#   bash run_train_synth_visibility_killarney.sh report
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../../.."
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
export RUN_JOB_EXCLUDE_NODES=kn101
MODE="${1:?verify|run|grid|report}"
A="benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1"
W="${SCRATCH}/train_synth_vis"; LOGS="${W}/logs"; mkdir -p "${LOGS}" "${W}/out"
PY="${A}/scripts/train_synth_edema_visibility.py"
ENVS="export PYTHONPATH='${W}/auglab_snapshot':\${PYTHONPATH:-}; export BLOSC2_NTHREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; cd '${PROJECT_ROOT}';
      md5sum '${W}/auglab_snapshot/auglab/transforms/gpu/palette_noisefill.py' '${W}/auglab_snapshot/auglab/transforms/gpu/fromSeg.py' > '${W}/out/provenance.txt';
      cat '${W}/auglab_snapshot/COMMIT' >> '${W}/out/provenance.txt' 2>/dev/null || true"
COMMON="--data-root '${W}/data' --cases-file '${W}/cases.txt' --out '${W}/out'"
case "${MODE}" in
  verify) run_job --name tsv_verify --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --log "${LOGS}/verify.log" -- bash -c "${ENVS}; .venv/bin/python ${PY} --verify ${COMMON} --verify-seeds ${VERIFY_SEEDS:-12}" ;;
  run)    C="${2:?t1n|t2w}"; S="${3:?a:b}"; D="${4:-30}"
          run_job --name "tsv_${C}_${S//:/-}" --gpus 0 --cpus 4 --mem 24G --time 00:30:00 --log "${LOGS}/run_${C}_${S//:/-}.log" -- bash -c "${ENVS}; .venv/bin/python ${PY} --run --contrast ${C} --case-slice ${S} --draws ${D} ${COMMON}" ;;
  grid)   run_job --name tsv_grid --gpus 0 --cpus 4 --mem 16G --time 00:20:00 --log "${LOGS}/grid.log" -- bash -c "${ENVS}; .venv/bin/python ${PY} --grid ${COMMON} ${2:+--grid-case $2}" ;;
  report) run_job --name tsv_report --gpus 0 --cpus 2 --mem 8G --time 00:15:00 --log "${LOGS}/report.log" -- bash -c "${ENVS}; .venv/bin/python ${PY} --report --out '${W}/out'" ;;
  *) echo "unknown mode ${MODE}" >&2; exit 1 ;;
esac
