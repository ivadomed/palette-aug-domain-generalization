#!/bin/bash
# CPU controller (submit via launch.sh) for ONE brats2024-glioma training family on BraTS-SSA 2024:
# every runs.txt row of that family with action predict|srcmatch (optionally only the given keys):
#   predict (3 parallel 1-GPU fold jobs via the shared predict driver, checkpoint_best, no TTA)
#   -> 0-byte / count check -> evaluate inline (one at a time, flock) -> eval_all row check.
# Resumable: a run whose 12 prediction dirs already hold 95 non-empty files is not re-predicted, and
# one whose 3 eval_all.csv already have 1140 rows is not re-evaluated.
# Status: $SCRATCH/review_20261020/brats_ssa/logs/<family>.status
#   bash run_family.sh <t1n|t2w|t2f|t1c> [key ...]
set -uo pipefail
export BSSA_REPO="${BSSA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"   # repo root (Vulcan or Killarney checkout)
cd "${BSSA_REPO}"
FAM="${1:?family}"; shift
KEYS=" $* "
HERE="${PWD}/scripts/cluster/brats_ssa_pa_alone"
if [ -z "${BSSA_SNAPSHOT:-}" ]; then     # private copy: later edits never hit a running controller
  snapd="${SCRATCH:?}/review_20261020/brats_ssa/logs/snapshot_${FAM}_$$"; mkdir -p "${snapd}"
  cp "${HERE}"/*.sh "${HERE}/runs.txt" "${snapd}/" && export BSSA_SNAPSHOT="${snapd}" && exec bash "${snapd}/run_family.sh" "${FAM}" "$@"
fi
H="${BSSA_SNAPSHOT}"
LOGD="${SCRATCH}/review_20261020/brats_ssa/logs"; ST="${LOGD}/${FAM}.status"; LOCK="${LOGD}/${FAM}.evallock"
PRED_ROOT="${PWD}/benchmark/03_archive/brats-ssa2024/8_results_brats-ssa2024/01_predictions/brats2024_glioma_model/${FAM}"
MET_ROOT="${PWD}/benchmark/03_archive/brats-ssa2024/8_results_brats-ssa2024/02_metrics/brats2024_glioma_model/${FAM}"
export RUN_JOB_ACCOUNT=aip-jcohen RUN_JOB_GPU_TYPE="${RUN_JOB_GPU_TYPE:-l40s}"
export RUN_JOB_CPUS_PER_GPU="${RUN_JOB_CPUS_PER_GPU:-8}" RUN_JOB_MEM_PER_GPU="${RUN_JOB_MEM_PER_GPU:-48G}"
unset RUN_JOB_DEPENDENCY RUN_JOB_INLINE RUN_JOB_PACK_DIR
log() { echo "[$(date '+%F %T')] $*" | tee -a "${ST}"; }

preds_ok() {  # $1 pred base, $2 tag
  local b="$1" t="$2" F it d n z
  for F in 0 1 2; do for it in t1n t1c t2w t2f; do
    d="${b}/fold${F}/${t:+${t}/}${it}"
    n=$(ls "${d}" 2>/dev/null | command grep -c '\.nii\.gz$' || true)
    z=$(/usr/bin/find "${d}" -maxdepth 1 -name '*.nii.gz' -size 0 2>/dev/null | wc -l)
    [ "${n}" = 95 ] && [ "${z}" = 0 ] || { echo "fold${F}/${t:+${t}/}${it}: n=${n} zero=${z}"; return 1; }
  done; done
}
evals_ok() {  # $1 metrics dir
  local F r; for F in 0 1 2; do
    [ -f "$1/fold${F}/eval_all.csv" ] || return 1
    r=$(( $(wc -l < "$1/fold${F}/eval_all.csv") - 1 )); [ "${r}" = 1140 ] || return 1
  done
}

one_run() {
  local key="$1" cat="$2" tr="$3" rid="$4" act="$5" tag="" suffix=""
  [ "${act}" = srcmatch ] && { tag=srcmatch; suffix="_srcmatch_${FAM}"; }
  local pb="${PRED_ROOT}/${cat}/${rid}" md="${MET_ROOT}/${cat}_${rid}${tag:+_${tag}}"
  if ! preds_ok "${pb}" "${tag}" >/dev/null; then
    log "${key}: predict ${rid}${tag:+ (${tag})}"
    env FAMILY="${FAM}" METHOD="${FAM}_${key}" TRAINER="${tr}" CATEGORY="${cat}" \
        PREDICT_INPUT_SUFFIX="${suffix}" PREDICT_OUTPUT_SUBDIR="${tag}" \
        bash "${H}/predict_shim.sh" "${rid}" all > "${LOGD}/${FAM}_${key}.predict.log" 2>&1
    local why; why="$(preds_ok "${pb}" "${tag}")" || { log "${key}: PREDICT INCOMPLETE ${why} (see ${FAM}_${key}.predict.log + ${PRED_ROOT}/${cat}/_logs)"; return 1; }
    log "${key}: predictions complete (12 x 95, no 0-byte)"
  fi
  if ! evals_ok "${md}"; then
    log "${key}: evaluate"
    ( flock 9; env FAMILY="${FAM}" CKPT_TAG="${tag}" bash "${H}/eval_run.sh" "${rid}" "${cat}" ) 9>"${LOCK}" \
        > "${LOGD}/${FAM}_${key}.eval.log" 2>&1 || { log "${key}: EVAL FAILED (see ${FAM}_${key}.eval.log)"; return 1; }
  fi
  evals_ok "${md}" && log "${key}: OK eval_all 3 x 1140 rows -> $(basename "${md}")" || { log "${key}: eval rows wrong"; return 1; }
}

log "START family=${FAM} keys='${KEYS}' job=${SLURM_JOB_ID:-none} host=$(hostname)"
pids=()
while IFS=$'\t' read -r mod key cat tr rid act; do
  [ "${mod}" = "${FAM}" ] || continue
  [ "${act}" = reuse ] && continue
  [ "${KEYS}" = "  " ] || [[ "${KEYS}" == *" ${key} "* ]] || continue
  one_run "${key}" "${cat}" "${tr}" "${rid}" "${act}" &
  pids+=($!)
  sleep 20      # space the sbatch calls out
done < <(command grep -v '^#' "${H}/runs.txt" | awk -F'\t' 'BEGIN{split("pa_alone auglab_default synthseg_EM pa_auglab srcsm baseline synthseg_noEM pa_auglab_val100",o," "); for(i in o) pr[o[i]]=i} {print pr[$2]"\t"$0}' | sort -n | cut -f2-)
FAIL=0; for p in "${pids[@]}"; do wait "$p" || FAIL=1; done
log "END family=${FAM} fail=${FAIL}"
exit ${FAIL}
