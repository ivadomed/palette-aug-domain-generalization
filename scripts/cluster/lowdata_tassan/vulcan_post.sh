#!/bin/bash
# Run ON VULCAN after tassan wrote /scratch/paulh/lowdata_tassan/pelvis_nNN.tassan_done: evaluate both runs of level N
# with the shared evaluate script (06_01, TRAINING_CONTRAST=mri, METRICS_SUBDIR=lowdata; one CPU run_job each, in
# parallel), check row counts (per fold: 30 ct + 27 mri cases x 11 labels), then rebuild the table (06_21).
#   bash scripts/cluster/lowdata_tassan/vulcan_post.sh <N> [<N> ...]
#   LOWDATA_CONTRAST=ct bash scripts/cluster/lowdata_tassan/vulcan_post.sh <N>   # CT-trained: pelvis_ct_nNN markers,
#                                                     # TRAINING_CONTRAST=ct, table 06_21 ct
# EVAL_MEM defaults to 64G (the N=10 MRI evaluation ran out of memory at 06_01's default 32G).
set -euo pipefail
REPO=/project/aip-jcohen/paulh/mri_synthesis_project; cd "${REPO}"
VSTAGE="${VSTAGE:-/scratch/paulh/lowdata_tassan}"
C="${LOWDATA_CONTRAST:-mri}"
case "${C}" in mri) MPRE=pelvis ;; ct) MPRE=pelvis_ct ;; *) echo "LOWDATA_CONTRAST must be mri or ct" >&2; exit 1 ;; esac
export EVAL_MEM="${EVAL_MEM:-64G}"
SC=benchmark/02_tasks/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic
M=benchmark/02_tasks/pelvis_healthy/totalseg-pelvic/8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/${C}/lowdata
P=benchmark/02_tasks/pelvis_healthy/totalseg-pelvic/8_results_totalseg-pelvic/01_predictions/totalseg_pelvic_model/${C}/auglab
log() { echo "$(date -u +%FT%TZ) vulcan_post $*" | tee -a "${VSTAGE}/TASSAN_LOG.txt"; }
for N in "$@"; do
  NN="$(printf '%02d' "${N}")"; mk="${VSTAGE}/${MPRE}_n${NN}.tassan_done"; L="n${NN}"; [ "${C}" = ct ] && L="ct_n${NN}"
  [ -f "${mk}" ] || { echo "ERROR: ${mk} missing" >&2; exit 1; }
  pids=(); rids=()
  for arm in flat texture; do
    rid="$(awk -v a="${arm}" '$1==a {print $2}' "${mk}")"; rids+=("${rid}")
    for f in 0 1 2; do
      [ "$(ls ${P}/${rid}/fold${f}/ct/*.nii.gz | wc -l)" = 30 ] && [ "$(ls ${P}/${rid}/fold${f}/mri/*.nii.gz | wc -l)" = 27 ] \
        || { log "${L} ${rid} fold${f}: prediction count wrong"; exit 1; }
    done
    [ -d "${M}/auglab_${rid}" ] && { log "${L}: ${M}/auglab_${rid} exists, move it aside before re-evaluating"; exit 1; }
    TRAINING_CONTRAST="${C}" METRICS_SUBDIR=lowdata bash "${SC}/06_evaluate/06_01_evaluate_run.sh" "${rid}" auglab & pids+=($!)
    sleep 5
  done
  for p in "${pids[@]}"; do wait "${p}"; done
  for rid in "${rids[@]}"; do for f in 0 1 2; do
    n=$(( $(wc -l < "${M}/auglab_${rid}/fold${f}/eval_all.csv") - 1 ))
    [ "${n}" = $(( (30 + 27) * 11 )) ] || { log "${L} ${rid} fold${f}: eval_all.csv has ${n} rows, expected 627"; exit 1; }
  done; done
  log "${L} evaluated: ${rids[*]} (627 rows per fold)"
  touch "${VSTAGE}/${MPRE}_n${NN}.vulcan_eval_done"
done
bash "${SC}/06_evaluate/06_21_lowdata_texture_step.sh" "${C}"
