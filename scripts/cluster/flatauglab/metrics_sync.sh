#!/usr/bin/env bash
# Bring the flatauglab arm's TamIA-side METRICS back to Vulcan (2026-10-18), one unit at a time, run by companions_poller.sh on the Vulcan login
# node (light: ssh + rsync of small csv / md trees). A unit is synced only when ALL its TamIA jobs are COMPLETED with exit 0:0 (the job
# bodies exit non-zero when their own count audit fails, so 0:0 = audit OK). Predictions are NOT synced (re-creatable).
#   own:<setting>   the setting's own test set: predict + eval job of run_ids.tsv (atlas: the inline 05_35 job)
#                   <tamia results>/02_metrics/<model>/<tc>/auglab_<RID> -> Vulcan <task>/<ds>/8_results_<ds>/02_metrics/<model>/<tc>/auglab_<RID>
#                   (toothfairy2: the mandible-only tree cbct_mandible_only/ -> cbct/, what the Vulcan headline runs hold)
#   comp:mandible   hanseg + pddca  .../toothfairy2_model/cbct/mandible_only_sif/auglab_<tf2 RID> -> .../toothfairy2_model/cbct/auglab_<RID>
#   comp:chaos_fov  amos + sliver07 .../chaos_model/<tc>/fov_crop/auglab_<chaos RID>   (same tail on Vulcan)
#   comp:hcc        lld-mmri-hcc + liverhccseg .../atlas_liver_hcc_model/t1w/auglab_<atlas RID>   (same tail)
# (path rules checked 2026-10-18 by md5-matching the synthseg_EM eval_all.csv files of both clusters.)
#   bash scripts/cluster/flatauglab/metrics_sync.sh <unit> [--dry-run]
# Exit: 0 synced (or already) | 3 not ready | 4 connection problem (retry, no attempt used) | 10 a job FAILED / audit failed / no metrics
#       (printed as "FAILED ... job <id>") | 5 md5 / rsync verification failed.
set -uo pipefail
REPO=/project/aip-jcohen/paulh/mri_synthesis_project; cd "${REPO}"
CD="${FLATAUG_COMPANIONS_DIR:-/scratch/paulh/flatauglab_20261028/companions}"
RUNS="${FLATAUG_RUN_TABLE:-/scratch/paulh/flatauglab_20261028/run_ids.tsv}"
TS="$(cat "${FLATAUG_TS_FILE:-/scratch/paulh/flatauglab_20261028/TS}")" || exit 2   # the pack root TS (copied from TamIA after `launch.sh record`)
PACKS_ROOT=/scratch/p/paulh/_packruns/flatauglab_${TS}
TAMIA=tamia.alliancecan.ca; SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 "${TAMIA}")
UNIT="${1:?usage: metrics_sync.sh own:<setting>|comp:mandible|comp:chaos_fov|comp:hcc [--dry-run]}"; DRY=0; [ "${2:-}" = --dry-run ] && DRY=1
T="${REPO}/benchmark/02_tasks"
UD="${CD}/_metrics/${UNIT/:/_}"; mkdir -p "${UD}"
log() { echo "[$(date '+%F %T')] [metrics ${UNIT}] $*"; }
[ "${DRY}" = 1 ] || [ ! -f "${UD}/METRICS_SYNCED" ] || { log "already METRICS_SYNCED"; exit 0; }
rid_of() { awk -F'\t' -v s="$1" '$1==s {print $3}' "${RUNS}" | sort -u; }
declare -A TASK=([brats]=brain_tumor/brats2024-glioma [chaos]=abdomen_healthy/chaos [onh]=brain_healthy/on-harmony [openms]=brain_ms/open-ms
  [tf2]=mandible_healthy/toothfairy2 [ispy2]=breast_cancer/ispy2 [pelvic]=pelvis_healthy/totalseg-pelvic [atlas]=liver_hcc/atlas-liver-hcc
  [panseg]=pancreas_disease/pansegdata)
SRCS=(); DSTS=(); JOBS=""
case "${UNIT}" in
  own:*)
    s="${UNIT#own:}"; RID="$(rid_of "$s")"; [ "$(echo "${RID}" | grep -c .)" = 1 ] || { log "unknown setting $s"; exit 2; }
    JOBS="$(awk -F'\t' -v s="$s" '$1==s {print $7; print $8}' "${RUNS}" | grep -E '^[0-9]+$' | sort -u | paste -sd, -)"
    ST="$("${SSH[@]}" 'bash -s' < scripts/cluster/flatauglab/tamia_ckpt_status.sh)" || { log "connection problem (ckpt status)"; exit 4; }
    SRC="$(echo "${ST}" | awk -F'\t' -v r="${RID}" '$1=="ST" && $2==r {print $5}' | sort -u)"
    [ "$(echo "${SRC}" | grep -c .)" = 1 ] || { log "run dir of ${RID} not found on TamIA"; exit 2; }
    tail="${SRC#*/01_predictions/}"; mt="${tail%/auglab/*}"   # <model>/<tc>
    K="${s%%_*}"; DS="$(basename "${TASK[$K]}")"
    src_mt="${mt}"; [ "$K" = tf2 ] && src_mt="${mt}_mandible_only"
    SRCS+=("${SRC%%/01_predictions/*}/02_metrics/${src_mt}/auglab_${RID}"); DSTS+=("${T}/${TASK[$K]}/8_results_${DS}/02_metrics/${mt}/auglab_${RID}") ;;
  comp:*)
    g="${UNIT#comp:}"
    Q="$("${SSH[@]}" "cat ${PACKS_ROOT}/post_companions/${g}/QUEUED")" || { log "connection problem or no QUEUED file for ${g}"; exit 4; }
    JOBS="$(echo "${Q}" | sed 's/ after .*//' | tr ' ' '\n' | grep -E '^[0-9]+$' | paste -sd, -)"
    case "$g" in
      mandible) RID="$(rid_of tf2_cbct)"
        for d in hanseg pddca; do SRCS+=("/scratch/p/paulh/${d}/8_results/02_metrics/toothfairy2_model/cbct/mandible_only_sif/auglab_${RID}")
          DSTS+=("${T}/mandible_healthy/${d}/8_results_${d}/02_metrics/toothfairy2_model/cbct/auglab_${RID}"); done ;;
      chaos_fov)
        for tc in t1in t2spir; do RID="$(rid_of chaos_${tc})"; for d in amos sliver07; do
          SRCS+=("/scratch/p/paulh/${d}/8_results_${d}/02_metrics/chaos_model/${tc}/fov_crop/auglab_${RID}")
          DSTS+=("${T}/abdomen_healthy/${d}/8_results_${d}/02_metrics/chaos_model/${tc}/fov_crop/auglab_${RID}"); done; done ;;
      hcc) RID="$(rid_of atlas_t1w)"
        for d in lld-mmri-hcc liverhccseg; do SRCS+=("/scratch/p/paulh/${d}/8_results_${d}/02_metrics/atlas_liver_hcc_model/t1w/auglab_${RID}")
          DSTS+=("${T}/liver_hcc/${d}/8_results_${d}/02_metrics/atlas_liver_hcc_model/t1w/auglab_${RID}"); done ;;
      *) log "unknown companion group ${g}"; exit 2 ;;
    esac ;;
  *) log "unknown unit"; exit 2 ;;
esac
[ -n "${JOBS}" ] || { log "no job ids for the unit"; exit 2; }
for i in "${!SRCS[@]}"; do [[ "${SRCS[$i]}" == *_auglabAugFlat_v26_6_2_train050_val000_${TS} ]] || { log "refusing: not a flatauglab dir ${SRCS[$i]}"; exit 2; }; done

OUT="$("${SSH[@]}" 'bash -s' -- "${JOBS}" "${SRCS[@]}" < scripts/cluster/flatauglab/tamia_metrics_status.sh)" || { log "connection problem (metrics status)"; exit 4; }
echo "${OUT}" | awk -F'\t' '$1=="JOB"' > "${UD}/jobs_last.tsv"
bad=0; notready=0
while IFS=$'\t' read -r _ j st ec reason; do
  case "${st}" in
    COMPLETED) [ "${ec}" = 0:0 ] || { log "FAILED job ${j}: COMPLETED with exit ${ec}"; bad=1; } ;;
    PENDING) if [ "${reason}" = DependencyNeverSatisfied ]; then log "FAILED job ${j}: PENDING ${reason} (an upstream job failed)"; bad=1; else notready=1; fi ;;
    RUNNING|REQUEUED|RESIZING|SUSPENDED|"") notready=1 ;;
    *) log "FAILED job ${j}: ${st} exit ${ec}"; bad=1 ;;
  esac
done < "${UD}/jobs_last.tsv"
[ "${bad}" = 0 ] || exit 10
[ "${notready}" = 0 ] || { log "not ready (jobs ${JOBS}: $(awk -F'\t' '{printf "%s=%s ", $2, $3}' "${UD}/jobs_last.tsv"))"; [ "${DRY}" = 1 ] || exit 3; }
miss="$(echo "${OUT}" | awk -F'\t' '$1=="DIR" && $3=="missing" {print $2}')"
if [ -n "${miss}" ]; then
  [ "${notready}" = 1 ] && { log "(dry-run) metrics dirs not there yet: $(echo ${miss})"; }
  [ "${notready}" = 1 ] || { log "FAILED: jobs ${JOBS} completed but no metrics at: $(echo ${miss})"; exit 10; }
fi
for i in "${!SRCS[@]}"; do
  s="${SRCS[$i]}"; d="${DSTS[$i]}"
  n="$(echo "${OUT}" | awk -F'\t' -v p="${s}/" '$1=="MD5" && index($3,p)==1' | grep -c . || true)"
  [ "${notready}" = 1 ] || [ "${n}" -ge 3 ] || { log "FAILED: ${s} has ${n} fold eval_all.csv (expect >= 3)"; exit 10; }
  [ -d "$(dirname "$(dirname "${d}")")" ] || { log "refusing: Vulcan tree $(dirname "$(dirname "${d}")") missing"; exit 2; }
  RS=(rsync -a --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=30" "${TAMIA}:${s}/" "${d}/")
  if [ "${DRY}" = 1 ]; then log "[dry] mkdir -p ${d} && ${RS[*]}"; continue; fi
  mkdir -p "${d}"; "${RS[@]}"; rc=$?
  case "${rc}" in 0) ;; 10|12|30|35|255) log "rsync connection problem rc=${rc}"; exit 4 ;; *) log "rsync failed rc=${rc} for ${s}"; exit 5 ;; esac
done
[ "${DRY}" = 0 ] || { log "dry-run done (nothing synced)"; exit 0; }
# md5 of every eval_all.csv (TamIA value taken before the rsync; the jobs are finished, the files are static)
v=0
while IFS=$'\t' read -r _ m f; do
  for i in "${!SRCS[@]}"; do s="${SRCS[$i]}"; [[ "$f" == "${s}/"* ]] || continue
    l="${DSTS[$i]}/${f#${s}/}"; lm="$(md5sum < "${l}" 2>/dev/null | cut -c1-32)"
    [ "${lm}" = "${m}" ] || { log "MD5 MISMATCH ${l} (${lm:-missing} vs ${m})"; v=1; }; done
done < <(echo "${OUT}" | awk -F'\t' '$1=="MD5"')
[ "${v}" = 0 ] || exit 5
{ echo "synced $(date '+%F %T') jobs ${JOBS}"; for i in "${!SRCS[@]}"; do echo "${SRCS[$i]} -> ${DSTS[$i]}"; done; echo "${OUT}" | awk -F'\t' '$1=="MD5"'; } > "${UD}/METRICS_SYNCED"
log "METRICS_SYNCED: ${#SRCS[@]} dir(s), $(echo "${OUT}" | grep -c '^MD5') eval_all.csv md5-verified"
printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "${UNIT}" metrics_synced "${#SRCS[@]} dirs" >> "${CD}/status.tsv"
