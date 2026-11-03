#!/usr/bin/env bash
# VULCAN login-node poller: hand pansegdata low-data (N=5) folds over to tassan (run_panseg.sh there) without training any
# fold twice, and keep predict + evaluate working. Every POLL_S:
#   - <VSTAGE>/Pancreas.tassan_none           -> tassan takes nothing: exit (the Vulcan jobs + post controllers stay as queued)
#   - Pancreas.tassan_failed before any cancel -> logged once, Vulcan untouched, keeps waiting (tassan may retry)
#   - Pancreas.tassan_taking (written by tassan only after every taken fold logged >= 2 epochs), first sight:
#       1. scancel the post controller (POST.queued in /scratch/paulh/lowdata_vulcan/post/<setting>/) of every setting named
#          in it -- BEFORE the fold jobs, since cancelling an afterany dependency would release it early onto missing checkpoints
#       2. scancel exactly the named Vulcan fold jobs (RUN_IDS_lowdata_vulcan.tsv); a job that had already STARTED is cancelled
#          too and its partial fold_k dir moved to toDelete/ (tassan's copy is the one kept); a fold that already FINISHED on
#          Vulcan -> CONFLICT, nothing moved, poller stops (resolve by hand)
#       -> Pancreas.vulcan_cancelled (what was cancelled/moved)
#   - per handed-over setting, once (a) none of its remaining Vulcan fold jobs is in squeue, (b) Pancreas_<setting>.tassan_done
#     exists and every tassan fold dir carries .lowdata_synced_from_tassan (md5-checked by tassan), (c) all 6 folds have
#     checkpoint_final + checkpoint_best: sbatch post_run.sh <setting> (same resources as launch.sh) -> <post>/<setting>/POST_JOB.
#     A remaining Vulcan fold that ended without checkpoint_final is logged with its resume command (not auto-resumed).
# One instance (flock); exits when every handed-over setting has POST_JOB, on <VSTAGE>/STOP_PANSEG, or after MAX_DAYS (5).
# DRY=1: one pass, scancel/sbatch/mv only printed; SIMULATE_TAKING=<file> reads that file instead of Pancreas.tassan_taking
# (dry runs only).
#   start: systemd-run --user --scope --unit=lowdata-panseg-handoff -- nohup bash scripts/cluster/lowdata_tassan/panseg_handoff.sh \
#            >> /scratch/paulh/lowdata_tassan/panseg_handoff.log 2>&1 &
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
source scripts/cluster/lowdata_vulcan/common.sh
VSTAGE="${VSTAGE:-/scratch/paulh/lowdata_tassan}"; POLL_S="${POLL_S:-300}"; MAX_DAYS="${MAX_DAYS:-5}"
TAKING="${VSTAGE}/Pancreas.tassan_taking"
if [ -n "${SIMULATE_TAKING:-}" ]; then [ -n "${DRY:-}" ] || { echo "SIMULATE_TAKING needs DRY=1"; exit 2; }; TAKING="${SIMULATE_TAKING}"; fi
if [ -z "${DRY:-}" ]; then exec 8> "${VSTAGE}/panseg_handoff.lock"; flock -n 8 || { echo "another panseg handoff poller holds the lock"; exit 1; }; fi
log() { echo "[$(date '+%F %T')] [panseg-handoff] $*"; [ -n "${DRY:-}" ] || echo "$(date -u +%FT%TZ) vulcan_handoff $*" >> "${VSTAGE}/TASSAN_LOG.txt"; }
run() { if [ -n "${DRY:-}" ]; then echo "[dry] $*"; else "$@"; fi; }
inq() { [ -n "$(squeue -h -j "$1" -o %i 2>/dev/null)" ]; }            # job still queued/running
jstate() { squeue -h -j "$1" -o %T 2>/dev/null; }
vjob() { awk -F'\t' -v r="$1" -v f="$2" '$4==r {j=$(5+f)} END {print j}' "${LD_TABLE}"; }
t_end=$(( $(date +%s) + MAX_DAYS * 86400 )); log "start pid $$ HEAD $(git rev-parse --short HEAD)${DRY:+ (DRY)}"
while :; do
  [ -f "${VSTAGE}/STOP_PANSEG" ] && { log "STOP_PANSEG -- exiting"; break; }
  [ "$(date +%s)" -lt "${t_end}" ] || { log "max lifetime -- exiting"; break; }
  [ -f "${VSTAGE}/Pancreas.tassan_none" ] && { log "tassan takes no folds (Pancreas.tassan_none) -- Vulcan jobs + post controllers unchanged; exiting"; break; }
  if [ ! -f "${VSTAGE}/Pancreas.vulcan_cancelled" ] || [ -n "${SIMULATE_TAKING:-}" ]; then
    if [ ! -f "${TAKING}" ]; then
      if [ -f "${VSTAGE}/Pancreas.tassan_failed" ] && [ -z "${FAILED_SEEN:-}" ]; then
        log "Pancreas.tassan_failed before any handover -- Vulcan untouched; still waiting (tassan may retry; Pancreas.tassan_none or STOP_PANSEG ends this)"; FAILED_SEEN=1; fi
      [ -n "${DRY:-}" ] && { log "dry: no Pancreas.tassan_taking yet (nothing to do)"; break; }
      sleep "${POLL_S}"; continue
    fi
    # ---- handover: controllers first, then the named fold jobs
    mapfile -t U < <(grep -v '^#' "${TAKING}" | awk 'NF>=4')
    [ "${#U[@]}" -gt 0 ] || { log "ERROR: ${TAKING} has no fold lines"; break; }
    SETS="$(printf '%s\n' "${U[@]}" | awk '{print $1}' | sort -u | tr '\n' ' ')"
    out=(); conflict=0
    for u in "${U[@]}"; do read -r S arm f rid g <<< "${u}"
      ld_setting "${S}" || { conflict=1; continue; }
      [ "$(ld_rid "${S}" "${arm}")" = "${rid}" ] || { log "ERROR: ${S} ${arm}: taking marker RUN_ID ${rid} is not the run table's"; conflict=1; }
      fd="$(ls -d "${LD_PRED_ROOT}/${rid}"/Dataset*/*/fold_${f} 2>/dev/null | head -1)"
      [ -n "${fd}" ] && [ -f "${fd}/checkpoint_final.pth" ] && [ ! -f "${fd}/.lowdata_synced_from_tassan" ] && { log "CONFLICT: ${S} ${arm} fold ${f} already finished on Vulcan (${fd})"; conflict=1; }
    done
    [ "${conflict}" = 0 ] || { log "handover refused (see above); nothing cancelled; exiting"; touch "${VSTAGE}/Pancreas.vulcan_CONFLICT"; break; }
    for S in ${SETS}; do
      pj="$(awk '{print $3}' "${LD_POST}/${S}/POST.queued" 2>/dev/null | tail -1)"
      if [ -n "${pj}" ] && inq "${pj}"; then run scancel "${pj}" && out+=("${S} post_controller ${pj} cancelled"); log "${S}: cancelled post controller ${pj}"
      else out+=("${S} post_controller ${pj:-none} not queued"); fi
    done
    mv_dirs=()
    for u in "${U[@]}"; do read -r S arm f rid g <<< "${u}"; ld_setting "${S}"
      j="$(vjob "${rid}" "${f}")"; st="$(jstate "${j}")"
      if [ -n "${st}" ]; then run scancel "${j}"; out+=("${S} ${arm} fold${f} job ${j} ${st} cancelled"); log "${S} ${arm} fold ${f}: cancelled Vulcan job ${j} (${st})"
        [ "${st}" = PENDING ] || mv_dirs+=("${rid} ${f}")
      else out+=("${S} ${arm} fold${f} job ${j} not in queue"); mv_dirs+=("${rid} ${f}"); fi
    done
    [ -n "${DRY:-}" ] || sleep 30
    for x in "${mv_dirs[@]}"; do read -r rid f <<< "${x}"
      for fd in "${LD_REPO}"/benchmark/02_tasks/pancreas_disease/pansegdata/8_results_pansegdata/01_predictions/pansegdata_model/*/auglab/"${rid}"/Dataset*/*/fold_"${f}"; do
        [ -d "${fd}" ] || continue; [ -f "${fd}/.lowdata_synced_from_tassan" ] && continue
        dst="${LD_REPO}/toDelete/panseg_lowdata_vulcan_partial_$(date +%Y%m%d_%H%M%S)/${rid}"; run mkdir -p "${dst}"
        run mv "${fd}" "${dst}/" && out+=("moved partial Vulcan ${fd} -> ${dst}/"); log "moved partial Vulcan fold dir ${fd} -> ${dst}/"
      done
    done
    if [ -z "${DRY:-}" ]; then printf '%s\n' "# $(date -u +%FT%TZ) from ${TAKING}" "${out[@]}" > "${VSTAGE}/Pancreas.vulcan_cancelled"
    else printf '[dry] would write Pancreas.vulcan_cancelled:\n'; printf '  %s\n' "${out[@]}"; fi
    log "handover done for ${SETS}(${#U[@]} folds)"
  fi
  # ---- post per handed-over setting
  mapfile -t U < <(grep -v '^#' "${TAKING}" | awk 'NF>=4')
  SETS="$(printf '%s\n' "${U[@]}" | awk '{print $1}' | sort -u | tr '\n' ' ')"
  left=0; summary=""
  for S in ${SETS}; do
    ld_setting "${S}"; SD="${LD_POST}/${S}"; [ -f "${SD}/POST_JOB" ] && continue; left=$((left + 1)); why=""
    for arm in flat texture; do rid="$(ld_rid "${S}" "${arm}")"
      for f in 0 1 2; do
        tassan="$(printf '%s\n' "${U[@]}" | awk -v s="${S}" -v a="${arm}" -v f="${f}" '$1==s && $2==a && $3==f' | wc -l)"
        fd="$(ls -d "${LD_PRED_ROOT}/${rid}"/Dataset*/*/fold_${f} 2>/dev/null | head -1)"
        if [ "${tassan}" = 1 ]; then
          [ -n "${fd}" ] && [ -f "${fd}/.lowdata_synced_from_tassan" ] || { why+=" ${arm}f${f}:tassan"; continue; }
        else
          j="$(vjob "${rid}" "${f}")"; if inq "${j}"; then why+=" ${arm}f${f}:vulcan-$(jstate "${j}")"; continue; fi
          { [ -n "${fd}" ] && [ -f "${fd}/checkpoint_final.pth" ]; } || { why+=" ${arm}f${f}:NEEDS-RESUME"
            echo "[$(date '+%F %T')] ${S} ${arm} fold ${f}: job ${j} ended without checkpoint_final -> resume: LOWDATA_CONTRAST=${LD_TC} RUN_JOB_TIME_DEFAULT=${LD_TIME} TRAIN_FOLDS=${f} bash ${LD_SCRIPTS}/${LD_TRAIN} ${arm} ${LD_N} ${rid}  (then fix the job id in ${LD_TABLE})"; continue; }
        fi
        [ -f "${fd}/checkpoint_best.pth" ] || why+=" ${arm}f${f}:no-best"
      done
    done
    [ -f "${VSTAGE}/Pancreas_${S}.tassan_done" ] || why+=" no-Pancreas_${S}.tassan_done"
    [ -f "${VSTAGE}/Pancreas.tassan_failed" ] && why+=" TASSAN_FAILED(read Pancreas.tassan_failed)"
    if [ -n "${why}" ]; then summary+=" ${S}(${why# })"; continue; fi
    if j="$(run sbatch --parsable --account=aip-jcohen --cpus-per-task=16 --mem=96G --time=40:00:00 --job-name="ldpost_${S}" \
            --output="${SD}/post_job_%j.log" --wrap "bash ${LD_REPO}/scripts/cluster/lowdata_vulcan/post_run.sh ${S}")"; then
      [ -n "${DRY:-}" ] || echo "${j} $(date '+%F %T') (panseg_handoff)" > "${SD}/POST_JOB"; log "${S}: all 6 folds on Vulcan -> post controller ${j}"; left=$((left - 1))
    else log "${S}: sbatch failed -- retry next cycle"; fi
  done
  [ -n "${summary}" ] && echo "[$(date '+%F %T')] waiting:${summary}"
  [ -z "${DRY:-}" ] || { log "dry pass done"; break; }
  [ "${left}" -gt 0 ] || { log "every handed-over setting has a post controller -- exiting"; break; }
  sleep "${POLL_S}"
done
