#!/usr/bin/env bash
# KILLARNEY login-node relay variant of onh_sync_poller.sh, for when Vulcan's ssh master to TamIA is down (it needs an interactive Duo
# login) but Killarney holds masters to BOTH clusters (bash ~/bin/ssh-master status). Same logic: every POLL_S, for each fold of the
# onh_<contrast> runs in Vulcan's run table (8_results_on-harmony/_launch/RUN_IDS_lowdata_vulcan.tsv), checks on TamIA that it is done
# (checkpoint_final + checkpoint_best, no checkpoint_latest, final > 10 min old), copies fold dir + plans/dataset json TamIA ->
# Killarney stage -> the Vulcan run dir post_run.sh reads, verifies size + md5 of checkpoint_final / checkpoint_best on Vulcan against
# TamIA (-> <fold dir>/.lowdata_synced_from_tamia on Vulcan), and once all 6 folds of a setting are on Vulcan submits post_run.sh
# <setting> on Vulcan (marker /scratch/paulh/lowdata_vulcan/post/<setting>/POST_JOB on Vulcan). Self-contained: run it from a copy
# (Killarney's checkout is bundle-synced and may not carry this file). One instance (flock); exits when every setting has POST_JOB,
# on <stage>/STOP_ONH, or after MAX_DAYS (default 8). DRY=1: one pass, prints what it would do.
#   start (klogin node holding both masters):
#     systemd-run --user --scope --unit=lowdata-onh-relay -- nohup bash <copy of this file> >> <stage>/relay.log 2>&1 &
set -uo pipefail
VREPO=/project/aip-jcohen/paulh/mri_synthesis_project
VPOST=/scratch/paulh/lowdata_vulcan/post
VTAB=${VREPO}/benchmark/02_tasks/brain_healthy/on-harmony/8_results_on-harmony/_launch/RUN_IDS_lowdata_vulcan.tsv
VPRED=${VREPO}/benchmark/02_tasks/brain_healthy/on-harmony/8_results_on-harmony/01_predictions/on_harmony_model
TPRED=/scratch/p/paulh/on-harmony/8_results/01_predictions/on_harmony_model
STAGE="${LDR_STAGE:-${SCRATCH:-/scratch/${USER}}/lowdata_onh_stage}"; mkdir -p "${STAGE}"
O=(-o BatchMode=yes -o ConnectTimeout=30)
V() { ssh -n "${O[@]}" vulcan.alliancecan.ca "$@"; }
T() { ssh -n "${O[@]}" tamia.alliancecan.ca "$@"; }
POLL_S="${POLL_S:-1200}"; MAX_DAYS="${MAX_DAYS:-8}"
if [ -z "${DRY:-}" ]; then exec 8> "${STAGE}/relay.lock"; flock -n 8 || { echo "another relay holds the lock"; exit 1; }; fi
log() { echo "[$(date '+%F %T')] [onh-relay] $*"; }
run() { if [ -n "${DRY:-}" ]; then echo "[dry] $*"; else "$@"; fi; }
MD5Q='for f in checkpoint_final.pth checkpoint_best.pth; do echo $f $(stat -c %s $f) $(md5sum < $f | cut -c1-32); done'
t_end=$(( $(date +%s) + MAX_DAYS * 86400 )); log "start pid $$ host $(hostname)${DRY:+ (DRY)}"
while :; do
  [ -f "${STAGE}/STOP_ONH" ] && { log "STOP_ONH -- exiting"; break; }
  [ "$(date +%s)" -lt "${t_end}" ] || { log "max lifetime -- exiting"; break; }
  V "cat ${VTAB}" > "${STAGE}/table.tsv" 2>/dev/null || { log "vulcan ssh failed"; sleep "${POLL_S}"; continue; }
  left=0; summary=""
  for S in onh_T1w onh_T2w onh_dwi_ap; do
    C="${S#onh_}"
    V "test -f ${VPOST}/${S}/POST_JOB" && continue; left=$((left + 1)); nd=0; st=""
    for ARM in flat texture; do
      rid="$(awk -F'\t' -v s="${S}" -v a="${ARM}" '$1==s && $2==a {r=$4} END {print r}' "${STAGE}/table.tsv")"
      [ -n "${rid}" ] || { st="${st} ${ARM}:no-run"; continue; }
      vrd="${VPRED}/${C}/auglab/${rid}"; trd="${TPRED}/${C}/auglab/${rid}"
      for f in 0 1 2; do
        V "ls ${vrd}/Dataset*/*__nnUNetPlans__3d_fullres/fold_${f}/.lowdata_synced_from_tamia" >/dev/null 2>&1 && { nd=$((nd + 1)); continue; }
        tfd="$(T "ls -d ${trd}/Dataset*/*__nnUNetPlans__3d_fullres/fold_${f} 2>/dev/null | head -1")" || { st="${st} ${ARM}f${f}:ssh?"; continue; }
        [ -n "${tfd}" ] || { st="${st} ${ARM}f${f}:no-dir"; continue; }
        tst="$(T "d=${tfd}; [ -f \$d/checkpoint_final.pth ] && [ -f \$d/checkpoint_best.pth ] && [ ! -e \$d/checkpoint_latest.pth ] && [ \$(( \$(date +%s) - \$(stat -c %Y \$d/checkpoint_final.pth) )) -gt 600 ] && echo done || echo training")" || tst=ssh
        [ "${tst}" = done ] || { st="${st} ${ARM}f${f}:${tst}"; continue; }
        rel="${tfd#${trd}/}"; sd="${STAGE}/${rid}/${rel}"; vfd="${vrd}/${rel}"
        log "${S} ${ARM} f${f}: tamia:${tfd} -> stage -> vulcan:${vfd}"
        run mkdir -p "${sd}"
        run rsync -a -e "ssh ${O[*]}" --include='*.json' --exclude='*' "tamia.alliancecan.ca:$(dirname "${tfd}")/" "$(dirname "${sd}")/" || continue
        run rsync -a -e "ssh ${O[*]}" --exclude=wandb/ --exclude='validation/*.nii.gz' --exclude='validation/*.npz' "tamia.alliancecan.ca:${tfd}/" "${sd}/" || continue
        run V "mkdir -p ${vfd}" || continue
        run rsync -a -e "ssh ${O[*]}" --include='*.json' --exclude='*' "$(dirname "${sd}")/" "vulcan.alliancecan.ca:$(dirname "${vfd}")/" || continue
        run rsync -a -e "ssh ${O[*]}" "${sd}/" "vulcan.alliancecan.ca:${vfd}/" || continue
        [ -z "${DRY:-}" ] || continue
        L="$(V "cd ${vfd} && ${MD5Q}")" || continue; Rm="$(T "cd ${tfd} && ${MD5Q}")" || continue
        if [ "${L}" = "${Rm}" ]; then
          printf '%s\n%s\n' "$(date '+%F %T') tamia ${tfd} (relay $(hostname))" "${L}" | ssh "${O[@]}" vulcan.alliancecan.ca "cat > ${vfd}/.lowdata_synced_from_tamia" \
            && { nd=$((nd + 1)); log "${S} ${ARM} f${f}: synced, md5 OK"; }
        else log "${S} ${ARM} f${f}: md5 MISMATCH (retry next cycle)"; fi
      done
    done
    if [ "${nd}" != 6 ]; then summary="${summary} ${S}(${nd}/6:${st# })"; continue; fi
    if j="$(run V "mkdir -p ${VPOST}/${S} && cd ${VREPO} && sbatch --parsable --account=aip-jcohen --cpus-per-task=16 --mem=96G --time=20:00:00 --job-name=ldpost_${S} --output=${VPOST}/${S}/post_job_%j.log --wrap 'bash ${VREPO}/scripts/cluster/lowdata_vulcan/post_run.sh ${S}'")"; then
      [ -n "${DRY:-}" ] || V "echo '${j} $(date '+%F %T') relay' > ${VPOST}/${S}/POST_JOB"; log "${S}: all 6 folds on Vulcan -> controller ${j}"; left=$((left - 1))
    else log "${S}: sbatch failed -- retry next cycle"; fi
  done
  log "waiting:${summary:- (none)}"
  [ -z "${DRY:-}" ] || { log "dry pass done"; break; }
  [ "${left}" -gt 0 ] || { log "every setting has a controller -- exiting"; break; }
  sleep "${POLL_S}"
done
