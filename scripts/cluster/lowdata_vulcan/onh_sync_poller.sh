#!/usr/bin/env bash
# VULCAN login-node poller of the TamIA-trained on-harmony low-data runs (tamia_onh.sh): every POLL_S checks each fold of each
# onh_<contrast> run of the Vulcan run table on TamIA (done = checkpoint_final + checkpoint_best, no checkpoint_latest, final > 10 min
# old), rsyncs the fold dir + plans/dataset json into the SAME Vulcan run dir post_run.sh reads (PREDICTIONS_ROOT/on_harmony_model/<c>/
# auglab/<RID>), checks size + md5 of checkpoint_final / checkpoint_best (-> <fold dir>/.lowdata_synced_from_tamia), and when all 6
# folds (2 arms x 3) of a setting are on Vulcan submits post_run.sh <setting> once (<post dir>/<setting>/POST_JOB). Same recipe as
# scripts/cluster/ginipa/post_poller.sh. One instance (flock); exits when every setting has POST_JOB, on <post dir>/STOP_ONH, or after
# MAX_DAYS (default 8). DRY=1: one pass, prints what it would do.
#   start: systemd-run --user --scope --unit=lowdata-onh -- nohup bash scripts/cluster/lowdata_vulcan/onh_sync_poller.sh \
#            >> /scratch/paulh/lowdata_vulcan/post/onh_poller.log 2>&1 &
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
source scripts/cluster/lowdata_vulcan/common.sh
TAMIA=tamia.alliancecan.ca; SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=30 "${TAMIA}")
TREPO=/project/aip-jcohen/paulh/mri_synthesis_project; TPRED=/scratch/p/paulh/on-harmony/8_results/01_predictions/on_harmony_model
POLL_S="${POLL_S:-1200}"; MAX_DAYS="${MAX_DAYS:-8}"; mkdir -p "${LD_POST}"
if [ -z "${DRY:-}" ]; then exec 8> "${LD_POST}/onh_poller.lock"; flock -n 8 || { echo "another onh poller holds the lock"; exit 1; }; fi
log() { echo "[$(date '+%F %T')] [onh-poll] $*"; }
run() { if [ -n "${DRY:-}" ]; then echo "[dry] $*"; else "$@"; fi; }
MD5Q='for f in checkpoint_final.pth checkpoint_best.pth; do echo $f $(stat -c %s $f) $(md5sum < $f | cut -c1-32); done'
t_end=$(( $(date +%s) + MAX_DAYS * 86400 )); log "start pid $$ HEAD $(git rev-parse --short HEAD)${DRY:+ (DRY)}"
while :; do
  [ -f "${LD_POST}/STOP_ONH" ] && { log "STOP_ONH -- exiting"; break; }
  [ "$(date +%s)" -lt "${t_end}" ] || { log "max lifetime -- exiting"; break; }
  left=0; summary=""
  for S in onh_T1w onh_T2w onh_dwi_ap; do
    ld_setting "${S}"; SD="${LD_POST}/${S}"; mkdir -p "${SD}"; [ -f "${SD}/POST_JOB" ] && continue; left=$((left + 1)); nd=0; st=""
    for ARM in flat texture; do
      rid="$(ld_rid "${S}" "${ARM}")"; [ -n "${rid}" ] || { st="${st} ${ARM}:no-run"; continue; }
      for f in 0 1 2; do
        vfd="$(ls -d "${LD_PRED_ROOT}/${rid}"/Dataset*/*__nnUNetPlans__3d_fullres/fold_${f} 2>/dev/null | head -1)"
        [ -n "${vfd}" ] && [ -f "${vfd}/.lowdata_synced_from_tamia" ] && { nd=$((nd + 1)); continue; }
        trd="${TPRED}/${LD_TC}/auglab/${rid}"
        tfd="$("${SSH[@]}" "ls -d ${trd}/Dataset*/*__nnUNetPlans__3d_fullres/fold_${f} 2>/dev/null | head -1")" || { st="${st} ${ARM}f${f}:ssh?"; continue; }
        [ -n "${tfd}" ] || { st="${st} ${ARM}f${f}:no-dir"; continue; }
        tst="$("${SSH[@]}" "d=${tfd}; [ -f \$d/checkpoint_final.pth ] && [ -f \$d/checkpoint_best.pth ] && [ ! -e \$d/checkpoint_latest.pth ] && [ \$(( \$(date +%s) - \$(stat -c %Y \$d/checkpoint_final.pth) )) -gt 600 ] && echo done || echo training")" || tst=ssh
        [ "${tst}" = done ] || { st="${st} ${ARM}f${f}:${tst}"; continue; }
        vfd="${LD_PRED_ROOT}/${rid}/${tfd#${trd}/}"; log "${S} ${ARM} f${f}: ${tfd} -> ${vfd}"
        run mkdir -p "${vfd}"
        run rsync -a -e "ssh -o BatchMode=yes" --include='*.json' --exclude='*' "${TAMIA}:$(dirname "${tfd}")/" "$(dirname "${vfd}")/" || continue
        run rsync -a -e "ssh -o BatchMode=yes" --exclude=wandb/ --exclude='validation/*.nii.gz' --exclude='validation/*.npz' "${TAMIA}:${tfd}/" "${vfd}/" || continue
        [ -z "${DRY:-}" ] || continue
        L="$(cd "${vfd}" && bash -c "${MD5Q}")"; Rm="$("${SSH[@]}" "cd ${tfd} && ${MD5Q}")" || continue
        if [ "${L}" = "${Rm}" ]; then printf '%s\n%s\n' "$(date '+%F %T') tamia ${tfd}" "${L}" > "${vfd}/.lowdata_synced_from_tamia"; nd=$((nd + 1)); log "${S} ${ARM} f${f}: synced, md5 OK"
        else log "${S} ${ARM} f${f}: md5 MISMATCH (retry next cycle)"; fi
      done
    done
    if [ "${nd}" != 6 ]; then summary="${summary} ${S}(${nd}/6:${st# })"; continue; fi
    if j="$(run sbatch --parsable --account=aip-jcohen --cpus-per-task=16 --mem=96G --time=20:00:00 --job-name="ldpost_${S}" \
            --output="${SD}/post_job_%j.log" --wrap "bash ${LD_REPO}/scripts/cluster/lowdata_vulcan/post_run.sh ${S}")"; then
      [ -n "${DRY:-}" ] || echo "${j} $(date '+%F %T')" > "${SD}/POST_JOB"; log "${S}: all 6 folds on Vulcan -> controller ${j}"; left=$((left - 1))
    else log "${S}: sbatch failed -- retry next cycle"; fi
  done
  log "waiting:${summary:- (none)}"
  [ -z "${DRY:-}" ] || { log "dry pass done"; break; }
  [ "${left}" -gt 0 ] || { log "every setting has a controller -- exiting"; break; }
  sleep "${POLL_S}"
done
