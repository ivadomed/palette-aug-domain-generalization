#!/usr/bin/env bash
# VULCAN login-node poller of the SLAug arm (slaug): brings every finished fold to Vulcan and queues ONE post-training controller per
# setting (post_run.sh: own test set + external cohorts, predict + evaluate on Vulcan only, audit vs the headline run). Clone of the
# flatauglab pollers (companions_poller.sh + moved_poller.sh + vulcan_post_poller.sh) folded into one loop, driven by the run table:
#   vulcan folds    : trained in the Vulcan run dir itself -> done when final + best exist, latest gone, final > 10 min old;
#   tamia folds     : checked on TamIA (ssh tamia.alliancecan.ca from this login node), fold dir + plans/dataset json rsynced into the SAME
#                     Vulcan run dir, size + md5 of checkpoint_final / checkpoint_best verified -> <fold dir>/.slaug_synced_from_tamia;
#   killarney folds : pushed by `killarney.sh syncback` (klogin02), which writes <fold dir>/.slaug_synced_from_killarney after its md5 check.
# When all 3 folds of a setting are on Vulcan: <root>/post/<setting>/SYNCED (run dir + md5s), then sbatch post_run.sh once (POST_JOB marker;
# a FAILED controller is left for a human: delete POST_JOB to resubmit). One instance (flock). Exits when every setting has POST_JOB, on
# <root>/post/STOP_POST, or after MAX_DAYS (default 12). DRY=1: one pass, prints what it would rsync / submit, changes nothing.
#   start : systemd-run --user --scope --unit=slaug-post -- nohup bash scripts/cluster/slaug/post_poller.sh >> <root>/post/poller.log 2>&1 &
#   stop  : touch <root>/post/STOP_POST
set -uo pipefail
REPO=/project/aip-jcohen/paulh/mri_synthesis_project; cd "${REPO}"
R="${SLAUG_ROOT:-$(cat /scratch/paulh/_slaug_root.txt)}"; RT="${R}/run_table.tsv"; PD="${R}/post"; mkdir -p "${PD}"
TAMIA=tamia.alliancecan.ca; SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=30 "${TAMIA}")
POLL_S="${POLL_S:-1200}"; MAX_DAYS="${MAX_DAYS:-12}"; T=benchmark/02_tasks
if [ -z "${DRY:-}" ]; then exec 8> "${PD}/poller.lock"; flock -n 8 || { echo "[slaug-post] another instance holds the lock"; exit 1; }; fi
log() { echo "[$(date '+%F %T')] [slaug-post] $*"; }
run() { if [ -n "${DRY:-}" ]; then echo "[dry] $*"; else "$@"; fi; }
# Vulcan run dir of a setting = what its train wrapper computes (PREDICTIONS_ROOT/MODEL_TYPE/TRAINING_CONTRAST/auglab/RID); checked
# 2026-11-02 equal to nnUNet_results of the recorded fold commands on Vulcan (onh_T2w, openms_flair, atlas_t1w) and TamIA (brats_t1n,
# onh_T1w, onh_T2w, with tamia_env*.sh)
rundir() {  # $1 scripts dir (under benchmark/02_tasks), $2 train wrapper, $3 RID, [$4 tamia env -> TamIA formula, evaluated there]
  local w="${T}/$1/04_train/$2" envf
  envf="$(grep -o 'source "$(dirname "$0")/../00_utils/env[a-z0-9_]*\.sh"' "${w}" | sed 's|.*/00_utils/||; s|"$||')"
  if [ -z "${4:-}" ]; then ( source "${T}/$1/00_utils/${envf}" >/dev/null 2>&1; echo "${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab/$3" )
  else "${SSH[@]}" "bash -lc 'cd ${REPO} && export SCRATCH=/scratch/p/paulh && source ${T}/$1/00_utils/${envf} >/dev/null 2>&1 && source scripts/cluster/$4 >/dev/null 2>&1 && echo \${PREDICTIONS_ROOT}/\${MODEL_TYPE}/\${TRAINING_CONTRAST}/auglab/$3'"; fi; }
fstate() { local d="$1"; [ -f "$d/checkpoint_final.pth" ] && [ -f "$d/checkpoint_best.pth" ] && [ ! -e "$d/checkpoint_latest.pth" ] \
  && [ $(( $(date +%s) - $(stat -c %Y "$d/checkpoint_final.pth") )) -gt 600 ] && echo done || echo training; }
MD5Q='for f in checkpoint_final.pth checkpoint_best.pth; do echo $f $(stat -c %s $f) $(md5sum < $f | cut -c1-32); done'
mapfile -t SETTINGS < <(awk -F'\t' 'NR>1 && $1 !~ /_parent$/ {print $1}' "${RT}" | awk '!s[$0]++')
declare -A TENV=([brats]=tamia_env.sh [chaos]=tamia_env_chaos.sh [onh]=tamia_env_onharmony.sh [openms]=tamia_env_openms.sh [tf2]=tamia_env_toothfairy2.sh
  [ispy2]=tamia_env_ispy2.sh [pelvic]=tamia_env_totalseg-pelvic.sh [atlas]=tamia_env_atlas-liver-hcc.sh [panseg]=tamia_env_pansegdata.sh)
t_end=$(( $(date +%s) + MAX_DAYS * 86400 )); log "start pid $$, ${#SETTINGS[@]} settings, root ${R}, HEAD $(git rev-parse --short HEAD)${DRY:+ (DRY)}"
while :; do
  [ -f "${PD}/STOP_POST" ] && { log "STOP_POST -- exiting"; break; }
  [ "$(date +%s)" -lt "${t_end}" ] || { log "max lifetime -- exiting"; break; }
  left=0; summary=""
  for s in "${SETTINGS[@]}"; do
    SD="${PD}/${s}"; mkdir -p "${SD}"; [ -f "${SD}/POST_JOB" ] && continue; left=$((left + 1))
    read -r rid tw sd < <(awk -F'\t' -v s="${s}" 'NR>1 && $1==s {print $16, $17, $18; exit}' "${RT}")
    VRD="$(rundir "${sd}" "${tw}" "${rid}")"; nd=0; st_=""
    while IFS=$'\t' read -r f cl; do
      fd="$(ls -d "${VRD}"/Dataset*/*__nnUNetPlans__3d_fullres/fold_${f} 2>/dev/null | head -1)"
      case "${cl}" in
        vulcan) [ -n "${fd}" ] && [ "$(fstate "${fd}")" = done ] && nd=$((nd + 1)) || st_="${st_} f${f}:vulcan" ;;
        killarney) [ -n "${fd}" ] && [ -f "${fd}/.slaug_synced_from_killarney" ] && nd=$((nd + 1)) || st_="${st_} f${f}:killarney" ;;
        tamia)
          if [ -n "${fd}" ] && [ -f "${fd}/.slaug_synced_from_tamia" ]; then nd=$((nd + 1)); continue; fi
          TRD="$(rundir "${sd}" "${tw}" "${rid}" "${TENV[${s%%_*}]}")" || { st_="${st_} f${f}:tamia(ssh?)"; continue; }
          tfd="$("${SSH[@]}" "ls -d ${TRD}/Dataset*/*__nnUNetPlans__3d_fullres/fold_${f} 2>/dev/null | head -1")" || { st_="${st_} f${f}:tamia(ssh?)"; continue; }
          [ -n "${tfd}" ] || { st_="${st_} f${f}:tamia(no fold dir yet under ${TRD})"; continue; }   # path shown: a wrong derivation is visible
          tst="$("${SSH[@]}" "d=${tfd}; [ -f \$d/checkpoint_final.pth ] && [ -f \$d/checkpoint_best.pth ] && [ ! -e \$d/checkpoint_latest.pth ] && [ \$(( \$(date +%s) - \$(stat -c %Y \$d/checkpoint_final.pth) )) -gt 600 ] && echo done || echo training")" || tst=ssh
          [ "${tst}" = done ] || { st_="${st_} f${f}:tamia(${tst})"; continue; }
          vfd="${VRD}/${tfd#${TRD}/}"; log "${s} f${f}: TamIA ${tfd} -> ${vfd}"
          run mkdir -p "${vfd}"
          run rsync -a -e "ssh -o BatchMode=yes" --include='*.json' --exclude='*' "${TAMIA}:$(dirname "${tfd}")/" "$(dirname "${vfd}")/" || continue
          run rsync -a -e "ssh -o BatchMode=yes" --exclude=wandb/ --exclude='validation/*.nii.gz' --exclude='validation/*.npz' "${TAMIA}:${tfd}/" "${vfd}/" || continue
          [ -z "${DRY:-}" ] || continue
          L="$(cd "${vfd}" && bash -c "${MD5Q}")"; Rm="$("${SSH[@]}" "cd ${tfd} && ${MD5Q}")" || continue
          if [ "$L" = "$Rm" ]; then printf '%s\n%s\n' "$(date '+%F %T') tamia ${tfd}" "$L" > "${vfd}/.slaug_synced_from_tamia"; nd=$((nd + 1)); log "${s} f${f}: synced + md5 OK"
          else log "${s} f${f}: md5 MISMATCH (retry next cycle)"; fi ;;
      esac
    done < <(awk -F'\t' -v s="${s}" 'NR>1 && $1==s {print $2"\t"$8}' "${RT}")
    if [ "${nd}" != 3 ]; then summary="${summary} ${s}(${nd}/3:${st_# })"; continue; fi
    if [ ! -f "${SD}/SYNCED" ]; then
      M=""; for f in 0 1 2; do fd="$(ls -d "${VRD}"/Dataset*/*__nnUNetPlans__3d_fullres/fold_${f} | head -1)"; M="${M}$(cd "${fd}" && bash -c "${MD5Q}" | sed "s|^|fold_${f} |")"$'\n'; done
      [ -n "${DRY:-}" ] || printf '%s %s\n%s' "${VRD}" "$(date '+%F %T')" "${M}" > "${SD}/SYNCED"; log "${s}: all 3 folds on Vulcan -> SYNCED"
    fi
    if j="$(run sbatch --parsable --account=aip-jcohen --cpus-per-task=16 --mem=96G --time=3-00:00:00 --job-name="slaugpost_${s}" \
            --output="${SD}/post_job_%j.log" --wrap "bash scripts/cluster/slaug/post_run.sh ${s}")"; then
      [ -n "${DRY:-}" ] || echo "${j} $(date '+%F %T')" > "${SD}/POST_JOB"; log "${s}: controller ${j}"; left=$((left - 1))
    else log "${s}: sbatch failed -- retry next cycle"; fi
    sleep 2
  done
  log "waiting:${summary:- (none)}"
  [ -z "${DRY:-}" ] || { log "dry pass done"; break; }
  [ "${left}" -gt 0 ] || { log "every setting has a controller -- exiting"; break; }
  for _ in $(seq $(( POLL_S / 30 ))); do [ -f "${PD}/STOP_POST" ] && break; sleep 30; done
done
log "exit"
