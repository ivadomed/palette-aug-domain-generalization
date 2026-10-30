#!/usr/bin/env bash
# flatauglab arm ("flat fill + Auglab", clone of scripts/cluster/sem50/companions_vulcan.sh; METHOD auglabAugFlat_v26_6_2_train050_val000), ONE training setting, run on the Vulcan LOGIN
# node (rsync from TamIA must go through the login-node relay): (a) check on TamIA that folds 0-2 are finished (tamia_ckpt_status.sh:
# checkpoint_final + checkpoint_best, checkpoint_latest gone, final > 10 min old); (b) rsync that setting's run directory (Dataset*/ tree:
# checkpoints, plans / dataset json, debug / progress / logs; NOT the own-test predictions fold<k>/, wandb/, checkpoint_latest,
# validation niftis) from TamIA scratch into Vulcan's matching results tree and verify size + md5 of every .pth; (c) for the companions whose
# test data lives only on Vulcan, RECORD the companion's own predict wrapper / shim (RUN_JOB_PACK_DIR, explicit RUN_ID, the companion's
# standard items = its crop rules: *_uniap A-P crops, per-case crop-before-predict pancreas test sets) and submit one L40S job per fold
# command (vulcan_companion_predict_job.sh: runs it, then audits #predictions == #inputs) + one CPU eval job per companion, afterok on its
# predict jobs (vulcan_companion_eval_job.sh: the companion's standard evaluate script, then companion_audit.py); (d) status line.
#   bash scripts/cluster/flatauglab/companions_vulcan.sh <setting> [--dry-run]
#     setting = a run_ids.tsv setting (ispy2_t1wce, panseg_t2w, brats_t1n, ...). Settings without a Vulcan companion are synced only
#     (checkpoint durability: TamIA scratch is purge-on-inactivity).
#     --dry-run: no rsync, no submit; prints the rsync / sbatch commands, records the predict commands against a STUB source tree under
#                <companions dir>/dryrun/ (predict_common needs the run dir to exist) and grep-verifies them. Works before training ends.
# Idempotent: <companions dir>/<setting>/DONE stops any rerun; SUBMITTED (written before the first sbatch) stops a second submission.
# Exit 4 = connection problem (ssh / rsync transport): the poller retries next cycle without using one of the setting's attempts.
# LOCAL_FOLDS mode (2026-10-30, vulcan_move.sh): <companions dir>/<setting>/LOCAL_FOLDS (line 1 = folds trained ON VULCAN, line 2 = the Vulcan
# run dir) -> those folds' state + md5 come from the Vulcan run dir, TamIA is asked only for the other folds (none if all 3 are local), and
# the rsync excludes the local fold dirs (TamIA keeps a stale copy of them). Driven by moved_poller.sh.
# Companions running on TamIA (hanseg / pddca, amos / sliver07 fov_crop, lld-mmri-hcc / liverhccseg): launch.sh queue-companions.
set -euo pipefail
REPO=/project/aip-jcohen/paulh/mri_synthesis_project; cd "${REPO}"
CD="${FLATAUG_COMPANIONS_DIR:-/scratch/paulh/flatauglab_20261028/companions}"
RUNS="${FLATAUG_RUN_TABLE:-/scratch/paulh/flatauglab_20261028/run_ids.tsv}"
METHOD=auglabAugFlat_v26_6_2_train050_val000
TS="$(cat "${FLATAUG_TS_FILE:-/scratch/paulh/flatauglab_20261028/TS}")"   # the pack root TS (copied from TamIA <PACKS_ROOT>/TS after `launch.sh record`)
TAMIA=tamia.alliancecan.ca; SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 "${TAMIA}")
SETTING="${1:?usage: companions_vulcan.sh <setting> [--dry-run]}"; DRY=0; [ "${2:-}" = --dry-run ] && DRY=1
T=benchmark/02_tasks
declare -A TASK=([brats]=brain_tumor/brats2024-glioma [chaos]=abdomen_healthy/chaos [onh]=brain_healthy/on-harmony [openms]=brain_ms/open-ms
  [tf2]=mandible_healthy/toothfairy2 [ispy2]=breast_cancer/ispy2 [pelvic]=pelvis_healthy/totalseg-pelvic [atlas]=liver_hcc/atlas-liver-hcc
  [panseg]=pancreas_disease/pansegdata)
# Vulcan companions per setting: name | task dir | kind (breast = generated 05_23/05_24 wrapper, panseg = the companion's roster shim with
#   an explicit RUN_ID) | wrapper or shim (05_predict/) | items (= what the combined config reads) | predict time | predict mem | eval mem | eval time
comps() {
  local w; case "$1" in
    ispy2_t1wce|ispy2_t2w) [ "$1" = ispy2_t1wce ] && w=05_29_predict_ispy2_t1wce_${METHOD}.sh || w=05_30_predict_ispy2_t2w_${METHOD}.sh
      echo "duke-breast-mri|breast_cancer/duke-breast-mri|breast|${w}|t1wce_uniap precontrast_uniap|01:30:00|40G|32000M|02:00:00"
      echo "ispy1|breast_cancer/ispy1|breast|${w}|t1wce precontrast|01:30:00|40G|32000M|02:00:00"
      echo "acrin6698|breast_cancer/acrin6698|breast|${w}|dwi_uniap|01:00:00|40G|32000M|02:00:00" ;;
    panseg_t1wce|panseg_t2w)
      echo "totalsegmri-pancreas|pancreas_disease/totalsegmri-pancreas|panseg|05_01_predict_pansegdata_common.sh|t1gre t2like|01:30:00|64G|64G|02:00:00"
      echo "msd-pancreas|pancreas_disease/msd-pancreas|panseg|05_01_predict_pansegdata_common.sh|ct|03:00:00|64G|64G|03:00:00" ;;
  esac; }
SD="${CD}/${SETTING}"; mkdir -p "${SD}"
log() { echo "[$(date '+%F %T')] [${SETTING}] $*"; }
status() { printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "${SETTING}" "$1" "$2" >> "${CD}/status.tsv"; log "STATUS $1 $2"; }
[ "${DRY}" = 1 ] || [ ! -f "${SD}/DONE" ] || { log "already DONE ($(cat "${SD}/DONE")) -- nothing to do"; exit 0; }

# ---- run id + TamIA state
RID="$(awk -F'\t' -v s="${SETTING}" '$1==s {print $3}' "${RUNS}" | sort -u)"
[ "$(echo "${RID}" | grep -c .)" = 1 ] || { log "ERROR: ${SETTING} has $(echo "${RID}" | grep -c .) run ids in ${RUNS}"; exit 2; }
[[ "${RID}" == *_${METHOD}_${TS} ]] || { log "ERROR: run id ${RID} is not the flatauglab arm"; exit 2; }
K="${SETTING%%_*}"; TC="${SETTING#*_}"; DS="$(basename "${TASK[$K]}")"
[[ "${RID}" == "${DS}_${TC}_${METHOD}_${TS}" ]] || { log "ERROR: run id ${RID} does not match ${DS}/${TC}"; exit 2; }
LF=""; LRD=""
if [ -f "${SD}/LOCAL_FOLDS" ]; then LF="$(sed -n 1p "${SD}/LOCAL_FOLDS")"; LRD="$(sed -n 2p "${SD}/LOCAL_FOLDS")"
  [ "$(basename "${LRD}")" = "${RID}" ] && [ -d "${LRD}" ] || { log "ERROR: LOCAL_FOLDS run dir '${LRD}' is not ${RID}"; exit 2; }; fi
islocal() { case " ${LF} " in *" $1 "*) return 0;; esac; return 1; }
REMOTE_FOLDS=""; for k in 0 1 2; do islocal "$k" || REMOTE_FOLDS="${REMOTE_FOLDS} ${k}"; done
ST=""
if [ -n "${REMOTE_FOLDS}" ]; then
  ST="$("${SSH[@]}" 'bash -s' < scripts/cluster/flatauglab/tamia_ckpt_status.sh | awk -F'\t' -v r="${RID}" -v lf=" ${LF} " '$1=="ST" && $2==r && index(lf, " " $3 " ")==0')" \
    || { log "ERROR: TamIA status call failed (relay down? re-auth: ssh -fN ${TAMIA})"; exit 4; }
fi
LST=""; now=$(date +%s)
for k in ${LF}; do
  d="$(ls -d "${LRD}"/Dataset*/*__nnUNetPlans__3d_fullres/fold_${k} 2>/dev/null | head -1)"; st=training
  if [ -n "$d" ] && [ -f "$d/checkpoint_final.pth" ]; then st=settling
    [ -f "$d/checkpoint_best.pth" ] && [ ! -e "$d/checkpoint_latest.pth" ] && [ $(( now - $(stat -c %Y "$d/checkpoint_final.pth") )) -gt 600 ] && st=done; fi
  LST="${LST}$(printf 'ST\t%s\t%s\t%s\t%s\t%s' "${RID}" "${k}" "${st}" "${LRD}" "${d#${LRD}/}")"$'\n'
done
ALLST="$(printf '%s\n%s' "${ST}" "${LST}" | grep . || true)"
ndone="$(echo "${ALLST}" | awk -F'\t' '$4=="done" {print $3}' | sort -u | grep -c . || true)"
log "run ${RID}: folds done ${ndone}/3 ($(echo "${ALLST}" | awk -F'\t' '{printf "f%s=%s ", $3, $4}'))${LF:+ (Vulcan-trained folds: ${LF})}"
if [ "${ndone}" != 3 ]; then [ "${DRY}" = 1 ] || { log "training not complete -- nothing done"; exit 3; }; log "(dry-run: continuing anyway)"; fi
if [ -n "${REMOTE_FOLDS}" ]; then
  SRC="$(echo "${ST}" | awk -F'\t' '{print $5}' | sort -u)"
  [ "$(echo "${SRC}" | grep -c .)" = 1 ] || { log "ERROR: run dir not unique on TamIA: ${SRC}"; exit 2; }
  [[ "${SRC}" == */01_predictions/*/auglab/${RID} ]] || { log "ERROR: unexpected TamIA run dir ${SRC}"; exit 2; }
  DEST="${REPO}/${T}/${TASK[$K]}/8_results_${DS}/01_predictions/${SRC#*/01_predictions/}"
else SRC="${LRD}"; DEST="${LRD}"; fi
[ -z "${LRD}" ] || [ "$(realpath -m "${LRD}")" = "$(realpath -m "${DEST}")" ] || { log "ERROR: Vulcan run dir ${LRD} != ${DEST}"; exit 2; }
PARENT="$(dirname "${DEST}")"
# proof of the right tree: Vulcan already has <results>/01_predictions/<model>/<contrast>/ for this setting (toothfairy2 keeps only a few
# runs on Vulcan, so an existing headline run is not required)
[ -d "$(dirname "${PARENT}")" ] || { log "ERROR: $(dirname "${PARENT}") does not exist -- not the setting's results tree?"; exit 2; }
FOLDREL="$(echo "${ALLST}" | awk -F'\t' '{print $6}' | sort)"   # Dataset*/<trainer>__nnUNetPlans__3d_fullres/fold_k
TRAINER_DIR="$(echo "${FOLDREL}" | head -1 | cut -d/ -f2)"; DSDIR="$(echo "${FOLDREL}" | head -1 | cut -d/ -f1)"
log "TamIA ${SRC}"; log "Vulcan ${DEST}"

# ---- (b) rsync + verify
LEXC=(); for k in ${LF}; do LEXC+=("--exclude=/Dataset*/*/fold_${k}/"); done
RS=(rsync -a --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=30" "${LEXC[@]}" --exclude=checkpoint_latest.pth --exclude=wandb/
    --exclude='validation/*.nii.gz' --exclude='validation/*.npz' --include='/Dataset*/***' --exclude='*' "${TAMIA}:${SRC}/" "${DEST}/")
if [ "${DRY}" = 1 ]; then
  log "[dry] mkdir -p ${DEST} && ${RS[*]}"
elif [ -f "${SD}/SYNCED" ]; then log "already synced ($(head -1 "${SD}/SYNCED"))"
else
  MD=""
  if [ -n "${REMOTE_FOLDS}" ]; then
    mkdir -p "${DEST}"; log "rsync ...${LF:+ (excluding Vulcan-trained folds ${LF})}"
    rrc=0; "${RS[@]}" || rrc=$?
    case "${rrc}" in 0) ;; 10|12|30|35|255) log "rsync connection problem rc=${rrc} (retry next cycle)"; exit 4 ;; *) log "ERROR: rsync failed rc=${rrc}"; exit 5 ;; esac
    MD="$("${SSH[@]}" 'bash -s' -- md5 "${RID}" < scripts/cluster/flatauglab/tamia_ckpt_status.sh)" || { log "connection problem (md5 call; retry next cycle)"; exit 4; }
    for k in ${LF}; do MD="$(echo "${MD}" | grep -v "/fold_${k}/" || true)"; done   # TamIA's stale copy of a Vulcan-trained fold
  fi
  nrem=$(( $(echo ${REMOTE_FOLDS} | wc -w) * 2 ))
  [ "$(echo "${MD}" | grep -c '^MD5' || true)" = "${nrem}" ] || { log "ERROR: expected ${nrem} checkpoint md5s from TamIA, got: ${MD}"; exit 5; }
  bad=0
  while IFS=$'\t' read -r _ m sz p; do
    lp="${DEST}/${p#${SRC}/}"; lsz="$(stat -c %s "${lp}" 2>/dev/null || echo 0)"; lm="$(md5sum < "${lp}" | cut -c1-32)"
    [ "${lsz}" = "${sz}" ] && [ "${lm}" = "${m}" ] && log "  ok  ${p#${SRC}/} ${sz} ${m}" || { log "  BAD ${p#${SRC}/} local ${lsz}/${lm} vs ${sz}/${m}"; bad=1; }
  done < <(echo "${MD}" | grep '^MD5' || true)
  LMD=""; for k in ${LF}; do for ck in checkpoint_final checkpoint_best; do f="$(ls "${LRD}"/Dataset*/*__nnUNetPlans__3d_fullres/fold_${k}/${ck}.pth)"
    LMD="${LMD}$(printf 'MD5\t%s\t%s\t%s' "$(md5sum < "$f" | cut -c1-32)" "$(stat -c %s "$f")" "$f")"$'\n'; log "  local ${f#${LRD}/}"; done; done
  MD="$(printf '%s\n%s' "${MD}" "${LMD}" | grep '^MD5' || true)"
  [ "$(echo "${MD}" | grep -c '^MD5')" = 6 ] || { log "ERROR: expected 6 checkpoint md5s in total, got: ${MD}"; exit 5; }
  [ "${bad}" = 0 ] || { log "ERROR: checkpoint verification failed"; status sync_failed "${DEST}"; exit 5; }
  echo "${DEST} $(date '+%F %T')" > "${SD}/SYNCED"; echo "${MD}" >> "${SD}/SYNCED"; status synced "${DEST}"
fi

# ---- (c) companions
mapfile -t CROWS < <(comps "${SETTING}")
if [ ${#CROWS[@]} = 0 ]; then
  [ "${DRY}" = 1 ] || { echo "synced_only $(date '+%F %T')" > "${SD}/DONE"; status done "no Vulcan companion (checkpoints synced)"; }
  exit 0
fi
if [ "${DRY}" = 0 ] && [ -f "${SD}/SUBMITTED" ]; then log "companions already submitted ($(cat "${SD}/SUBMITTED")) -- refusing a second submission"; exit 6; fi
MODE_TAG=real; [ "${DRY}" = 1 ] && MODE_TAG=dryrun
STAMP="$(date +%Y%m%d_%H%M%S)"
if [ "${DRY}" = 1 ]; then   # stub source run dir (Dataset*/<trainer> dir names as on TamIA) so predict_common / the roster helpers resolve
  STUB="${CD}/dryrun/${SETTING}/src_predictions"; rm -rf "${CD}/dryrun/${SETTING}"
  STUB_RUN="${STUB}/${SRC#*/01_predictions/}"; mkdir -p "${STUB_RUN}/${DSDIR}/${TRAINER_DIR}"
fi
# record + verify every companion first (nothing submitted unless ALL companions record cleanly)
declare -a RECS=()
for row in "${CROWS[@]}"; do IFS='|' read -r CN CT KIND W ITEMS PT PM EM ET <<<"${row}"
  RD="${SD}/${CN}/${MODE_TAG}_${STAMP}"; mkdir -p "${RD}"; S="${T}/${CT}/5_scripts_${CN}"
  (
    set -euo pipefail
    unset PREDICTIONS_ROOT METRICS_ROOT nnUNet_raw nnUNet_preprocessed RUN_JOB_INLINE
    if [ "${DRY}" = 1 ]; then
      export PREDICTIONS_ROOT="${CD}/dryrun/${SETTING}/${CN}_predictions"
      case "${KIND}" in breast) export ISPY2_PREDICTIONS_ROOT="${STUB}";; panseg) export PANSEG_PREDICTIONS_ROOT="${STUB}";; esac
    fi
    export RUN_JOB_PACK_DIR="${RD}/pack"
    case "${KIND}" in
      breast) bash "${S}/05_predict/${W}" "${RID}" all ${ITEMS} ;;
      panseg)   # = run_all_predict_cross_common.sh's per-run body, with the explicit flatauglab RUN_ID instead of the source pin file
        source "${S}/00_utils/env.sh"; source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_00_utils/roster_lib.sh"
        rd="$(roster_src_run_dir PANSEG "${TC}" auglab "${RID}")"; roster_run_trainer_id "${rd}" || { echo "no trained model under ${rd}"; exit 1; }
        export PANSEG_TRAINING_CONTRAST="${TC}" PANSEG_DATASET_ID="${RUN_DATASET_ID}"
        METHOD="${METHOD}"; TRAINER="${RUN_TRAINER}"; CATEGORY=auglab
        source "${S}/05_predict/${W}" "${RID}" all ;;
    esac
  ) > "${RD}/record.log" 2>&1 || { log "ERROR: recording ${CN} failed (${RD}/record.log)"; tail -5 "${RD}/record.log"; exit 7; }
  # verify the recorded fold commands
  n="$(grep -c . "${RD}/pack/index.tsv" 2>/dev/null || echo 0)"; [ "${n}" = 3 ] || { log "ERROR: ${CN}: ${n} fold commands recorded (expect 3)"; exit 7; }
  case "${SETTING}" in ispy2_t1wce) did=100;; ispy2_t2w) did=101;; panseg_t1wce) did=150;; panseg_t2w) did=151;; esac
  RAW="$( unset nnUNet_raw PREDICTIONS_ROOT; source "${S}/00_utils/env.sh" >/dev/null 2>&1; echo "${nnUNet_raw}" )"
  expres="${DEST}"; [ "${DRY}" = 1 ] && expres="${STUB_RUN}"
  for k in 0 1 2; do
    f="$(awk -F'\t' -v k="${k}" '$3 ~ ("_fold" k "$") {print $1}' "${RD}/pack/index.tsv")"
    [ -f "${f}" ] || { log "ERROR: ${CN}: no fold-${k} command"; exit 7; }
    c="$(sed -e "s/\\\\'/'/g" -e 's/\\n/\n/g' "${f}")"
    chk() { printf '%s\n' "${c}" | grep -qF -- "$1" || { log "ERROR: ${CN} fold ${k}: command lacks '$1' (${f})"; exit 7; }; }
    res="$(printf '%s\n' "${c}" | sed -n "s/^ *export nnUNet_results='\(.*\)'\$/\1/p" | head -1)"
    [ "$(realpath -m "${res}")" = "$(realpath -m "${expres}")" ] || { log "ERROR: ${CN} fold ${k}: nnUNet_results '${res}' is not ${expres}"; exit 7; }
    chk "-tr ${TRAINER_DIR%%__*}"; chk "-f ${k} --disable_tta -chk checkpoint_best.pth"; chk "-d ${did} "
    for it in ${ITEMS}; do chk "-i '${RAW}/imagesTs_${it}' -o '"; chk "/${RID}/fold${k}/${it}'"; done
    ni="$(printf '%s\n' "${c}" | grep -o "nnUNetv2_predict -i " | wc -l)"; nw="$(echo ${ITEMS} | wc -w)"
    [ "${ni}" = "${nw}" ] || { log "ERROR: ${CN} fold ${k}: ${ni} predict calls for ${nw} items"; exit 7; }
  done
  log "${CN}: 3 fold commands verified (run ${RID}, -d ${did}, items ${ITEMS}, checkpoint_best, no TTA) -> ${RD}/pack"
  RECS+=("${row}|${RD}")
done

# submit (or print)
[ "${DRY}" = 1 ] || echo "submitting $(date '+%F %T')" > "${SD}/SUBMITTED"
SUB() { if [ "${DRY}" = 1 ]; then echo "[dry] sbatch $*" >&2; echo "DRY${RANDOM}"; else sbatch --parsable "$@"; fi; }
alljobs=""
for rec in "${RECS[@]}"; do IFS='|' read -r CN CT KIND W ITEMS PT PM EM ET RD <<<"${rec}"
  ids=""
  while IFS=$'\t' read -r f _log name _done; do
    j="$(SUB --account=aip-jcohen --gres=gpu:l40s:1 --cpus-per-task=8 --mem="${PM}" --time="${PT}" --job-name="flataugc_${CN}_${SETTING}_${name##*_}" \
         --output="${RD}/predict_${name##*_}_%j.out" scripts/cluster/flatauglab/vulcan_companion_predict_job.sh "${f}")"
    ids="${ids:+${ids}:}${j}"; sleep 2
  done < "${RD}/pack/index.tsv"
  e="$(SUB --account=aip-jcohen --dependency=afterok:"${ids}" --kill-on-invalid-dep=yes --cpus-per-task=4 --mem="${EM}" --time="${ET}" \
       --job-name="flataugc_eval_${CN}_${SETTING}" --output="${RD}/eval_%j.out" \
       scripts/cluster/flatauglab/vulcan_companion_eval_job.sh "${CN}" "${RID}" "${TC}" ${ITEMS})"
  echo -e "predict\t${ids}\neval\t${e}" > "${RD}/JOBS"
  log "${CN}: predict ${ids//:/ } -> eval ${e}"; alljobs="${alljobs} ${CN}:${ids}->${e}"; sleep 2
done
if [ "${DRY}" = 0 ]; then
  echo "submitted ${alljobs} $(date '+%F %T')" > "${SD}/SUBMITTED"; echo "submitted ${alljobs}" > "${SD}/DONE"
  status submitted "${alljobs}"
else log "dry-run complete (nothing synced or submitted)"; fi
