#!/usr/bin/env bash
# Third Vulcan login-node poller of the flatauglab arm (2026-10-30), for the settings whose folds were MOVED from TamIA to Vulcan by
# vulcan_move.sh (packs B2 + B3, then G1-G3 + B1): onh_T2w, openms_t1w, atlas_t1w, panseg_t2w (all 3 folds on Vulcan) and panseg_t1wce (fold 2 on Vulcan,
# folds 0-1 still in TamIA pack B1). companions_poller.sh never fires for these (it waits for 3/3 folds done ON TAMIA), so every POLL_S
# (default 1800 s) this runs companions_vulcan.sh <setting> in its LOCAL_FOLDS mode (<companions dir>/<setting>/LOCAL_FOLDS, written here):
# Vulcan-trained folds are checked + md5'd in place, the TamIA folds (panseg_t1wce 0-1) are rsynced as usual; it writes SYNCED (-> the
# running vulcan_post_poller.sh submits the own-test + HCC controller) and, for panseg_*, submits the totalsegmri / msd-pancreas companions.
# Same retry rules as companions_poller.sh (rc 3 = training not complete, 4 / 124 = connection: no attempt used). One instance (flock).
# Exits when every moved setting is DONE / FAILED, on STOP_MOVED, or after MAX_DAYS (default 9).
# 2026-10-30 evening (second move, packs G1-G3 -> Vulcan, B1 -> Killarney): + brats_t1n/t2w/t2f/t1c (Vulcan), onh_T1w, openms_flair
# (Killarney, synced back into the Vulcan run dir by `PACK=B1 killarney_move.sh syncback`), and panseg_t1wce is now 0 1 2 (fold 2 Vulcan,
# folds 0-1 Killarney -> synced back): nothing of this arm is left on TamIA. The MOVED array is read once at start: restart the poller
# (STOP_MOVED, wait for "exit", rm STOP_MOVED, start) after changing it; settings already DONE are skipped.
#   start : systemd-run --user --scope --unit=flatauglab-moved -- nohup bash scripts/cluster/flatauglab/moved_poller.sh \
#             >> /scratch/paulh/flatauglab_20261028/companions/moved_poller.log 2>&1 &
#   stop  : touch /scratch/paulh/flatauglab_20261028/companions/STOP_MOVED
set -uo pipefail
REPO=/project/aip-jcohen/paulh/mri_synthesis_project; cd "${REPO}"
CD="${FLATAUG_COMPANIONS_DIR:-/scratch/paulh/flatauglab_20261028/companions}"
RUNS="${FLATAUG_RUN_TABLE:-/scratch/paulh/flatauglab_20261028/run_ids.tsv}"
TS="$(cat /scratch/paulh/flatauglab_20261028/TS)"; METHOD=auglabAugFlat_v26_6_2_train050_val000
POLL_S="${POLL_S:-1800}"; MAX_TRIES="${MAX_TRIES:-4}"; MAX_DAYS="${MAX_DAYS:-9}"
# setting | Vulcan-trained folds | Vulcan run dir
T="${REPO}/benchmark/02_tasks"
MOVED=(
"onh_T2w|0 1 2|${T}/brain_healthy/on-harmony/8_results_on-harmony/01_predictions/on_harmony_model/T2w/auglab/on-harmony_T2w_${METHOD}_${TS}"
"openms_t1w|0 1 2|${T}/brain_ms/open-ms/8_results_open-ms/01_predictions/open_ms_model/t1w/auglab/open-ms_t1w_${METHOD}_${TS}"
"atlas_t1w|0 1 2|${T}/liver_hcc/atlas-liver-hcc/8_results_atlas-liver-hcc/01_predictions/atlas_liver_hcc_model/t1w/auglab/atlas-liver-hcc_t1w_${METHOD}_${TS}"
"panseg_t2w|0 1 2|${T}/pancreas_disease/pansegdata/8_results_pansegdata/01_predictions/pansegdata_model/t2w/auglab/pansegdata_t2w_${METHOD}_${TS}"
"panseg_t1wce|0 1 2|${T}/pancreas_disease/pansegdata/8_results_pansegdata/01_predictions/pansegdata_model/t1wce/auglab/pansegdata_t1wce_${METHOD}_${TS}"
"brats_t1n|0 1 2|${T}/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t1n/auglab/brats2024-glioma_t1n_${METHOD}_${TS}"
"brats_t2w|0 1 2|${T}/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2w/auglab/brats2024-glioma_t2w_${METHOD}_${TS}"
"brats_t2f|0 1 2|${T}/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t2f/auglab/brats2024-glioma_t2f_${METHOD}_${TS}"
"brats_t1c|0 1 2|${T}/brain_tumor/brats2024-glioma/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model/t1c/auglab/brats2024-glioma_t1c_${METHOD}_${TS}"
"onh_T1w|0 1 2|${T}/brain_healthy/on-harmony/8_results_on-harmony/01_predictions/on_harmony_model/T1w/auglab/on-harmony_T1w_${METHOD}_${TS}"
"openms_flair|0 1 2|${T}/brain_ms/open-ms/8_results_open-ms/01_predictions/open_ms_model/flair/auglab/open-ms_flair_${METHOD}_${TS}"
)
exec 7> "${CD}/moved_poller.lock"; flock -n 7 || { echo "[moved-poller] another instance holds the lock -- exiting"; exit 1; }
echo "$$" > "${CD}/moved_poller.pid"
log() { echo "[$(date '+%F %T')] [moved-poller] $*"; }
t_end=$(( $(date +%s) + MAX_DAYS * 86400 ))
for row in "${MOVED[@]}"; do IFS='|' read -r s lf rd <<<"${row}"; mkdir -p "${CD}/${s}"
  [ -d "${rd}" ] || { log "ERROR: ${rd} missing"; exit 2; }
  printf '%s\n%s\n' "${lf}" "${rd}" > "${CD}/${s}/LOCAL_FOLDS.new"; mv "${CD}/${s}/LOCAL_FOLDS.new" "${CD}/${s}/LOCAL_FOLDS"; done
log "start pid $$, ${#MOVED[@]} moved settings, poll ${POLL_S}s, repo HEAD $(git rev-parse --short HEAD)"
while :; do
  [ -f "${CD}/STOP_MOVED" ] && { log "STOP_MOVED file -- exiting"; break; }
  [ "$(date +%s)" -lt "${t_end}" ] || { log "max lifetime reached -- exiting"; break; }
  left=0; summary=""
  for row in "${MOVED[@]}"; do IFS='|' read -r s lf rd <<<"${row}"
    { [ -f "${CD}/${s}/DONE" ] || [ -f "${CD}/${s}/FAILED" ]; } && continue
    left=$((left + 1)); L="${CD}/${s}/moved_$(date +%Y%m%d_%H%M%S).log"
    timeout 7200 bash scripts/cluster/flatauglab/companions_vulcan.sh "${s}" > "${L}" 2>&1; rc=$?
    if [ -f "${CD}/${s}/DONE" ]; then log "${s}: DONE ($(cat "${CD}/${s}/DONE"))"; left=$((left - 1))
    elif [ "${rc}" = 3 ]; then summary="${summary} ${s}:$(grep -o 'folds done [0-9]/3' "${L}" | tail -1 | cut -d' ' -f3)"; rm -f "${L}"
    elif [ "${rc}" = 4 ] || [ "${rc}" = 124 ]; then log "${s}: rc=${rc} (connection) -- retry next cycle"
    else
      tries=$(( $(cat "${CD}/${s}/TRIES" 2>/dev/null || echo 0) + 1 )); echo "${tries}" > "${CD}/${s}/TRIES"
      log "${s}: rc=${rc}, attempt ${tries}/${MAX_TRIES} ($(tail -1 "${L}"))"
      [ "${tries}" -lt "${MAX_TRIES}" ] && [ "${rc}" != 6 ] || { echo "rc=${rc} after ${tries} attempts, see ${L}" > "${CD}/${s}/FAILED"; log "${s}: FAILED -- left for a human"; }
    fi
  done
  [ "${left}" -gt 0 ] || { log "all moved settings handled -- exiting"; break; }
  log "training:${summary:- (none)}"
  for _ in $(seq $(( POLL_S / 30 ))); do [ -f "${CD}/STOP_MOVED" ] && break; sleep 30; done
done
rm -f "${CD}/moved_poller.pid"; log "exit"
