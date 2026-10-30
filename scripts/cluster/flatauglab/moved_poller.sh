#!/usr/bin/env bash
# Third Vulcan login-node poller of the flatauglab arm (2026-10-30), for the settings whose folds were MOVED from TamIA to Vulcan by
# vulcan_move.sh (packs B2 + B3): onh_T2w, openms_t1w, atlas_t1w, panseg_t2w (all 3 folds on Vulcan) and panseg_t1wce (fold 2 on Vulcan,
# folds 0-1 still in TamIA pack B1). companions_poller.sh never fires for these (it waits for 3/3 folds done ON TAMIA), so every POLL_S
# (default 1800 s) this runs companions_vulcan.sh <setting> in its LOCAL_FOLDS mode (<companions dir>/<setting>/LOCAL_FOLDS, written here):
# Vulcan-trained folds are checked + md5'd in place, the TamIA folds (panseg_t1wce 0-1) are rsynced as usual; it writes SYNCED (-> the
# running vulcan_post_poller.sh submits the own-test + HCC controller) and, for panseg_*, submits the totalsegmri / msd-pancreas companions.
# Same retry rules as companions_poller.sh (rc 3 = training not complete, 4 / 124 = connection: no attempt used). One instance (flock).
# Exits when every moved setting is DONE / FAILED, on STOP_MOVED, or after MAX_DAYS (default 9).
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
"panseg_t1wce|2|${T}/pancreas_disease/pansegdata/8_results_pansegdata/01_predictions/pansegdata_model/t1wce/auglab/pansegdata_t1wce_${METHOD}_${TS}"
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
