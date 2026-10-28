#!/usr/bin/env bash
# Background poller on the Vulcan LOGIN node for the flatauglab arm (2026-10-18): every POLL_S (default 1800 s) it asks TamIA (ONE batched ssh,
# tamia_ckpt_status.sh) which of the 19 settings have all 3 folds finished, and runs companions_vulcan.sh once for each finished setting
# not yet handled: checkpoint sync to Vulcan for every setting (durability; TamIA scratch is purge-on-inactivity) + the Vulcan-only
# companion predict / eval jobs (ispy2_* -> duke / ispy1 / acrin6698, panseg_* -> totalsegmri-pancreas / msd-pancreas).
# Light: ssh + rsync + md5 of the 6 checkpoints of a finished setting; the compute runs as Slurm jobs. Never touches other runs.
# METRICS back to Vulcan (metrics_sync.sh, added 2026-10-18): units own:<setting> (19) + comp:mandible / comp:chaos_fov / comp:hcc
# (the TamIA companions); a unit is synced once all its TamIA jobs are COMPLETED 0:0, marker <dir>/_metrics/<unit>/METRICS_SYNCED;
# a failed / audit-failed / never-runnable job is logged "FAILED ... job <id>" and marked METRICS_FAILED (left for a human).
# Connection problems (rc 4, or a 124 timeout) never use an attempt: back off and retry next cycle.
# Idempotent: <companions dir>/<setting>/DONE (written by companions_vulcan.sh) = handled, never re-run; a setting that fails
# MAX_TRIES times gets FAILED and is left for a human. One instance at a time (flock). Exits when all 19 are DONE / FAILED, on a STOP
# file, or after MAX_DAYS (default 9).
#   start : systemd-run --user --scope --unit=flatauglab-companions -- nohup bash scripts/cluster/flatauglab/companions_poller.sh \
#             >> /scratch/paulh/flatauglab_20261028/companions/poller.log 2>&1 &
#   stop  : touch /scratch/paulh/flatauglab_20261028/companions/STOP   (or: systemctl --user stop flatauglab-companions.scope)
#   look  : tail /scratch/paulh/flatauglab_20261028/companions/{poller.log,status.tsv}
set -uo pipefail
REPO=/project/aip-jcohen/paulh/mri_synthesis_project; cd "${REPO}"
CD="${FLATAUG_COMPANIONS_DIR:-/scratch/paulh/flatauglab_20261028/companions}"
RUNS="${FLATAUG_RUN_TABLE:-/scratch/paulh/flatauglab_20261028/run_ids.tsv}"
POLL_S="${POLL_S:-1800}"; MAX_TRIES="${MAX_TRIES:-4}"; MAX_DAYS="${MAX_DAYS:-9}"
TAMIA=tamia.alliancecan.ca
mkdir -p "${CD}"
exec 9> "${CD}/poller.lock"; flock -n 9 || { echo "[poller] another instance holds ${CD}/poller.lock -- exiting"; exit 1; }
echo "$$" > "${CD}/poller.pid"
log() { echo "[$(date '+%F %T')] [poller] $*"; }
t_end=$(( $(date +%s) + MAX_DAYS * 86400 ))
mapfile -t SETTINGS < <(awk -F'\t' 'NR>1 {print $1}' "${RUNS}" | awk '!s[$0]++')
UNITS=(); for s in "${SETTINGS[@]}"; do UNITS+=("own:${s}"); done; UNITS+=(comp:mandible comp:chaos_fov comp:hcc)
log "start pid $$, ${#SETTINGS[@]} settings, poll ${POLL_S}s, max ${MAX_DAYS} d, repo HEAD $(git rev-parse --short HEAD)"
while :; do
  [ -f "${CD}/STOP" ] && { log "STOP file -- exiting"; break; }
  [ "$(date +%s)" -lt "${t_end}" ] || { log "max lifetime reached -- exiting"; break; }
  pend=(); for s in "${SETTINGS[@]}"; do [ -f "${CD}/${s}/DONE" ] || [ -f "${CD}/${s}/FAILED" ] || pend+=("$s"); done
  mpend=(); for u in "${UNITS[@]}"; do m="${CD}/_metrics/${u/:/_}"; [ -f "${m}/METRICS_SYNCED" ] || [ -f "${m}/METRICS_FAILED" ] || mpend+=("$u"); done
  [ ${#pend[@]} -gt 0 ] || [ ${#mpend[@]} -gt 0 ] || { log "all ${#SETTINGS[@]} settings and ${#UNITS[@]} metrics units handled -- exiting"; break; }
  if ! ssh -O check "${TAMIA}" >/dev/null 2>&1; then
    log "RELAY DOWN: no ControlMaster for ${TAMIA} -- Paul: mkdir -p -m 700 /dev/shm/paulh-ssh && ssh -fN ${TAMIA}  (retry in ${POLL_S}s)"
  elif ! STAT="$(timeout 300 ssh -o BatchMode=yes -o ConnectTimeout=30 "${TAMIA}" 'bash -s' < scripts/cluster/flatauglab/tamia_ckpt_status.sh)"; then
    log "TamIA status call failed (retry in ${POLL_S}s)"
  else
    summary=""
    for s in "${pend[@]}"; do
      rid="$(awk -F'\t' -v s="$s" '$1==s {print $3}' "${RUNS}" | sort -u)"
      n="$(echo "${STAT}" | awk -F'\t' -v r="${rid}" '$1=="ST" && $2==r && $4=="done" {print $3}' | sort -u | grep -c . || true)"
      summary="${summary} ${s}:${n}/3"
      [ "${n}" = 3 ] || continue
      mkdir -p "${CD}/${s}"; L="${CD}/${s}/run_$(date +%Y%m%d_%H%M%S).log"
      log "${s}: training complete -> companions_vulcan.sh (log ${L})"
      timeout 7200 bash scripts/cluster/flatauglab/companions_vulcan.sh "${s}" > "${L}" 2>&1; rc=$?
      if [ -f "${CD}/${s}/DONE" ]; then log "${s}: DONE ($(cat "${CD}/${s}/DONE"))"
      elif [ "${rc}" = 4 ] || [ "${rc}" = 124 ] || [ "${rc}" = 3 ]; then log "${s}: rc=${rc} (connection / not complete) -- retry next cycle, no attempt used"
      else
        tries=$(( $(cat "${CD}/${s}/TRIES" 2>/dev/null || echo 0) + 1 )); echo "${tries}" > "${CD}/${s}/TRIES"
        log "${s}: rc=${rc}, attempt ${tries}/${MAX_TRIES} ($(tail -1 "${L}"))"
        [ "${tries}" -lt "${MAX_TRIES}" ] && [ "${rc}" != 6 ] || { echo "rc=${rc} after ${tries} attempts, see ${L}" > "${CD}/${s}/FAILED"; log "${s}: FAILED -- left for a human"; }
      fi
    done
    log "pending:${summary}"
    # ---- metrics back to Vulcan
    msum=""
    for u in "${mpend[@]}"; do
      case "$u" in own:*) need="${u#own:}";; comp:mandible) need=tf2_cbct;; comp:chaos_fov) need="chaos_t1in chaos_t2spir";; comp:hcc) need=atlas_t1w;; esac
      ok=1; for s in ${need}; do rid="$(awk -F'\t' -v s="$s" '$1==s {print $3}' "${RUNS}" | sort -u)"
        n="$(echo "${STAT}" | awk -F'\t' -v r="${rid}" '$1=="ST" && $2==r && $4=="done" {print $3}' | sort -u | grep -c . || true)"; [ "${n}" = 3 ] || ok=0; done
      [ "${ok}" = 1 ] || continue                       # training not finished: its post jobs cannot be done
      m="${CD}/_metrics/${u/:/_}"; mkdir -p "${m}"; L="${m}/run_$(date +%Y%m%d_%H%M%S).log"
      timeout 3600 bash scripts/cluster/flatauglab/metrics_sync.sh "${u}" > "${L}" 2>&1; rc=$?
      case "${rc}" in
        0) log "${u}: METRICS_SYNCED ($(tail -1 "${L}"))" ;;
        3) msum="${msum} ${u}:waiting"; rm -f "${L}" ;;
        4|124) log "${u}: connection problem rc=${rc} -- retry next cycle, no attempt used" ;;
        10) grep -h 'FAILED' "${L}" | while read -r line; do log "${u}: ${line#*] }"; done
            grep -h 'FAILED' "${L}" > "${m}/METRICS_FAILED"; log "${u}: METRICS_FAILED -- left for a human (${L})" ;;
        *) tries=$(( $(cat "${m}/TRIES" 2>/dev/null || echo 0) + 1 )); echo "${tries}" > "${m}/TRIES"
           log "${u}: rc=${rc}, attempt ${tries}/${MAX_TRIES} ($(tail -1 "${L}"))"
           [ "${tries}" -lt "${MAX_TRIES}" ] || { echo "rc=${rc} after ${tries} attempts, see ${L}" > "${m}/METRICS_FAILED"; log "${u}: METRICS_FAILED"; } ;;
      esac
    done
    [ -z "${msum}" ] || log "metrics waiting:${msum}"
  fi
  for _ in $(seq $(( POLL_S / 30 ))); do [ -f "${CD}/STOP" ] && break; sleep 30; done
done
rm -f "${CD}/poller.pid"; log "exit"
