#!/usr/bin/env bash
# Second Vulcan login-node poller of the flatauglab arm (2026-10-29), next to companions_poller.sh (which keeps syncing checkpoints
# TamIA -> Vulcan and running the breast / pancreas companions). No TamIA access needed: every POLL_S (default 600 s) it submits ONE
# Vulcan post-training controller (vulcan_post_run.sh: own-test predict + eval + the TamIA-planned companions + audit vs headline) per
# setting whose checkpoints are synced and md5-verified (<companions dir>/<setting>/SYNCED). Idempotent: <setting>/POST_JOB (written
# right after sbatch) or a queued/running job named fapost_<setting> = never resubmitted; a FAILED controller is left for a human
# (delete POST_JOB to resubmit). One instance (flock). Exits when all settings have POST_JOB, on a STOP_POST file, or after MAX_DAYS.
#   start : systemd-run --user --scope --unit=flatauglab-post -- nohup bash scripts/cluster/flatauglab/vulcan_post_poller.sh \
#             >> /scratch/paulh/flatauglab_20261028/companions/post_poller.log 2>&1 &
#   stop  : touch /scratch/paulh/flatauglab_20261028/companions/STOP_POST
#   look  : tail /scratch/paulh/flatauglab_20261028/companions/post_poller.log; tail -2 .../companions/*/POST.status
set -uo pipefail
REPO=/project/aip-jcohen/paulh/mri_synthesis_project; cd "${REPO}"
CD="${FLATAUG_COMPANIONS_DIR:-/scratch/paulh/flatauglab_20261028/companions}"
RUNS="${FLATAUG_RUN_TABLE:-/scratch/paulh/flatauglab_20261028/run_ids.tsv}"
POLL_S="${POLL_S:-600}"; MAX_DAYS="${MAX_DAYS:-9}"
exec 8> "${CD}/post_poller.lock"; flock -n 8 || { echo "[post-poller] another instance holds the lock -- exiting"; exit 1; }
echo "$$" > "${CD}/post_poller.pid"
log() { echo "[$(date '+%F %T')] [post-poller] $*"; }
t_end=$(( $(date +%s) + MAX_DAYS * 86400 ))
mapfile -t SETTINGS < <(awk -F'\t' 'NR>1 {print $1}' "${RUNS}" | awk '!s[$0]++')
log "start pid $$, ${#SETTINGS[@]} settings, poll ${POLL_S}s, repo HEAD $(git rev-parse --short HEAD)"
while :; do
  [ -f "${CD}/STOP_POST" ] && { log "STOP_POST file -- exiting"; break; }
  [ "$(date +%s)" -lt "${t_end}" ] || { log "max lifetime reached -- exiting"; break; }
  Q="$(squeue -u "${USER}" -h -o '%j' 2>/dev/null)" || { log "squeue failed -- retry"; sleep 60; continue; }
  wait_=""; left=0
  for s in "${SETTINGS[@]}"; do
    [ -f "${CD}/${s}/POST_JOB" ] && continue
    left=$((left + 1))
    [ -f "${CD}/${s}/SYNCED" ] || { wait_="${wait_} ${s}"; continue; }
    grep -qx "fapost_${s}" <<<"${Q}" && { log "${s}: fapost_${s} already in the queue (no POST_JOB file?) -- not resubmitted"; continue; }
    mkdir -p "${CD}/${s}"
    if j="$(sbatch --parsable --account=aip-jcohen --cpus-per-task=16 --mem=96G --time=16:00:00 --job-name="fapost_${s}" \
            --output="${CD}/${s}/post_job_%j.log" --wrap "bash scripts/cluster/flatauglab/vulcan_post_run.sh ${s}")"; then
      echo "${j} $(date '+%F %T')" > "${CD}/${s}/POST_JOB"; log "${s}: controller job ${j}"
      printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T')" "${s}" post_submitted "${j}" >> "${CD}/status.tsv"; left=$((left - 1))
    else log "${s}: sbatch failed -- retry next cycle"; fi
    sleep 3
  done
  [ "${left}" -gt 0 ] || { log "all ${#SETTINGS[@]} settings have a controller -- exiting"; break; }
  log "waiting for checkpoint sync:${wait_:- (none)}"
  for _ in $(seq $(( POLL_S / 30 ))); do [ -f "${CD}/STOP_POST" ] && break; sleep 30; done
done
rm -f "${CD}/post_poller.pid"; log "exit"
