#!/usr/bin/env bash
# GIN-IPA arm (ginipa): NaN / dead-run WATCHDOG. Run on a KILLARNEY login node (it reaches Vulcan and TamIA through that node's ssh
# masters, ~/bin/ssh-master). Every POLL_S (default 1800 s) it scans the nnU-Net training_log of every fold of the arm's run table on the
# cluster that trains it (scripts/cluster/ginipa/nan_scan.py, stdlib only) and appends one status line per fold to
# <root>/watch/watch.log on Vulcan. A fold is
#   NAN  : any train_loss / val_loss / loss_seg / loss_consist that is nan or inf (the 2026-11-02 launch died this way, see
#          AugLab NOTES_stability.md). Writes <root>/watch/NAN_<setting>_f<k> and, for a single-fold L40S unit (vulcan / killarney),
#          scancels that unit's whole chain (JOBS.tsv) so a dead run stops holding a GPU; a TamIA node shares 4 folds and is only
#          reported (decide by hand: ginipa.sh move / resubmit). No automatic resubmit: resuming the same weights repeats the failure.
#   DEAD : epoch >= DEAD_EPOCH (default 60) and pseudo-Dice mean exactly 0.000 -> <root>/watch/DEAD_<setting>_f<k>, report only.
#   ok   : otherwise (epoch, pseudo-Dice mean logged).
#   bash scripts/cluster/ginipa/nan_watch.sh once [--no-cancel]    # one pass (prints)
#   systemd-run --user --scope --unit=ginipa-watch -- nohup bash scripts/cluster/ginipa/nan_watch.sh loop >> <log> 2>&1 &
#   stop: touch <root>/watch/STOP_WATCH (on Vulcan)
set -uo pipefail
GDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-once}"; NOCANCEL="${2:-}"
V=vulcan.alliancecan.ca; TA=tamia.alliancecan.ca
VSSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=30 "${V}"); TSSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 "${TA}")
VREPO=/project/aip-jcohen/paulh/mri_synthesis_project; KREPO=/project/6102268/paulh/mri_synthesis_project
POLL_S="${POLL_S:-1800}"; DEAD_EPOCH="${DEAD_EPOCH:-60}"
R="${GINIPA_ROOT:-$("${VSSH[@]}" 'cat /scratch/paulh/_ginipa_root.txt')}" || { echo "no root"; exit 1; }
KR="/scratch/paulh/$(basename "${R}")"      # Killarney copy of the arm root (killarney.sh prep)
SCAN="${GDIR}/nan_scan.py"
log() { echo "[$(date -u '+%F %T')] [ginipa-watch] $*"; }
scan() {  # $1 cluster -> TSV lines (run_dir fold log epoch first_nan n_nan dice)
  local rid="*_auglab_ginipa_$(basename "${R}" | sed 's/^ginipa_//')"
  case "$1" in
    killarney) python3 -I "${SCAN}" "${rid}" "${KREPO}/benchmark/02_tasks" ;;
    vulcan) ssh -o BatchMode=yes -o ConnectTimeout=30 "${V}" "cd /scratch/paulh && python3 -I - '${rid}' ${VREPO}/benchmark/02_tasks" < "${SCAN}" ;;
    tamia) "${TSSH[@]}" "cd /scratch/p/paulh && python3 -I - '${rid}' /scratch/p/paulh ${VREPO}/benchmark/02_tasks" < "${SCAN}" ;;
  esac
}
pass() {
  local RT; RT="$("${VSSH[@]}" "cat ${R}/run_table.tsv")" || { log "cannot read run table"; return; }
  "${VSSH[@]}" "mkdir -p ${R}/watch"
  declare -A LINE=()
  for cl in vulcan tamia killarney; do
    while IFS=$'\t' read -r rd f lg ep fn nn dice; do [ -n "${rd}" ] && LINE["${rd}|${f}"]="${ep}"$'\t'"${fn}"$'\t'"${nn}"$'\t'"${dice}"$'\t'"${cl}"; done < <(scan "${cl}" 2>/dev/null)
  done
  local out="" nbad=0
  while IFS=$'\t' read -r s f _ _ _ _ _ cl unit _ _ _ _ _ _ rid _; do
    local v="${LINE["${rid}|${f}"]:-}" st
    if [ -z "${v}" ]; then out+="${s}:f${f}:${cl}:no-log "; continue; fi
    IFS=$'\t' read -r ep fn nn dice where <<<"${v}"
    [ "${fn}" = - ] && fn=""; [ "${dice}" = - ] && dice=""   # nan_scan.py prints '-' for an empty field
    if [ -n "${fn}" ]; then
      st=NAN; nbad=$((nbad + 1))
      if ! "${VSSH[@]}" "[ -f ${R}/watch/NAN_${s}_f${f} ]"; then
        log "NAN ${s} f${f} (${cl} unit ${unit}): first non-finite epoch ${fn}, ${nn} bad epochs, last epoch ${ep}"
        "${VSSH[@]}" "echo '$(date -u '+%F %T') ${s} f${f} ${cl} ${unit} first_nan_epoch=${fn} last_epoch=${ep}' > ${R}/watch/NAN_${s}_f${f}"
        if [ "${cl}" != tamia ] && [ -z "${NOCANCEL}" ] && [ "${MODE}" = loop ]; then
          local jf jobs
          if [ "${cl}" = killarney ]; then jf="${KR}/killarney/${unit}/JOBS.tsv"; jobs="$(cut -f3 "${jf}" 2>/dev/null | tail -1)"
            [ -n "${jobs}" ] && scancel ${jobs//,/ } && log "  scancelled Killarney chain ${jobs}"
          else jobs="$("${VSSH[@]}" "cut -f3 ${R}/vulcan/${unit}/JOBS.tsv 2>/dev/null | tail -1")"
            [ -n "${jobs}" ] && "${VSSH[@]}" "scancel ${jobs//,/ }" && log "  scancelled Vulcan chain ${jobs}"; fi
        fi
      fi
    elif [ "${ep}" -ge "${DEAD_EPOCH}" ] 2>/dev/null && [ "${dice}" = 0.000 ]; then
      st=DEAD; nbad=$((nbad + 1)); "${VSSH[@]}" "[ -f ${R}/watch/DEAD_${s}_f${f} ] || echo '$(date -u '+%F %T') epoch ${ep}' > ${R}/watch/DEAD_${s}_f${f}"
    else st=ok; fi
    out+="${s}:f${f}:${cl}:e${ep}:${st}${dice:+:d${dice}} "
  done < <(printf '%s\n' "${RT}" | awk -F'\t' 'NR>1 && $1 !~ /_parent$/')
  log "bad=${nbad} ${out}"
  "${VSSH[@]}" "echo '[$(date -u '+%F %T')] bad=${nbad} ${out}' >> ${R}/watch/watch.log"
}
if [ "${MODE}" = once ]; then NOCANCEL=1; pass; exit 0; fi
[ "${MODE}" = loop ] || { echo "usage: once|loop"; exit 1; }
log "start pid $$ root ${R}"
while :; do
  "${VSSH[@]}" "[ -f ${R}/watch/STOP_WATCH ]" && { log "STOP_WATCH -- exiting"; break; }
  pass
  sleep "${POLL_S}"
done
