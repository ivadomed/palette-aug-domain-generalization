#!/usr/bin/env bash
# GIN-IPA arm (ginipa): move STARTED TamIA units to Vulcan L40S when TamIA's fair-share stalls their next 24 h segment. Runs on a
# KILLARNEY login node holding ssh masters to both Vulcan and TamIA (~/bin/ssh-master): Vulcan lost its TamIA master on 2026-11-07, so
# `ginipa.sh move <unit>` (which rsyncs from TamIA itself) cannot run. Same RUN_IDs, resume from TamIA's checkpoint_latest (<= 50 epochs
# repeated). Finished folds of the unit are left alone (they stay TamIA rows; tamia_relay.sh copies them for predict + eval).
#   push [--dry-run] UNIT...  refuses while TamIA queues a ginipa_<UNIT> job; copies every unfinished fold dir (no wandb/, no validation
#                             images) + the trainer-level *.json through this node into the Vulcan run dir, compares size + md5 of
#                             checkpoint_latest / checkpoint_best, writes <Vulcan fold dir>/.ginipa_moved_from_tamia
#   auto UNIT...              loop (POLL_S, default 600 s): once a unit has no TamIA job left (scancel its pending segments first, the
#                             running one is left to end), push it, then on Vulcan: ginipa.sh move <UNIT> --from relay, record, submit.
#                             Done units: <root>/post/tamia_push_done/<UNIT>; stop: touch <root>/post/STOP_PUSH (Vulcan).
#   systemd-run --user --scope --unit=ginipa-push -- nohup bash tamia_push.sh auto T1 T2 >> push.log 2>&1 &
set -uo pipefail
MODE="${1:?usage: push [--dry-run] UNIT... | auto UNIT...}"; shift; POLL_S="${POLL_S:-600}"
V=vulcan.alliancecan.ca; TA=tamia.alliancecan.ca; VREPO=/project/aip-jcohen/paulh/mri_synthesis_project; T=benchmark/02_tasks
v() { ssh -n -o BatchMode=yes -o ConnectTimeout=30 "${V}" "$1"; }
t() { ssh -n -o BatchMode=yes -o ConnectTimeout=30 "${TA}" "$1"; }
R="${GINIPA_ROOT:-$(v 'cat /scratch/paulh/_ginipa_root.txt')}" || exit 1
log() { echo "[$(date -u '+%F %T')] [ginipa-push] $*"; }
Q='for f in checkpoint_latest.pth checkpoint_best.pth; do [ -f $f ] && echo $f $(stat -c %s $f) $(md5sum < $f | cut -c1-32); done; true'

queued() { t "squeue -u paulh -h -o '%i %j'" | awk -v n="ginipa_$1" '$2==n {print $1}'; }

push_unit() {   # push_unit UNIT DRY -> 0 ok, 1 refused/failed
  local U="$1" DRY="$2" RT q n=0
  q="$(queued "${U}")" || { log "${U}: TamIA unreachable"; return 1; }
  [ -z "${q}" ] || { log "REFUSING ${U}: TamIA still queues ${q//$'\n'/ } -- scancel them (tail first)"; return 1; }
  RT="$(v "cat ${R}/run_table.tsv")" || { log "no run table"; return 1; }
  while IFS=$'\t' read -r s f rid tw sd; do
    vrd="$(v "cd ${VREPO} && w=${T}/${sd}/04_train/${tw}; e=\$(grep -o 'source \"\$(dirname \"\$0\")/../00_utils/env[a-z0-9_]*\.sh\"' \$w | sed 's|.*/00_utils/||; s|\"\$||'); source ${T}/${sd}/00_utils/\$e >/dev/null 2>&1; echo \${PREDICTIONS_ROOT}/\${MODEL_TYPE}/\${TRAINING_CONTRAST}/auglab/${rid}")"
    [ -n "${vrd}" ] || { log "${s} f${f}: no Vulcan run dir"; return 1; }
    tfd="$(t "ls -d /scratch/p/paulh/*/8_results*/01_predictions/*/*/auglab/${rid}/Dataset*/*/fold_${f} 2>/dev/null | head -1")"
    [ -n "${tfd}" ] || { log "REFUSING ${U}: ${s} f${f}: no TamIA fold dir"; return 1; }
    if t "[ -f ${tfd}/checkpoint_final.pth ]"; then log "${s} f${f}: finished on TamIA -- stays a TamIA row (relay)"; continue; fi
    t "[ -f ${tfd}/checkpoint_latest.pth ]" || { log "REFUSING ${U}: ${s} f${f}: no checkpoint_latest in ${tfd}"; return 1; }
    ep="$(t "grep -oE 'Epoch [0-9]+' \$(ls -t ${tfd}/training_log_*.txt | head -1) | tail -1")"
    rel="${tfd#*/auglab/${rid}/}"; vfd="${vrd}/${rel}"; tpar="$(dirname "${tfd}")"; vpar="$(dirname "${vfd}")"
    log "${s} f${f}: ${TA}:${tfd} -> ${V}:${vfd} (last logged ${ep})"
    [ "${DRY}" = 1 ] && { n=$((n + 1)); continue; }
    ssh -n -o BatchMode=yes "${TA}" "cd ${tpar} && tar cf - --exclude='fold_*/wandb' --exclude='fold_*/validation' *.json fold_${f}" \
      | ssh -o BatchMode=yes "${V}" "mkdir -p ${vpar} && tar xf - -C ${vpar}" || { log "  copy failed"; return 1; }
    L="$(t "cd ${tfd} && ${Q}")"; M="$(v "cd ${vfd} && ${Q}")"
    [ -n "${L}" ] && [ "${L}" = "${M}" ] || { log "  MISMATCH TamIA '${L}' / Vulcan '${M}'"; return 1; }
    printf '%s\n%s\n' "$(date -u '+%F %T') moved from tamia ${tfd} (${ep}; relay $(hostname -s))" "${L}" | ssh -o BatchMode=yes "${V}" "cat > ${vfd}/.ginipa_moved_from_tamia"
    log "  md5 OK, marker written"; n=$((n + 1))
  done < <(printf '%s\n' "${RT}" | awk -F'\t' -v u="${U}" 'NR>1 && $8=="tamia" && $9==u {print $1"\t"$2"\t"$16"\t"$17"\t"$18}')
  log "${U}: ${n} unfinished fold(s) pushed"; return 0
}

if [ "${MODE}" = push ]; then
  DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
  for u in "$@"; do push_unit "${u}" "${DRY}" || exit 1; done; exit 0
fi

[ "${MODE}" = auto ] || { echo "usage: push [--dry-run] UNIT... | auto UNIT..."; exit 1; }
v "mkdir -p ${R}/post/tamia_push_done"
while :; do
  v "[ -e ${R}/post/STOP_PUSH ]" && { log "STOP_PUSH -- exiting"; exit 0; }
  left=0
  for U in "$@"; do
    v "[ -e ${R}/post/tamia_push_done/${U} ]" && continue
    left=$((left + 1))
    q="$(queued "${U}")" || continue
    [ -z "${q}" ] || continue                                   # its segment still runs (or a pending one was not cancelled)
    push_unit "${U}" 0 || { log "${U}: push failed (retry next pass)"; continue; }
    out="$(v "cd ${VREPO} && bash -lc 'bash scripts/cluster/ginipa/ginipa.sh move ${U} --from relay'" 2>&1)"; echo "${out}"
    newu="$(sed -n 's/^next: ALLOW_EXISTING=1 bash scripts\/cluster\/ginipa\/ginipa.sh record vulcan\(.*\) && .*/\1/p' <<<"${out}")"
    if [ -z "${newu// /}" ]; then
      # nothing unfinished (all folds ended on TamIA) or move refused: done only when no TamIA row of the unit is unfinished
      grep -q '^no pushed tamia rows' <<<"${out}" && { v "echo moved: none \$(date -u) > ${R}/post/tamia_push_done/${U}"; log "${U}: nothing to move"; }
      continue
    fi
    rec="$(v "cd ${VREPO} && bash -lc 'ALLOW_EXISTING=1 bash scripts/cluster/ginipa/ginipa.sh record vulcan${newu} && bash scripts/cluster/ginipa/ginipa.sh submit vulcan${newu}'" 2>&1)"; rc=$?
    echo "${rec}"
    [ "${rc}" = 0 ] || { log "${U}: record/submit FAILED rc=${rc} -- rows already switched to Vulcan, fix by hand (units${newu})"; v "echo FAILED rc=${rc} units${newu} \$(date -u) > ${R}/post/tamia_push_done/${U}"; continue; }
    v "echo moved units${newu} \$(date -u) > ${R}/post/tamia_push_done/${U}"; log "${U}: moved -> Vulcan units${newu}"
  done
  [ "${left}" = 0 ] && { log "all units handled -- exiting"; exit 0; }
  sleep "${POLL_S}"
done
