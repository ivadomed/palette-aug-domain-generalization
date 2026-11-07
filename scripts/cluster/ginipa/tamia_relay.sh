#!/usr/bin/env bash
# GIN-IPA arm (ginipa): TamIA -> Vulcan fold relay, run on a KILLARNEY login node that holds ssh masters to BOTH Vulcan and TamIA
# (~/bin/ssh-master). Needed since 2026-11-07: Vulcan's login node lost its TamIA relay master, so post_poller.sh (Vulcan) cannot rsync
# TamIA folds itself. Every POLL_S (1800 s) for each TamIA row of the run table: when the TamIA fold is done (final + best, latest gone,
# final > 10 min old), copy the trainer-level *.json and the fold dir (no wandb/, no validation images) through this node into the SAME
# Vulcan run dir that post_poller.sh derives (rundir(): the train wrapper's env on Vulcan), compare size + md5 of checkpoint_final /
# checkpoint_best on both sides and write <fold dir>/.ginipa_synced_from_tamia on Vulcan -- the marker post_poller.sh waits for.
#   bash scripts/cluster/ginipa/tamia_relay.sh once|loop        stop: touch <root>/post/STOP_RELAY (Vulcan)
#   systemd-run --user --scope --unit=ginipa-relay -- nohup bash scripts/cluster/ginipa/tamia_relay.sh loop >> <log> 2>&1 &
set -uo pipefail
MODE="${1:-once}"; POLL_S="${POLL_S:-1800}"
V=vulcan.alliancecan.ca; TA=tamia.alliancecan.ca; VREPO=/project/aip-jcohen/paulh/mri_synthesis_project; T=benchmark/02_tasks
v() { ssh -n -o BatchMode=yes -o ConnectTimeout=30 "${V}" "$1"; }
t() { ssh -n -o BatchMode=yes -o ConnectTimeout=30 "${TA}" "$1"; }
R="${GINIPA_ROOT:-$(v 'cat /scratch/paulh/_ginipa_root.txt')}" || exit 1
log() { echo "[$(date -u '+%F %T')] [ginipa-relay] $*"; }
MD5Q='for f in checkpoint_final.pth checkpoint_best.pth; do echo $f $(stat -c %s $f) $(md5sum < $f | cut -c1-32); done'
pass() {
  local RT left=0; RT="$(v "cat ${R}/run_table.tsv")" || { log "no run table"; return 1; }
  while IFS=$'\t' read -r s f rid tw sd; do
    vrd="$(v "cd ${VREPO} && w=${T}/${sd}/04_train/${tw}; e=\$(grep -o 'source \"\$(dirname \"\$0\")/../00_utils/env[a-z0-9_]*\.sh\"' \$w | sed 's|.*/00_utils/||; s|\"\$||'); source ${T}/${sd}/00_utils/\$e >/dev/null 2>&1; echo \${PREDICTIONS_ROOT}/\${MODEL_TYPE}/\${TRAINING_CONTRAST}/auglab/${rid}")"
    [ -n "${vrd}" ] || { log "${s} f${f}: no Vulcan run dir"; continue; }
    if v "ls ${vrd}/Dataset*/*/fold_${f}/.ginipa_synced_from_tamia" >/dev/null 2>&1; then continue; fi
    left=$((left + 1))
    tfd="$(t "ls -d /scratch/p/paulh/*/8_results*/01_predictions/*/*/auglab/${rid}/Dataset*/*/fold_${f} ${VREPO}/${T}/*/*/8_results*/01_predictions/*/*/auglab/${rid}/Dataset*/*/fold_${f} 2>/dev/null | head -1")"
    [ -n "${tfd}" ] || continue
    st="$(t "d=${tfd}; [ -f \$d/checkpoint_final.pth ] && [ -f \$d/checkpoint_best.pth ] && [ ! -e \$d/checkpoint_latest.pth ] && [ \$(( \$(date +%s) - \$(stat -c %Y \$d/checkpoint_final.pth) )) -gt 600 ] && echo done || echo training")"
    [ "${st}" = done ] || continue
    rel="${tfd#*/auglab/${rid}/}"; vfd="${vrd}/${rel}"; tpar="$(dirname "${tfd}")"; vpar="$(dirname "${vfd}")"
    log "${s} f${f}: ${TA}:${tfd} -> ${V}:${vfd}"
    ssh -o BatchMode=yes "${TA}" "cd ${tpar} && tar cf - --exclude='fold_*/wandb' --exclude='fold_*/validation/*.nii.gz' --exclude='fold_*/validation/*.npz' *.json fold_${f}" \
      | ssh -o BatchMode=yes "${V}" "mkdir -p ${vpar} && tar xf - -C ${vpar}" || { log "  copy failed (retry next pass)"; continue; }
    L="$(t "cd ${tfd} && ${MD5Q}")"; M="$(v "cd ${vfd} && ${MD5Q}")"
    if [ -n "${L}" ] && [ "${L}" = "${M}" ]; then
      printf '%s\n%s\n' "$(date -u '+%F %T') tamia ${tfd} (relay $(hostname -s))" "${L}" | ssh -o BatchMode=yes "${V}" "cat > ${vfd}/.ginipa_synced_from_tamia"
      log "  md5 OK, marker written"; left=$((left - 1))
    else log "  md5 MISMATCH (retry next pass)"; fi
  done < <(printf '%s\n' "${RT}" | awk -F'\t' 'NR>1 && $8=="tamia" {print $1"\t"$2"\t"$16"\t"$17"\t"$18}')
  log "pass done: ${left} TamIA folds not yet on Vulcan"; [ "${left}" -gt 0 ]
}
if [ "${MODE}" = once ]; then pass; exit 0; fi
log "start pid $$ root ${R}"
while :; do
  v "[ -f ${R}/post/STOP_RELAY ]" && { log "STOP_RELAY -- exiting"; break; }
  pass || { log "every TamIA fold relayed -- exiting"; break; }
  sleep "${POLL_S}"
done
