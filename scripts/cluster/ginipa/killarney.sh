#!/usr/bin/env bash
# GIN-IPA arm (ginipa): the KILLARNEY units (one fold per L40S), clone of scripts/cluster/flatauglab/killarney_move.sh. Run on a KILLARNEY
# login node; Vulcan is reached through klogin02's ssh master. The fold commands are RECORDED ON VULCAN (`ginipa.sh record killarney`,
# Vulcan paths, Vulcan pin inserted, VERIFY PASSED) and rewritten here (repo prefix + pin path only), so Killarney's older checkout never
# records anything. Order:
#   bash scripts/cluster/ginipa/killarney.sh prep [UNIT...]    # fetch <root>/killarney from Vulcan, rewrite, verify (diff vs Vulcan cmd,
#                                                              #   pin = PIN.conf + disk, config md5, preprocessed md5 vs Vulcan)
#   bash scripts/cluster/ginipa/killarney.sh submit [--dry-run] [UNIT...]   # chained guarded segments per fold (same body as ginipa.sh)
#   bash scripts/cluster/ginipa/killarney.sh status
#   bash scripts/cluster/ginipa/killarney.sh release [--dry-run] UNIT...   # hand a NEVER-STARTED unit back to Vulcan (after scancel of its whole
#     chain): refuses unless every job is CANCELLED, no seg_*.out, no fold dir; moves the Killarney unit dir to <root>/_superseded_retarget/
#     and writes <root>/killarney/<UNIT>/RELEASED_FROM_KILLARNEY on Vulcan; then on Vulcan: ginipa.sh retarget <UNIT> (see there)
#   ssh klogin02 'systemd-run --user --scope --unit=ginipa-ksync -- nohup bash <repo>/scripts/cluster/ginipa/killarney.sh syncback >> <log> 2>&1 &'
#     syncback (klogin02: needs its Vulcan master): every 30 min each finished fold dir (final + best, latest gone, final > 10 min) is
#     rsynced (--delete, that fold dir only) into the SAME Vulcan run dir, md5-verified, and marked on Vulcan with
#     <fold dir>/.ginipa_synced_from_killarney (post_poller.sh waits for that marker).
# Prerequisites on Killarney (launch time): the checkout has scripts/cluster/ginipa/ + the 9 trainers/ginipa.py (bundle + reset --mixed,
# then `git checkout -- <those paths>`, skill git-workflow), `bash scripts/cluster/ginipa/ginipa.sh shims install` passed, and the
# datasets of its units are staged (Dataset031, 070, 080, 150, 151 are; BraTS 051-054 sit in /project/6102268/paulh/toDelete_20261031/).
set -euo pipefail
KREPO=/project/6102268/paulh/mri_synthesis_project; VREPO=/project/aip-jcohen/paulh/mri_synthesis_project
cd "${KREPO}"; source scripts/cluster/ginipa/pin_env.sh   # Killarney pin (refuses on mismatch)
[ "${GINIPA_CLUSTER}" = killarney ] || { echo "run on Killarney"; exit 1; }
vrun() { if [ "$(hostname -s)" = klogin02 ]; then ssh -n -o BatchMode=yes vulcan.alliancecan.ca "$1"
         else ssh -n klogin02 "ssh -o BatchMode=yes vulcan.alliancecan.ca $(printf '%q' "$1")"; fi; }
RS_E="ssh klogin02 ssh -o BatchMode=yes"; [ "$(hostname -s)" = klogin02 ] && RS_E="ssh -o BatchMode=yes"
STATE="${GINIPA_ROOT_FILE:-/scratch/paulh/_ginipa_root${PROBE:+_probe}.txt}"   # same path on Vulcan and Killarney
VPIN="$(vrun "awk -F'\t' '\$1==\"pin_vulcan\"{print \$2}' ${VREPO}/scripts/cluster/ginipa/PIN.conf")"
MODE="${1:?usage: prep|submit|status|syncback|release|push}"; shift || true
clean_cmd() { sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$1"; }
getv() { printf '%s\n' "$1" | sed -n "s/^ *export $2=//p" | head -1 | tr -d "'"; }

if [ "${MODE}" = prep ]; then
  R="$(vrun "cat ${STATE}")"   # the arm root has the same path on both clusters (/scratch/paulh)
  echo "${R}" > "${STATE}"; mkdir -p "${R}"
  # prep [UNIT...]: once any Killarney unit is submitted, only the named units are fetched / rewritten (a running unit's command,
  # index.tsv and body are never rewritten under it)
  SELU="$*"
  if [ -z "${SELU}" ]; then
    ls "${R}"/killarney/*/JOBS.tsv >/dev/null 2>&1 && { echo "REFUSING: Killarney units already submitted -- name the units to prep"; exit 1; }
    SELU="$(vrun "ls ${R}/killarney" | tr '\n' ' ')"
  fi
  for u in ${SELU}; do [ ! -f "${R}/killarney/${u}/JOBS.tsv" ] || { echo "REFUSING: ${u} already submitted"; exit 1; }; done
  vrun "cd ${R} && tar cf - run_table.tsv TS PIN.conf LAUNCH_INFO $(for u in ${SELU}; do printf 'killarney/%s ' "${u}"; done)" | tar xf - -C "${R}"
  # the run table's pin must be this cluster's verified pin
  awk -F'\t' 'NR>1 {print $13" "$15}' "${R}/run_table.tsv" | sort -u | while read -r cm md5; do
    [ "${cm}" = "${GINIPA_COMMIT}" ] && [ "${md5}" = "${GINIPA_CONFIG_MD5}" ] || { echo "REFUSING: run table pins ${cm:0:12}/${md5}, Killarney pin ${GINIPA_COMMIT:0:12}/${GINIPA_CONFIG_MD5}"; exit 1; }
  done
  bad=0
  for u in ${SELU}; do UD="${R}/killarney/${u}"
    [ -f "${UD}/VERIFY_PASSED" ] || { echo "REFUSING: ${u} not VERIFY_PASSED on Vulcan"; exit 1; }
    mkdir -p "${UD}/vulcan_cmds"; : > "${UD}/index.k.tsv"
    while IFS=$'\t' read -r vcmd vlog name vdone; do
      v="${UD}/vulcan_cmds/$(basename "${vcmd}")"; cp "${UD}/$(basename "${vcmd}")" "${v}" 2>/dev/null || cp "${vcmd}" "${v}"
      k="${UD}/$(basename "${vcmd}")"
      sed -e "s#${VPIN}#${GINIPA_PIN}#g" -e "s#${VREPO}#${KREPO}#g" "${v}" > "${k}"
      printf '%s\t%s\t%s\t%s\n' "${k}" "${k%.sh}.log" "${name}" "${vdone//${VREPO}/${KREPO}}" >> "${UD}/index.k.tsv"
      c="$(clean_cmd "${k}")"
      rest="$(diff <(clean_cmd "${v}" | sed -e "s#${VPIN}#<PIN>#g" -e "s#${VREPO}#<REPO>#g") <(printf '%s\n' "$c" | sed -e "s#${GINIPA_PIN}#<PIN>#g" -e "s#${KREPO}#<REPO>#g") || true)"
      pre=$(getv "$c" nnUNet_preprocessed); res=$(getv "$c" nnUNet_results); pp=$(getv "$c" PYTHONPATH); cfgp=$(getv "$c" AUGLAB_PARAMS_GPU_JSON)
      ok=OK; [ -z "${rest}" ] || ok="${ok},CMD_DIFF"; [ -d "${pre}" ] || ok="${ok},NO_PREPROC(${pre})"; [[ "${pp}" == "${GINIPA_PIN}:"* ]] || ok="${ok},PIN_NOT_FIRST"
      [ "${cfgp}" = "${GINIPA_PIN}/auglab/configs/${GINIPA_CONFIG}" ] && [ "$(md5sum < "${cfgp}" | cut -c1-32)" = "${GINIPA_CONFIG_MD5}" ] || ok="${ok},CFG"
      if [ -n "${PROBE:-}" ]; then [[ "${res}" == */_ginipa_probe/*"/auglab/${name#fold?_}" ]] || ok="${ok},PROBE_NOT_THROWAWAY(${res})"
      else [[ "${res}" == "${KREPO}/benchmark/02_tasks/"*"/auglab/${name#fold?_}" ]] || ok="${ok},RESULTS(${res})"; fi
      [ ! -e "${res}" ] || [ -n "${ALLOW_EXISTING:-}" ] || ok="${ok},RESULTS_DIR_EXISTS"
      grep -q '/project/aip-jcohen' "${k}" && ok="${ok},VULCAN_PATH_LEFT"
      echo "   ${ok} ${u} ${name}"; [ -z "${rest}" ] || echo "${rest}" | sed 's/^/      /'; [ "${ok}" = OK ] || bad=1
    done < "${UD}/index.tsv"
    mv "${UD}/index.tsv" "${UD}/index.vulcan.tsv"; mv "${UD}/index.k.tsv" "${UD}/index.tsv"
  done
  # preprocessed parity with Vulcan, per dataset id of the commands
  SELF="$(for u in ${SELU}; do ls "${R}/killarney/${u}"/fold*.sh; done)"
  for did in $(cat ${SELF} 2>/dev/null | grep -oE 'nnUNetv2_train [0-9]+' | awk '{print $2}' | sort -u); do
    f="$(grep -l "nnUNetv2_train ${did} " ${SELF} | head -1)"; kp=$(getv "$(clean_cmd "$f")" nnUNet_preprocessed); vp="${kp//${KREPO}/${VREPO}}"
    dsd="$(cd "$kp" && ls -d Dataset${did}_* | head -1)"
    q='for x in nnUNetPlans.json dataset.json splits_final.json dataset_fingerprint.json; do echo "$(md5sum < $x | cut -c1-32) $x"; done; echo "nfiles $(ls nnUNetPlans_3d_fullres | wc -l)"'
    L="$(cd "$kp/$dsd" && bash -c "$q")"; Rm="$(vrun "cd $vp/$dsd && bash -c '$(printf '%s' "$q" | sed "s/'/'\\\\''/g")'")"
    if [ "$L" = "$Rm" ]; then echo "   preprocessed ${dsd}: identical to Vulcan ($(echo "$L" | tail -1))"; else echo "   PREPROCESSED DIFF ${dsd}"; diff <(echo "$Rm") <(echo "$L") || true; bad=1; fi
  done
  [ "${bad}" = 0 ] && { for u in ${SELU}; do date '+%F %T' > "${R}/killarney/${u}/KPREP_PASSED"; done; echo "PREP PASSED (${SELU})"; } || { echo "PREP FAILED"; exit 1; }
  exit 0
fi

[ -f "${STATE}" ] || { echo "no ${STATE}: run prep"; exit 1; }
R="$(cat "${STATE}")"; RT="${R}/run_table.tsv"

if [ "${MODE}" = submit ]; then
  DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
  SEL="${*:-$(ls "${R}/killarney")}"
  for u in ${SEL}; do UD="${R}/killarney/${u}"
    [ -f "${UD}/KPREP_PASSED" ] || { echo "REFUSING: ${u} not prepped"; exit 1; }
    [ "${DRY}" = 1 ] || [ ! -f "${UD}/JOBS.tsv" ] || { echo "[submit] ${u} already submitted"; continue; }
    echo 0 > "${UD}/GPU_MAP"
    # the same unit body as ginipa.sh (preflight: pin, imports from the pin, config path + md5, pin first in the cmd), Killarney paths
    body="$(GINIPA_ROOT_FILE="${STATE}" bash scripts/cluster/ginipa/ginipa.sh body "${UD}" killarney)"
    nseg="$(awk -F'\t' -v u="${u}" 'NR>1 && $9==u {print $11; exit}' "${RT}")"; st="$(awk -F'\t' -v u="${u}" 'NR>1 && $9==u {print $12; exit}' "${RT}")"
    [ -z "${PROBE:-}" ] || { nseg=1; st=00:50:00; }
    common=(--account=aip-jcohen --gres=gpu:l40s:1 --cpus-per-task=16 --mem=110G --time="${st}" --job-name="ginipa${PROBE:+probe}_${u}" --output="${UD}/seg_%j.out")
    prev=""; jobs_=""
    for ((k=1; k<=nseg; k++)); do dep=(); [ -z "${prev}" ] || dep=(--dependency=afterany:"${prev}")
      if [ "${DRY}" = 1 ]; then echo "[dry] sbatch ${common[*]} ${dep[*]} ${body}"; prev="DRY${k}"; else prev="$(sbatch --parsable "${common[@]}" "${dep[@]}" "${body}")"; sleep 2; fi
      jobs_="${jobs_:+${jobs_},}${prev}"; done
    [ "${DRY}" = 1 ] || printf '%s\tkillarney\t%s\t%s\n' "${u}" "${jobs_}" "$(date '+%F %T')" | tee -a "${UD}/JOBS.tsv" >> "${R}/JOBS_ALL_killarney.tsv"
    echo "[submit] killarney/${u}: ${nseg} x ${st}: ${jobs_}"
  done
  exit 0
fi

if [ "${MODE}" = release ]; then
  DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
  [ $# -gt 0 ] || { echo "usage: release [--dry-run] UNIT..."; exit 1; }
  SUP="${R}/_superseded_retarget"; declare -A ST
  for u in "$@"; do UD="${R}/killarney/${u}"
    [ -f "${UD}/JOBS.tsv" ] && [ -f "${UD}/index.tsv" ] || { echo "REFUSING: no ${UD}/JOBS.tsv + index.tsv"; exit 1; }
    [ ! -e "${SUP}/${u}" ] || { echo "REFUSING: ${SUP}/${u} exists"; exit 1; }
    sts=""
    for j in $(cut -f3 "${UD}/JOBS.tsv" | tr ',' ' '); do
      st="$(sacct -j "${j}" -X -n -o State | head -1 | tr -d ' ')"; [[ "${st}" == CANCELLED* ]] || { echo "REFUSING: ${u} job ${j} is ${st:-unknown} (scancel the whole chain first)"; exit 1; }
      sts="${sts} ${j}:${st}"; done
    if ls "${UD}"/seg_*.out >/dev/null 2>&1; then echo "REFUSING: ${u} has segment output (a segment ran): $(ls "${UD}"/seg_*.out | tr '\n' ' ')"; exit 1; fi
    while IFS=$'\t' read -r cmd log name done_; do
      res="$(getv "$(clean_cmd "${cmd}")" nnUNet_results)"; [ -n "${res}" ] || { echo "REFUSING: ${u} ${name}: no nnUNet_results in ${cmd}"; exit 1; }
      if ls -d "${res}"/Dataset*/*/fold_"${name:4:1}" >/dev/null 2>&1; then echo "REFUSING: ${u} ${name} has a fold dir under ${res} (it started)"; exit 1; fi
    done < "${UD}/index.tsv"
    ST[${u}]="${sts# }"; echo "[release] ${u}: chain cancelled (${sts# }), no segment output, no fold dir -- never started"
  done
  [ "${DRY}" = 0 ] || { echo "[dry] would move each unit dir to ${SUP}/ and write ${R}/killarney/<unit>/RELEASED_FROM_KILLARNEY on Vulcan"; exit 0; }
  for u in "$@"; do
    msg="$(date '+%F %T') released never-started Killarney unit ${u} (jobs ${ST[${u}]}) from $(hostname -s)"
    vrun "test -d ${R}/killarney/${u} && printf '%s\n' '${msg}' > ${R}/killarney/${u}/RELEASED_FROM_KILLARNEY" || { echo "could not write the Vulcan marker for ${u} -- nothing moved for it"; exit 1; }
    mkdir -p "${SUP}"; mv "${R}/killarney/${u}" "${SUP}/${u}"; echo "${msg}" | tee -a "${R}/LAUNCH_INFO"
  done
  echo "next (Vulcan): bash scripts/cluster/ginipa/ginipa.sh retarget <UNIT> --dry-run, then without --dry-run, then ginipa.sh submit vulcan V_..."
  exit 0
fi

# push [--dry-run] UNIT...: move STARTED Killarney units to Vulcan (Killarney queue stall). Refuses while any job of the unit's chain is still queued /
# running here (scancel the whole chain first, tail first). Per fold: plans/dataset json + the fold dir (checkpoint_latest/best, logs, debug json;
# no validation niftis, no wandb) rsynced into the SAME Vulcan run dir, size + md5 of checkpoint_latest/best compared, then the Vulcan marker
# <fold dir>/.ginipa_moved_from_killarney and the local <unit>/MOVED_TO_VULCAN. On Vulcan next: ginipa.sh move <unit> --from killarney
# (run table -> vulcan unit MV_<setting>_f<k>), ALLOW_EXISTING=1 ginipa.sh record vulcan MV_..., ginipa.sh submit vulcan MV_... (resume from
# checkpoint_latest at runtime, same RUN_ID). syncback skips moved units.
if [ "${MODE}" = push ]; then
  DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }; [ $# -gt 0 ] || { echo "usage: push [--dry-run] UNIT..."; exit 1; }
  Q='for f in checkpoint_latest.pth checkpoint_best.pth; do [ -f $f ] && echo $f $(stat -c %s $f) $(md5sum < $f | cut -c1-32); done; true'
  live="$(squeue -u "${USER}" -h -o %i)"
  for u in "$@"; do UD="${R}/killarney/${u}"; [ -f "${UD}/JOBS.tsv" ] || { echo "no ${UD}/JOBS.tsv"; exit 1; }
    for j in $(cut -f3 "${UD}/JOBS.tsv" | tr ',' ' '); do grep -qx "${j}" <<<"${live}" && { echo "REFUSING: ${u} job ${j} still queued/running -- scancel the chain first"; exit 1; }; done
  done
  for u in "$@"; do UD="${R}/killarney/${u}"
    while IFS=$'\t' read -r cmd log name done_; do
      fd="$(dirname "${done_}")"; vfd="${fd//${KREPO}/${VREPO}}"
      [ -f "${fd}/checkpoint_latest.pth" ] || { echo "REFUSING: no checkpoint_latest in ${fd}"; exit 1; }
      ep="$(grep -oE 'Epoch [0-9]+' "$(ls -t "${fd}"/training_log_*.txt | head -1)" | tail -1)"
      echo "[push] ${u} ${name}: ${fd} -> Vulcan ${vfd} (last logged ${ep})"
      [ "${DRY}" = 1 ] && continue
      vrun "mkdir -p ${vfd}"
      rsync -a -e "${RS_E}" --include='*.json' --exclude='*' "$(dirname "${fd}")/" "vulcan.alliancecan.ca:$(dirname "${vfd}")/" < /dev/null
      rsync -a -e "${RS_E}" --exclude=wandb/ --exclude='validation/' "${fd}/" "vulcan.alliancecan.ca:${vfd}/" < /dev/null
      L="$(cd "${fd}" && bash -c "$Q")"; Rm="$(vrun "cd ${vfd} && ${Q}")"
      [ "$L" = "$Rm" ] && [ -n "$L" ] || { echo "MISMATCH ${name}: Killarney '$L' / Vulcan '$Rm'"; exit 5; }
      vrun "printf '%s\n' '$(date '+%F %T') moved from Killarney ${fd} (${ep})' '$(echo "$L" | tr '\n' ';')' > ${vfd}/.ginipa_moved_from_killarney"
      echo "   md5 OK: $(echo "$L" | tr '\n' ' ')"
    done < "${UD}/index.tsv"
    [ "${DRY}" = 1 ] || echo "$(date '+%F %T') pushed to Vulcan (chain cancelled)" > "${UD}/MOVED_TO_VULCAN"
  done
  [ "${DRY}" = 1 ] || echo "next (Vulcan): for u in $*; do bash scripts/cluster/ginipa/ginipa.sh move \$u --from killarney; done; then ALLOW_EXISTING=1 ginipa.sh record vulcan MV_...; ginipa.sh submit vulcan MV_..."
  exit 0
fi

if [ "${MODE}" = status ]; then GINIPA_ROOT_FILE="${STATE}" bash scripts/cluster/ginipa/ginipa.sh status killarney; exit 0; fi

if [ "${MODE}" = syncback ]; then
  exec 6> "/scratch/paulh/ginipa_syncback.lock"; flock -n 6 || { echo "another syncback holds the lock"; exit 1; }
  log() { echo "[$(date '+%F %T')] [ginipa-ksync] $*"; }
  t_end=$(( $(date +%s) + ${MAX_DAYS:-12} * 86400 )); log "start pid $$ on $(hostname -s), root ${R}"
  while :; do
    [ -f "${R}/STOP_SYNCBACK" ] && { log "STOP file -- exiting"; break; }
    [ "$(date +%s)" -lt "${t_end}" ] || { log "max lifetime -- exiting"; break; }
    left=0; now=$(date +%s)
    for idx in "${R}"/killarney/*/index.tsv; do
      [ -f "$(dirname "${idx}")/MOVED_TO_VULCAN" ] && continue   # pushed to Vulcan: trains there now
      while IFS=$'\t' read -r cmd log_ name done_; do
        fd="$(dirname "${done_}")"; [ -f "${cmd%.sh}.SYNCED_TO_VULCAN" ] && continue; left=$((left + 1))
        [ -f "${fd}/checkpoint_final.pth" ] && [ -f "${fd}/checkpoint_best.pth" ] && [ ! -e "${fd}/checkpoint_latest.pth" ] \
          && [ $(( now - $(stat -c %Y "${fd}/checkpoint_final.pth") )) -gt 600 ] || continue
        vfd="${fd//${KREPO}/${VREPO}}"; log "${name}: finished -> rsync --delete to Vulcan ${vfd}"
        vrun "mkdir -p ${vfd}" || continue
        rsync -a -e "${RS_E}" --include='*.json' --exclude='*' "$(dirname "${fd}")/" "vulcan.alliancecan.ca:$(dirname "${vfd}")/" < /dev/null || { log "rsync json failed"; continue; }
        rsync -a --delete -e "${RS_E}" "${fd}/" "vulcan.alliancecan.ca:${vfd}/" < /dev/null || { log "${name}: rsync failed (retry)"; continue; }
        q='for f in checkpoint_final.pth checkpoint_best.pth; do echo $f $(stat -c %s $f) $(md5sum < $f | cut -c1-32); done; ls checkpoint_latest.pth 2>/dev/null; true'
        L="$(cd "${fd}" && bash -c "$q")"; Rm="$(vrun "cd ${vfd} && ${q}")" || { log "${name}: md5 call failed (retry)"; continue; }
        if [ "$L" = "$Rm" ]; then
          vrun "printf '%s\n' '$(date '+%F %T') killarney ${fd}' > ${vfd}/.ginipa_synced_from_killarney" || continue
          printf '%s\n%s\n' "$(date '+%F %T') ${vfd}" "$L" > "${cmd%.sh}.SYNCED_TO_VULCAN"; log "${name}: SYNCED"; left=$((left - 1))
        else log "${name}: md5 MISMATCH (retry next cycle)"; fi
      done < "${idx}"
    done
    [ "${left}" -gt 0 ] || { log "all Killarney folds synced -- exiting"; break; }
    for _ in $(seq 60); do [ -f "${R}/STOP_SYNCBACK" ] && break; sleep 30; done
  done
  exit 0
fi
echo "unknown mode"; exit 1
