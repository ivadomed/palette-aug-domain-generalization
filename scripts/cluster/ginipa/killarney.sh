#!/usr/bin/env bash
# GIN-IPA arm (ginipa): the KILLARNEY units (one fold per L40S), clone of scripts/cluster/flatauglab/killarney_move.sh. Run on a KILLARNEY
# login node; Vulcan is reached through klogin02's ssh master. The fold commands are RECORDED ON VULCAN (`ginipa.sh record killarney`,
# Vulcan paths, Vulcan pin inserted, VERIFY PASSED) and rewritten here (repo prefix + pin path only), so Killarney's older checkout never
# records anything. Order:
#   bash scripts/cluster/ginipa/killarney.sh prep              # fetch <root>/killarney from Vulcan, rewrite, verify (diff vs Vulcan cmd,
#                                                              #   pin = PIN.conf + disk, config md5, preprocessed md5 vs Vulcan)
#   bash scripts/cluster/ginipa/killarney.sh submit [--dry-run] [UNIT...]   # chained guarded segments per fold (same body as ginipa.sh)
#   bash scripts/cluster/ginipa/killarney.sh status
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
MODE="${1:?usage: prep|submit|status|syncback}"; shift || true
clean_cmd() { sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$1"; }
getv() { printf '%s\n' "$1" | sed -n "s/^ *export $2=//p" | head -1 | tr -d "'"; }

if [ "${MODE}" = prep ]; then
  R="$(vrun "cat ${STATE}")"   # the arm root has the same path on both clusters (/scratch/paulh)
  echo "${R}" > "${STATE}"; mkdir -p "${R}"
  vrun "cd ${R} && tar cf - run_table.tsv TS PIN.conf LAUNCH_INFO killarney" | tar xf - -C "${R}"
  # the run table's pin must be this cluster's verified pin
  awk -F'\t' 'NR>1 {print $13" "$15}' "${R}/run_table.tsv" | sort -u | while read -r cm md5; do
    [ "${cm}" = "${GINIPA_COMMIT}" ] && [ "${md5}" = "${GINIPA_CONFIG_MD5}" ] || { echo "REFUSING: run table pins ${cm:0:12}/${md5}, Killarney pin ${GINIPA_COMMIT:0:12}/${GINIPA_CONFIG_MD5}"; exit 1; }
  done
  bad=0
  for UD in "${R}"/killarney/*/; do UD="${UD%/}"; u="$(basename "${UD}")"
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
      [[ "${res}" == "${KREPO}/benchmark/02_tasks/"*"/auglab/${name#fold?_}" ]] || ok="${ok},RESULTS(${res})"
      [ ! -e "${res}" ] || [ -n "${ALLOW_EXISTING:-}" ] || ok="${ok},RESULTS_DIR_EXISTS"
      grep -q '/project/aip-jcohen' "${k}" && ok="${ok},VULCAN_PATH_LEFT"
      echo "   ${ok} ${u} ${name}"; [ -z "${rest}" ] || echo "${rest}" | sed 's/^/      /'; [ "${ok}" = OK ] || bad=1
    done < "${UD}/index.tsv"
    mv "${UD}/index.tsv" "${UD}/index.vulcan.tsv"; mv "${UD}/index.k.tsv" "${UD}/index.tsv"
  done
  # preprocessed parity with Vulcan, per dataset id of the commands
  for did in $(cat "${R}"/killarney/*/fold*.sh 2>/dev/null | grep -oE 'nnUNetv2_train [0-9]+' | awk '{print $2}' | sort -u); do
    f="$(grep -l "nnUNetv2_train ${did} " "${R}"/killarney/*/fold*.sh | head -1)"; kp=$(getv "$(clean_cmd "$f")" nnUNet_preprocessed); vp="${kp//${KREPO}/${VREPO}}"
    dsd="$(cd "$kp" && ls -d Dataset${did}_* | head -1)"
    q='for x in nnUNetPlans.json dataset.json splits_final.json dataset_fingerprint.json; do echo "$(md5sum < $x | cut -c1-32) $x"; done; echo "nfiles $(ls nnUNetPlans_3d_fullres | wc -l)"'
    L="$(cd "$kp/$dsd" && bash -c "$q")"; Rm="$(vrun "cd $vp/$dsd && bash -c '$(printf '%s' "$q" | sed "s/'/'\\\\''/g")'")"
    if [ "$L" = "$Rm" ]; then echo "   preprocessed ${dsd}: identical to Vulcan ($(echo "$L" | tail -1))"; else echo "   PREPROCESSED DIFF ${dsd}"; diff <(echo "$Rm") <(echo "$L") || true; bad=1; fi
  done
  [ "${bad}" = 0 ] && { for UD in "${R}"/killarney/*/; do date '+%F %T' > "${UD%/}/KPREP_PASSED"; done; echo "PREP PASSED"; } || { echo "PREP FAILED"; exit 1; }
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
