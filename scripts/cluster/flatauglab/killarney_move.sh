#!/usr/bin/env bash
# flatauglab pack B3 (atlas-liver-hcc t1w x3, pansegdata t2w folds 1-2) on KILLARNEY L40S (2026-10-30): B3 was first moved TamIA -> Vulcan
# by vulcan_move.sh (same RUN_IDs, TamIA segments cancelled, fold state pulled + md5-verified on Vulcan); Vulcan's queue then put B3's
# starts at 17:37 MDT .. Oct 31 00:33 MDT while Killarney L40S estimated ~08:18 MDT, so B3 continues on Killarney instead. Run on a
# KILLARNEY login node; Vulcan is reached through klogin02's ssh master (ssh klogin02 ssh vulcan.alliancecan.ca). Order:
#   bash killarney_move.sh prep       # fetch Vulcan's recorded B3 fold cmds, rewrite repo prefix + AugLab pin, verify (diff vs Vulcan cmd,
#                                     #   preprocessed md5 vs Vulcan, pin HEAD / clean / config md5)
#   (cancel the Vulcan B3 jobs, segment 2 first)
#   bash killarney_move.sh pull       # refuses while a Vulcan flataugmv B3 job is queued; rsync fold dirs Vulcan -> Killarney, md5
#   bash killarney_move.sh submit     # 2 guarded 23:59 segments per fold (preflight: every auglab module from the pin)
#   bash killarney_move.sh status
#   ssh klogin02 'systemd-run --user --scope --unit=flatauglab-ksync -- nohup bash <copy>/killarney_move.sh syncback >> <log> 2>&1 &'
#     syncback (klogin02 only: needs its Vulcan master): every 30 min, each finished fold dir (final + best, latest gone, final > 10 min)
#     is rsynced (--delete, that fold dir only) into the SAME Vulcan run dir and md5-verified -> Vulcan's moved_poller.sh (LOCAL_FOLDS
#     0 1 2) sees the folds done and the normal post chain runs. Marker <kmove>/B3/<fold name>.SYNCED_TO_VULCAN.
# AugLab: a detached clone at f018536 on Killarney scratch (the shared Killarney AugLab tree is 7b761b5 and is not touched); the cmd puts it
# first on PYTHONPATH and reads the configs from it. Every module of the old tree exists in the pin (checked), so path order = pin wins.
set -euo pipefail
KREPO=/project/6102268/paulh/mri_synthesis_project; VREPO=/project/aip-jcohen/paulh/mri_synthesis_project
KM=/scratch/paulh/flatauglab_kmove; P="${KM}/B3"; PIN=/scratch/paulh/flatauglab_pin/AugLab
VMV=/scratch/paulh/flatauglab_20261028/vulcan_move/B3
vrun() { if [ "$(hostname -s)" = klogin02 ]; then ssh -n -o BatchMode=yes vulcan.alliancecan.ca "$1"
         else ssh -n klogin02 "ssh -o BatchMode=yes vulcan.alliancecan.ca $(printf '%q' "$1")"; fi; }
RS_E="ssh klogin02 ssh -o BatchMode=yes"; [ "$(hostname -s)" = klogin02 ] && RS_E="ssh -o BatchMode=yes"
AUGLAB_SHA=f018536553ee096fb61d1bf566d70e89c5cec144
CFG=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2FlatFillGPUTransform_train050.json; CFG_MD5=673c0114a1dde8c7a5d0339315fe1b40
VCFG=transform_params_gpu_VALsynthonly_ImageContrastV26_6_2FlatFillGPUTransform.json; VCFG_MD5=3e4a2b74d2083950044f5eb4f1a37670
VJOBS_B3="2030097 2030099 2030101 2030102 2030104 2030106 2030108 2030109 2030111 2030112"
MODE="${1:?usage: prep|pull|submit|status|syncback}"
mkdir -p "${P}"
clean_cmd() { sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$1"; }
getv() { printf '%s\n' "$1" | sed -n "s/^ *export $2=//p" | head -1 | tr -d "'"; }

if [ "${MODE}" = prep ]; then
  [ "$(git -C "${PIN}" rev-parse HEAD)" = "${AUGLAB_SHA}" ] && [ -z "$(git -C "${PIN}" status --porcelain)" ] || { echo "PIN not clean at ${AUGLAB_SHA}"; exit 1; }
  [ "$(md5sum < "${PIN}/auglab/configs/${CFG}" | cut -c1-32)" = "${CFG_MD5}" ] && [ "$(md5sum < "${PIN}/auglab/configs/${VCFG}" | cut -c1-32)" = "${VCFG_MD5}" ] || { echo "PIN config md5"; exit 1; }
  echo "[prep] pin ${PIN} at ${AUGLAB_SHA:0:7}, clean, configs ok"
  rm -rf "${P}/vulcan_cmds"; mkdir -p "${P}/vulcan_cmds"
  vrun "cd ${VMV} && tar cf - index.tsv fold*_auglabAugFlat_*_194832.sh" | tar xf - -C "${P}/vulcan_cmds"
  : > "${P}/index.tsv"; bad=0
  while IFS=$'\t' read -r vcmd vlog name vdone; do
    v="${P}/vulcan_cmds/$(basename "${vcmd}")"; k="${P}/$(basename "${vcmd}")"
    sed -e "s#${VREPO}/sub-workspaces/auglab_workspace/AugLab/auglab/configs/#${PIN}/auglab/configs/#g" \
        -e "s#export PYTHONPATH=\\\\'${VREPO}/#export PYTHONPATH=\\\\'${PIN}:${VREPO}/#" \
        -e "s#${VREPO}#${KREPO}#g" "${v}" > "${k}"
    printf '%s\t%s\t%s\t%s\n' "${k}" "${P}/${name}.train.log" "${name}" "${vdone//${VREPO}/${KREPO}}" >> "${P}/index.tsv"
    c="$(clean_cmd "${k}")"; d="$(diff <(clean_cmd "${v}") <(printf '%s\n' "$c") || true)"
    # allowed differences: the repo prefix, the pin on PYTHONPATH, the two config paths
    rest="$(diff <(clean_cmd "${v}" | sed -e "s#${VREPO}/sub-workspaces/auglab_workspace/AugLab#<AUGLAB>#g" -e "s#${VREPO}#<REPO>#g" -e "s#PYTHONPATH=<REPO>#PYTHONPATH=<AUGLAB>:<REPO>#") \
                 <(printf '%s\n' "$c" | sed -e "s#${PIN}#<AUGLAB>#g" -e "s#${KREPO}#<REPO>#g") || true)"
    pre=$(getv "$c" nnUNet_preprocessed); res=$(getv "$c" nnUNet_results); pp=$(getv "$c" PYTHONPATH)
    ok=OK; [ -z "${rest}" ] || ok="${ok},CMD_DIFF"; [ -d "${pre}" ] || ok="${ok},NO_PREPROC"; [[ "${pp}" == "${PIN}:"* ]] || ok="${ok},PIN_NOT_FIRST(${pp})"
    [ "$(md5sum < "$(getv "$c" AUGLAB_PARAMS_GPU_JSON)" | cut -c1-32)" = "${CFG_MD5}" ] || ok="${ok},CFG"
    [ "$(md5sum < "$(getv "$c" AUGLAB_VAL_PARAMS_GPU_JSON)" | cut -c1-32)" = "${VCFG_MD5}" ] || ok="${ok},VCFG"
    [[ "${res}" == "${KREPO}/benchmark/02_tasks/"*"/auglab/${name#fold?_}" ]] || ok="${ok},RESULTS(${res})"
    grep -q '/project/aip-jcohen' "${k}" && ok="${ok},VULCAN_PATH_LEFT"
    echo "   ${ok} ${name}"; [ -z "${rest}" ] || echo "${rest}" | sed 's/^/      /'; [ "${ok}" = OK ] || bad=1
  done < "${P}/vulcan_cmds/index.tsv"
  # preprocessed parity with Vulcan (dataset ids of the cmds)
  for did in $(cat "${P}"/fold*_194832.sh | grep -oE 'nnUNetv2_train [0-9]+' | awk '{print $2}' | sort -u); do
    f="$(awk -F'\t' 'NR==1{print $1}' "${P}/index.tsv")"
    for k in "${P}"/fold*_194832.sh; do grep -q "nnUNetv2_train ${did} " "$k" && { f="$k"; break; }; done
    kp=$(getv "$(clean_cmd "$f")" nnUNet_preprocessed); vp="${kp//${KREPO}/${VREPO}}"
    dsd="$(cd "$kp" && ls -d Dataset${did}_* | head -1)"
    q='for x in nnUNetPlans.json dataset.json splits_final.json dataset_fingerprint.json; do echo "$(md5sum < $x | cut -c1-32) $x"; done; echo "nfiles $(ls nnUNetPlans_3d_fullres | wc -l) $(du -sb nnUNetPlans_3d_fullres | cut -f1)"'
    L="$(cd "$kp/$dsd" && bash -c "$q")"; R="$(vrun "cd $vp/$dsd && bash -c '$(printf '%s' "$q" | sed "s/'/'\\\\''/g")'")"
    if [ "$L" = "$R" ]; then echo "   preprocessed ${dsd}: identical to Vulcan ($(echo "$L" | tail -1))"; else echo "   PREPROCESSED DIFF ${dsd}"; diff <(echo "$R") <(echo "$L") || true; bad=1; fi
  done
  [ "${bad}" = 0 ] && echo "PREP PASSED" || { echo "PREP FAILED"; exit 1; }
  exit 0
fi

if [ "${MODE}" = pull ]; then
  q="$(vrun "squeue -u paulh -h -o '%i %j %T'")"
  live="$(echo "$q" | awk '{print $1}' | grep -xE "$(echo ${VJOBS_B3} | tr ' ' '|')" || true)"
  [ -z "${live}" ] || { echo "REFUSING: Vulcan B3 jobs still queued: ${live}"; exit 1; }
  while IFS=$'\t' read -r cmd log name done_; do
    kfold="$(dirname "${done_}")"; vfold="${kfold//${KREPO}/${VREPO}}"
    if ls "${kfold}"/checkpoint_*.pth >/dev/null 2>&1 && [ -z "${ALLOW_EXISTING:-}" ]; then echo "REFUSING: ${kfold} already has checkpoints"; exit 2; fi
    mkdir -p "${kfold}"; echo "[$(date '+%F %T')] ${name}: Vulcan ${vfold} -> ${kfold}" | tee -a "${P}/PULL.log"
    rsync -a -e "${RS_E}" --exclude='validation/' "vulcan.alliancecan.ca:${vfold}/" "${kfold}/" < /dev/null
    rsync -a -e "${RS_E}" --include='*.json' --exclude='*' "vulcan.alliancecan.ca:$(dirname "${vfold}")/" "$(dirname "${kfold}")/" < /dev/null
    q2='for f in checkpoint_latest.pth checkpoint_best.pth; do echo $f $(stat -c %s $f) $(md5sum < $f | cut -c1-32); done; ls checkpoint_final.pth 2>/dev/null; true'
    R="$(vrun "cd ${vfold} && ${q2}")"; L="$(cd "${kfold}" && bash -c "${q2}")"
    [ "$R" = "$L" ] && echo "   verified: $(echo "$L" | tr '\n' ' ')" | tee -a "${P}/PULL.log" || { echo "   MISMATCH Vulcan: $R / Killarney: $L" | tee -a "${P}/PULL.log"; exit 5; }
    # and equal to what the Vulcan pull verified against TamIA
    lm="$(echo "$L" | awk '{print $3}' | tr '\n' ' ')"; grep -q "$(echo "$lm" | awk '{print $1" "$2}' | sed 's/ /.*/')" <(vrun "sed -E 's/checkpoint_(latest|best).pth [0-9]+ //g' ${VMV}/PULL.log") \
      && echo "   = TamIA md5 recorded in ${VMV}/PULL.log" | tee -a "${P}/PULL.log" || { echo "   NOT the TamIA md5 of ${VMV}/PULL.log"; exit 5; }
  done < "${P}/index.tsv"
  echo "PULLED $(date '+%F %T')" >> "${P}/PULL.log"; echo "PULL OK"; exit 0
fi

if [ "${MODE}" = submit ]; then
  grep -q '^PULLED' "${P}/PULL.log" 2>/dev/null || { echo "REFUSING: not pulled"; exit 1; }
  [ -f "${P}/JOBS.tsv" ] && { echo "already submitted"; exit 1; }
  while IFS=$'\t' read -r cmd log name done_; do
    fd="$(dirname "${done_}")"; [ -f "${fd}/checkpoint_latest.pth" ] || { echo "REFUSING: no checkpoint_latest in ${fd}"; exit 1; }
    body="${P}/${name}.job.sh"
    cat > "${body}" <<EOS
#!/bin/bash
# flatauglab B3 fold on Killarney (killarney_move.sh): ${name}
set -uo pipefail
FD='${fd}'
if [ -f "\${FD}/checkpoint_final.pth" ] && [ ! -e "\${FD}/checkpoint_latest.pth" ]; then echo "[kmove] \${FD}: finished -- nothing to do"; exit 0; fi
[ -f "\${FD}/checkpoint_latest.pth" ] || { echo "[kmove] no checkpoint_latest in \${FD} -- refusing (would restart from epoch 0)"; exit 3; }
cd '${KREPO}'
PYTHONPATH='${PIN}' .venv/bin/python - <<'PY' || { echo "[kmove] AugLab preflight FAILED -- refusing"; exit 4; }
import importlib, sys
pin = '${PIN}/auglab'
bad = []
for spec in ['auglab.transforms.gpu.transforms', 'auglab.transforms.gpu.palette_noisefill', 'auglab.trainers', 'auglab.configs']:
    m = importlib.import_module(spec); f = getattr(m, '__file__', None) or list(m.__path__)[0]
    print('[kmove] preflight', spec, '->', f); (f.startswith(pin) or bad.append((spec, f)))
sys.exit(1 if bad else 0)
PY
echo "[kmove] host=\$(hostname) job=\${SLURM_JOB_ID} gpu=\${CUDA_VISIBLE_DEVICES:-?} resume ${name}"
bash '${cmd}' >> '${log}' 2>&1
EOS
    common=(--account=aip-jcohen --gres=gpu:l40s:1 --cpus-per-task=16 --mem=110G --time=23:59:00 --job-name="flataugkm_${name%%_auglabAugFlat*}")
    j1="$(sbatch --parsable "${common[@]}" --output="${P}/${name}.seg1_%j.out" "${body}")"; sleep 2
    j2="$(sbatch --parsable "${common[@]}" --dependency=afterany:"${j1}" --output="${P}/${name}.seg2_%j.out" "${body}")"; sleep 2
    printf '%s\t%s\t%s\t%s\n' "${name}" "${j1}" "${j2}" "${fd}" | tee -a "${P}/JOBS.tsv"
  done < "${P}/index.tsv"
  exit 0
fi

if [ "${MODE}" = status ]; then
  while IFS=$'\t' read -r name j1 j2 fd; do
    L="$(ls -t "${fd}"/training_log_*.txt 2>/dev/null | head -1)"
    last=$(grep -oE 'Epoch [0-9]+' "$L" 2>/dev/null | tail -1 | grep -oE '[0-9]+' || echo NA)
    med=$(grep -o "Epoch time: [0-9.]* s" "$L" 2>/dev/null | tail -20 | awk '{print $3}' | sort -n | awk '{a[NR]=$1} END{if(NR) print a[int((NR+1)/2)]; else print "NA"}')
    fin=no; [ -f "${fd}/checkpoint_final.pth" ] && [ ! -e "${fd}/checkpoint_latest.pth" ] && fin=yes
    printf '%-62s epoch=%-5s s/ep(last20)=%-6s final=%s synced=%s jobs=%s,%s\n' "${name%%_auglabAugFlat*}" "$last" "$med" "$fin" \
      "$([ -f "${P}/${name}.SYNCED_TO_VULCAN" ] && echo yes || echo no)" "$j1" "$j2"
  done < "${P}/JOBS.tsv"
  exit 0
fi

if [ "${MODE}" = syncback ]; then
  exec 6> "${KM}/syncback.lock"; flock -n 6 || { echo "another syncback holds the lock"; exit 1; }
  log() { echo "[$(date '+%F %T')] [ksync] $*"; }
  t_end=$(( $(date +%s) + 9 * 86400 )); log "start pid $$ on $(hostname -s)"
  while :; do
    [ -f "${KM}/STOP_SYNCBACK" ] && { log "STOP file -- exiting"; break; }
    [ "$(date +%s)" -lt "${t_end}" ] || { log "max lifetime -- exiting"; break; }
    left=0; now=$(date +%s)
    while IFS=$'\t' read -r name j1 j2 fd; do
      [ -f "${P}/${name}.SYNCED_TO_VULCAN" ] && continue; left=$((left + 1))
      [ -f "${fd}/checkpoint_final.pth" ] && [ -f "${fd}/checkpoint_best.pth" ] && [ ! -e "${fd}/checkpoint_latest.pth" ] \
        && [ $(( now - $(stat -c %Y "${fd}/checkpoint_final.pth") )) -gt 600 ] || continue
      vfd="${fd//${KREPO}/${VREPO}}"
      log "${name}: finished -> rsync --delete to Vulcan ${vfd}"
      if ! rsync -a --delete -e "${RS_E}" "${fd}/" "vulcan.alliancecan.ca:${vfd}/" < /dev/null; then log "${name}: rsync failed (retry next cycle)"; continue; fi
      q='for f in checkpoint_final.pth checkpoint_best.pth; do echo $f $(stat -c %s $f) $(md5sum < $f | cut -c1-32); done; ls checkpoint_latest.pth 2>/dev/null; true'
      L="$(cd "${fd}" && bash -c "$q")"; R="$(vrun "cd ${vfd} && ${q}")" || { log "${name}: md5 call failed (retry)"; continue; }
      if [ "$L" = "$R" ]; then echo "$(date '+%F %T') ${vfd}"$'\n'"$L" > "${P}/${name}.SYNCED_TO_VULCAN"; log "${name}: SYNCED ($(echo "$L" | tr '\n' ' '))"; left=$((left - 1))
      else log "${name}: md5 MISMATCH Killarney '$L' vs Vulcan '$R' (retry next cycle)"; fi
    done < "${P}/JOBS.tsv"
    [ "${left}" -gt 0 ] || { log "all B3 folds synced to Vulcan -- exiting"; break; }
    for _ in $(seq 60); do [ -f "${KM}/STOP_SYNCBACK" ] && break; sleep 30; done
  done
  exit 0
fi
echo "unknown mode"; exit 1
