#!/usr/bin/env bash
# Move WHOLE flatauglab TamIA packs to Vulcan L40S (2026-10-30, to lighten TamIA): the unfinished folds of packs B2 and B3 resume on
# Vulcan with the SAME RUN_IDs from the TamIA checkpoint_latest (nnU-Net --c). One 1-GPU job per fold + a second 23:59 segment
# (afterany), each guarded by "checkpoint_final present and checkpoint_latest gone -> exit 0" so a finished fold never re-runs
# on_train_end / validation. Folds finished on TamIA before the move (on-harmony dwi_ap, pelvis ct/mri) are not touched here.
# Run on the VULCAN login node, in this order:
#   bash scripts/cluster/flatauglab/vulcan_move.sh record      # re-record the folds with the dataset wrappers (Vulcan paths) + verify
#   bash scripts/cluster/flatauglab/vulcan_move.sh datacheck   # TamIA vs Vulcan: preprocessed plans/splits/dataset md5, cmd diff, venv
#   (cancel the packs' pending TamIA segments, tail first: scancel 651752 651755; then 651751 651754)
#   bash scripts/cluster/flatauglab/vulcan_move.sh pull        # refuses while a flatauglab_B2/B3 job is queued on TamIA; rsync + md5
#   bash scripts/cluster/flatauglab/vulcan_move.sh submit      # 2 chained sbatch segments per fold
#   bash scripts/cluster/flatauglab/vulcan_move.sh status      # epoch reached per fold (Vulcan) + done
# Records: /scratch/paulh/flatauglab_20261028/vulcan_move/<pack>/ (index.tsv, fold cmds, PULL.log, JOBS.tsv).
# Post-training: moved_poller.sh (writes <companions>/<setting>/SYNCED from the Vulcan checkpoints; companions_vulcan.sh LOCAL_FOLDS mode).
set -euo pipefail
REPO=/project/aip-jcohen/paulh/mri_synthesis_project; cd "${REPO}"
export SCRATCH="${SCRATCH:-/scratch/paulh}"
REC=/scratch/paulh/flatauglab_20261028; MV="${REC}/vulcan_move"; mkdir -p "${MV}"
TS="$(cat "${REC}/TS")"; TROOT="/scratch/p/paulh/_packruns/flatauglab_${TS}"
TAMIA=tamia.alliancecan.ca; SSH=(ssh -n -o BatchMode=yes -o ConnectTimeout=30 "${TAMIA}")
METHOD=auglabAugFlat_v26_6_2_train050_val000
ACFG=sub-workspaces/auglab_workspace/AugLab/auglab/configs
CFG=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2FlatFillGPUTransform_train050.json; CFG_MD5=673c0114a1dde8c7a5d0339315fe1b40
VCFG=transform_params_gpu_VALsynthonly_ImageContrastV26_6_2FlatFillGPUTransform.json; VCFG_MD5=3e4a2b74d2083950044f5eb4f1a37670
MODE="${1:?usage: record|datacheck|pull|submit|status}"
T=benchmark/02_tasks
O=$T/brain_healthy/on-harmony/5_scripts_on-harmony; M=$T/brain_ms/open-ms/5_scripts_open-ms
A=$T/liver_hcc/atlas-liver-hcc/5_scripts_atlas-liver-hcc; PS=$T/pancreas_disease/pansegdata/5_scripts_pansegdata
# setting | scripts dir | train wrapper | expected trainer | epochs | DA | compile | task dir   (rows copied from launch.sh)
SETTINGS=(
"onh_T2w|$O|04_89_train_t2w_${METHOD}.sh|nnUNetTrainerOnHarmonyAugLabDefault|2000|0|0|brain_healthy/on-harmony"
"openms_t1w|$M|04_70_train_t1w_${METHOD}.sh|nnUNetTrainerOpenMSAugLabDefault|2000|8|1|brain_ms/open-ms"
"atlas_t1w|$A|04_22_train_t1w_${METHOD}.sh|nnUNetTrainerAtlasLiverHCCAugLabDualVal|2000|8|1|liver_hcc/atlas-liver-hcc"
"panseg_t1wce|$PS|04_41_train_t1wce_${METHOD}.sh|nnUNetTrainerPANSEGDATAAugLabDualVal|2000|8|1|pancreas_disease/pansegdata"
"panseg_t2w|$PS|04_42_train_t2w_${METHOD}.sh|nnUNetTrainerPANSEGDATAAugLabDualVal|2000|8|1|pancreas_disease/pansegdata"
)
# moved pack | its unfinished folds at the move (the TamIA pack's finished folds stay on TamIA)
MOVED=(
"B2|onh_T2w:0 onh_T2w:1 onh_T2w:2 openms_t1w:0 openms_t1w:1 openms_t1w:2 panseg_t1wce:2 panseg_t2w:0"
"B3|atlas_t1w:0 atlas_t1w:1 atlas_t1w:2 panseg_t2w:1 panseg_t2w:2"
)
TAMIA_JOBS="651751 651752 651754 651755"
srow() { local s; for s in "${SETTINGS[@]}"; do [ "${s%%|*}" = "$1" ] && { echo "$s"; return 0; }; done; return 1; }
clean_cmd() { sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$1"; }
getv() { printf '%s\n' "$1" | sed -n "s/^ *export $2=//p" | head -1 | tr -d "'"; }
# TamIA fold dir of <run_id> fold <k> = dirname of the done column of the TamIA pack index
tamia_index() { "${SSH[@]}" "cat ${TROOT}/$1/index.tsv"; }

if [ "${MODE}" = record ]; then
  for pk in "${MOVED[@]}"; do p="${pk%%|*}"; spec="${pk#*|}"; PD="${MV}/${p}"
    [ -f "${PD}/index.tsv" ] && { echo "[record] ${p} already recorded (${PD}) -- skipped"; continue; }
    for sf in ${spec}; do s="${sf%%:*}"; k="${sf#*:}"
      IFS='|' read -r _ dir tw etr ep eda ecmp task <<<"$(srow "$s")"
      wrapper="${dir}/04_train/${tw}"; [ -f "${wrapper}" ] || { echo "MISSING ${wrapper}"; exit 1; }
      envf="$(grep -o 'source "$(dirname "$0")/../00_utils/env[a-z0-9_]*\.sh"' "${wrapper}" | sed 's|.*/00_utils/||; s|"$||')"
      echo "[record] ${p} <- ${s} fold ${k} ($(basename "${wrapper}"), env ${envf})"
      ( unset PREDICTIONS_ROOT METRICS_ROOT nnUNet_raw nnUNet_preprocessed nnUNet_results RUN_JOB_INLINE
        source "${dir}/00_utils/${envf}"
        export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"
        TRAIN_FOLDS="${k}" RUN_JOB_PACK_DIR="${PD}" bash "${wrapper}" "${DATASET_NAME}_${TRAINING_CONTRAST}_${METHOD}_${TS}" < /dev/null )
    done
  done
  set -- verify
  MODE=verify
fi

if [ "${MODE}" = verify ]; then
  bad=0
  for pk in "${MOVED[@]}"; do p="${pk%%|*}"; read -ra SP <<<"${pk#*|}"; PD="${MV}/${p}"
    n=$(grep -c . "${PD}/index.tsv"); echo "-- ${p}: ${n} folds (expect ${#SP[@]})"; [ "$n" = "${#SP[@]}" ] || { echo "   FOLD COUNT MISMATCH"; bad=1; }
    i=0
    while IFS=$'\t' read -r cmd log name done_; do
      c="$(clean_cmd "$cmd")"; s="${SP[$i]%%:*}"; k="${SP[$i]#*:}"
      IFS='|' read -r _ dir tw etr eep eda ecmp task <<<"$(srow "$s")"
      ds="$(basename "${task}")"; tc="${s#*_}"; rid="${ds}_${tc}_${METHOD}_${TS}"
      pre=$(getv "$c" nnUNet_preprocessed); res=$(getv "$c" nnUNet_results); cfgp=$(getv "$c" AUGLAB_PARAMS_GPU_JSON)
      vcfg=$(getv "$c" AUGLAB_VAL_PARAMS_GPU_JSON); ep=$(getv "$c" NNUNET_NUM_EPOCHS); da=$(getv "$c" nnUNet_n_proc_DA)
      cm=$(getv "$c" nnUNet_compile); so=$(getv "$c" NNUNET_SEED_OFFSET)
      tr=$(printf '%s\n' "$c" | grep -o '\-tr [A-Za-z0-9_]*' | head -1 | cut -d' ' -f2)
      fl=$(printf '%s\n' "$c" | grep -oE 'nnUNetv2_train [0-9]+ 3d_fullres [0-9]+' | head -1 | awk '{print $4}')
      ok=OK
      [[ "$res" == "${REPO}/${T}/${task}/8_results_${ds}/01_predictions/"*"/${tc}/auglab/${rid}" ]] || ok="${ok},RESULTS($res)"
      [ -d "$pre" ] || ok="${ok},NO_PREPROC($pre)"
      [ "$name" = "fold${k}_${rid}" ] || ok="${ok},NAME($name)"; [ "$fl" = "$k" ] || ok="${ok},TRAIN_FOLD($fl)"
      [ "${cfgp##*/}" = "${CFG}" ] && [ "$(md5sum < "$cfgp" | cut -c1-32)" = "${CFG_MD5}" ] || ok="${ok},CFG($cfgp)"
      if [[ "$etr" == *DualVal ]]; then [ "${vcfg##*/}" = "${VCFG}" ] && [ "$(md5sum < "$vcfg" | cut -c1-32)" = "${VCFG_MD5}" ] || ok="${ok},VCFG($vcfg)"
      else [ -z "$vcfg" ] || ok="${ok},VCFG_SET"; fi
      [ "$tr" = "$etr" ] || ok="${ok},TRAINER($tr)"; [ "$ep" = "$eep" ] || ok="${ok},EPOCHS($ep)"
      [ "$da" = "$eda" ] && [ "$cm" = "$ecmp" ] || ok="${ok},LOADER($da/$cm)"; [ "${so:-0}" = 0 ] || ok="${ok},SEED($so)"
      grep -q 'checkpoint_latest.pth' "$cmd" || ok="${ok},NO_RESUME_FLAG"
      printf '   %-6s %-13s %s\n      trainer=%s ep=%s DA=%s compile=%s results=%s\n' "$ok" "$s" "$name" "$tr" "$ep" "$da" "$cm" "$res"
      [ "$ok" = OK ] || bad=1; i=$((i+1))
    done < "${PD}/index.tsv"
  done
  [ "$bad" = 0 ] && echo "VERIFY PASSED" || { echo "VERIFY FAILED"; exit 1; }
  exit 0
fi

if [ "${MODE}" = datacheck ]; then
  bad=0; declare -A seen=()
  for pk in "${MOVED[@]}"; do p="${pk%%|*}"; PD="${MV}/${p}"
    TIDX="$(tamia_index "${p}")" || { echo "connection problem"; exit 4; }
    while IFS=$'\t' read -r cmd log name done_; do
      tcmd="$(echo "${TIDX}" | awk -F'\t' -v n="${name}" '$3==n {print $1}')"
      [ -n "${tcmd}" ] || { echo "BAD: ${name} not in the TamIA ${p} index"; bad=1; continue; }
      c="$(clean_cmd "$cmd")"; tc_="$("${SSH[@]}" "cat '${tcmd}'" | sed -e 's/\\n/\n/g' -e "s/\\\\'//g")"
      pre=$(getv "$c" nnUNet_preprocessed); tpre=$(getv "$tc_" nnUNet_preprocessed)
      did=$(printf '%s\n' "$c" | grep -oE 'nnUNetv2_train [0-9]+' | head -1 | awk '{print $2}')
      echo "== ${name}"
      # cmd diff with the per-cluster data roots normalised; only paths / [code] / [loader]-free lines may differ
      norm() { local e=() v; for v in nnUNet_raw:RAW nnUNet_preprocessed:PRE nnUNet_results:RES SPLITS_DIR:SPLITS; do
                 local x; x="$(getv "$1" "${v%%:*}")"; [ -n "$x" ] && e+=(-e "s#${x}#<${v#*:}>#g"); done
               sed "${e[@]}" -e '/^ *echo "\[code\]/d'; }
      d="$(diff <(printf '%s\n' "$tc_" | norm "$tc_") <(printf '%s\n' "$c" | norm "$c") || true)"
      if [ -n "$d" ]; then echo "   CMD DIFF (TamIA < > Vulcan):"; echo "$d" | sed 's/^/     /'; bad=1; else echo "   cmd identical up to data roots"; fi
      key="${did}"; [ -n "${seen[$key]:-}" ] && continue; seen[$key]=1
      dsd="$(cd "$pre" && ls -d Dataset${did}_* | head -1)"
      L="$(cd "$pre/$dsd" && for f in nnUNetPlans.json dataset.json splits_final.json dataset_fingerprint.json; do printf '%s %s\n' "$(md5sum < "$f" | cut -c1-32)" "$f"; done; echo "nfiles $(ls nnUNetPlans_3d_fullres | wc -l)")"
      R="$("${SSH[@]}" "cd '$tpre/$dsd' && for f in nnUNetPlans.json dataset.json splits_final.json dataset_fingerprint.json; do printf '%s %s\n' \"\$(md5sum < \$f | cut -c1-32)\" \$f; done; echo nfiles \$(ls nnUNetPlans_3d_fullres | wc -l)")"
      if [ "$L" = "$R" ]; then echo "   preprocessed ${dsd}: identical"; echo "$L" | sed 's/^/     /'
      else echo "   PREPROCESSED DIFF ${dsd}:"; diff <(echo "$R") <(echo "$L") | sed "s/^/     /" || true; bad=1; fi
    done < "${PD}/index.tsv"
  done
  echo "== venv + code"
  PKGS='torch|nnunetv2|kornia|batchgeneratorsv2|batchgenerators|numpy|blosc2|dynamic-network-architectures|acvl-utils|monai|SimpleITK'
  LV="$(.venv/bin/pip freeze 2>/dev/null | grep -iE "^(${PKGS})==" | sort)"
  RV="$("${SSH[@]}" "cd ${REPO} && .venv/bin/pip freeze 2>/dev/null" | grep -iE "^(${PKGS})==" | sort)"
  [ "$LV" = "$RV" ] && { echo "   pip versions identical:"; echo "$LV" | tr '\n' ' '; echo; } || { echo "   PIP DIFF (TamIA < > Vulcan):"; diff <(echo "$RV") <(echo "$LV") || true; [ "$(diff <(echo "$RV" | grep -vi simpleitk) <(echo "$LV" | grep -vi simpleitk))" = "" ] && echo "   (SimpleITK only: NIfTI I/O, not used by training on preprocessed data -- accepted)" || bad=1; }
  FILES="nnunetv2/training/logging/nnunet_logger.py nnunetv2/training/nnUNetTrainer/OnHarmonyTrainers.py nnunetv2/training/nnUNetTrainer/OpenMSTrainers.py nnunetv2/training/nnUNetTrainer/AtlasLiverHCCTrainers.py nnunetv2/training/nnUNetTrainer/PANSEGDATATrainers.py"
  LM="$(cd .venv/lib/python3.*/site-packages && md5sum ${FILES})"; RM="$("${SSH[@]}" "cd ${REPO}/.venv/lib/python3.*/site-packages && md5sum ${FILES}")"
  if [ "$LM" = "$RM" ]; then echo "   logger patch + 4 trainer shims identical"
  else  # shims differ: accept only path / comment lines (the shims' own _scripts path is absent on both clusters; classes import via the cmd PYTHONPATH)
    for f in ${FILES}; do lf=$(echo .venv/lib/python3.*/site-packages/$f)
      dl="$(diff <("${SSH[@]}" "cat ${REPO}/.venv/lib/python3.*/site-packages/$f") "$lf" | grep '^[<>]' || true)"
      [ -z "$dl" ] && continue
      sub="$(echo "$dl" | grep -vE 'datasets|benchmark|CLAUDE.md|project notes' || true)"
      if [ -z "$sub" ]; then echo "   shim ${f##*/}: differs only in path/comment lines (inert) -- accepted"; echo "$dl" | sed 's/^/        /'
      else echo "   SHIM ${f##*/} differs in code:"; echo "$sub"; bad=1; fi
    done
  fi
  TH="$("${SSH[@]}" "cd ${REPO} && git rev-parse HEAD && git -C sub-workspaces/auglab_workspace/AugLab rev-parse HEAD")"
  echo "   TamIA main/AugLab: $(echo $TH)   Vulcan main/AugLab: $(git rev-parse HEAD) $(git -C sub-workspaces/auglab_workspace/AugLab rev-parse HEAD) (AugLab dirty=$(git -C sub-workspaces/auglab_workspace/AugLab status --porcelain | wc -l))"
  th="$(echo "$TH" | head -1)"
  TP='^(src/|benchmark/00_commun_scripts/00_01_train/|benchmark/00_commun_scripts/00_00_utils/common_env.sh|scripts/job_runner/|benchmark/02_tasks/.*/(04_train|00_utils|trainers)/)'
  chg="$(git diff --name-only "$th" HEAD | grep -E "$TP" || true)"
  dirty="$( { git status --porcelain | awk '{print $NF}' | grep -E "$TP"; git -C sub-workspaces/auglab_workspace/AugLab status --porcelain; } || true)"
  [ -z "$chg" ] && echo "   main ${th:0:7}..HEAD: no training-path change" || { echo "   TRAINING-PATH CHANGES since TamIA HEAD:"; echo "$chg"; bad=1; }
  [ -z "$dirty" ] && echo "   no uncommitted training-path change on Vulcan" || { echo "   DIRTY training paths:"; echo "$dirty"; bad=1; }
  [ "$bad" = 0 ] && echo "DATACHECK PASSED" || { echo "DATACHECK FAILED"; exit 1; }
  exit 0
fi

if [ "${MODE}" = pull ]; then
  q="$("${SSH[@]}" "squeue -u paulh -h -o '%i %j %T'")" || { echo "connection problem"; exit 4; }
  if echo "$q" | grep -qE 'flatauglab_B[23]( |$)'; then echo "REFUSING: TamIA still has B2/B3 jobs queued:"; echo "$q" | grep -E 'flatauglab_B[23]'; exit 1; fi
  for pk in "${MOVED[@]}"; do p="${pk%%|*}"; PD="${MV}/${p}"; PL="${PD}/PULL.log"
    TIDX="$(tamia_index "${p}")"
    while IFS=$'\t' read -r cmd log name done_; do
      res=$(getv "$(clean_cmd "$cmd")" nnUNet_results)
      tdone="$(echo "${TIDX}" | awk -F'\t' -v n="${name}" '$3==n {print $4}')"; tfold="$(dirname "${tdone}")"
      trun="${tdone%%/Dataset[0-9]*}"; rel="${tfold#${trun}/}"; ttr="$(dirname "${tfold}")"
      [ "$(basename "$trun")" = "$(basename "$res")" ] || { echo "BAD run id ${trun} vs ${res}"; exit 2; }
      dest="${res}/${rel}"
      if ls "${dest}"/checkpoint_*.pth >/dev/null 2>&1 && [ -z "${ALLOW_EXISTING:-}" ]; then echo "REFUSING: ${dest} already has checkpoints"; exit 2; fi
      mkdir -p "${dest}"
      echo "[$(date '+%F %T')] ${name}: ${TAMIA}:${tfold} -> ${dest}" | tee -a "${PL}"
      rsync -a --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=30" --exclude='validation/' "${TAMIA}:${tfold}/" "${dest}/"
      rsync -a -e "ssh -o BatchMode=yes -o ConnectTimeout=30" --include='*.json' --exclude='*' "${TAMIA}:${ttr}/" "$(dirname "${dest}")/"
      R="$("${SSH[@]}" "cd '${tfold}' && for f in checkpoint_latest.pth checkpoint_best.pth; do echo \$f \$(stat -c %s \$f) \$(md5sum < \$f | cut -c1-32); done; grep -oE 'Epoch [0-9]+' \$(ls -t training_log_*.txt | head -1) | tail -1")"
      Lc="$(cd "${dest}" && for f in checkpoint_latest.pth checkpoint_best.pth; do echo $f $(stat -c %s $f) $(md5sum < $f | cut -c1-32); done; grep -oE 'Epoch [0-9]+' $(ls -t training_log_*.txt | head -1) | tail -1)"
      if [ "$R" = "$Lc" ]; then echo "   verified: $(echo "$Lc" | tr '\n' ' ')" | tee -a "${PL}"
      else echo "   MISMATCH TamIA: $(echo "$R" | tr '\n' ' ') Vulcan: $(echo "$Lc" | tr '\n' ' ')" | tee -a "${PL}"; exit 5; fi
      [ ! -e "${dest}/checkpoint_final.pth" ] || { echo "   UNEXPECTED checkpoint_final in ${dest}"; exit 5; }
    done < "${PD}/index.tsv"
    echo "PULLED $(date '+%F %T')" >> "${PL}"
  done
  echo "PULL OK"; exit 0
fi

if [ "${MODE}" = submit ]; then
  for pk in "${MOVED[@]}"; do p="${pk%%|*}"; PD="${MV}/${p}"
    grep -q '^PULLED' "${PD}/PULL.log" 2>/dev/null || { echo "REFUSING: ${p} not pulled"; exit 1; }
    [ -f "${PD}/JOBS.tsv" ] && { echo "[submit] ${p} already submitted ($(wc -l < "${PD}/JOBS.tsv") folds) -- skipped"; continue; }
    while IFS=$'\t' read -r cmd log name done_; do
      res=$(getv "$(clean_cmd "$cmd")" nnUNet_results); fold="${name#fold}"; fold="${fold%%_*}"
      fd="$(ls -d "${res}"/Dataset*/*__nnUNetPlans__3d_fullres/fold_${fold})"
      [ -f "${fd}/checkpoint_latest.pth" ] || { echo "REFUSING: no checkpoint_latest in ${fd}"; exit 1; }
      body="${PD}/${name}.job.sh"
      cat > "${body}" <<EOS
#!/bin/bash
# flatauglab moved fold (vulcan_move.sh): ${name}
set -uo pipefail
FD='${fd}'
if [ -f "\${FD}/checkpoint_final.pth" ] && [ ! -e "\${FD}/checkpoint_latest.pth" ]; then echo "[moved] \${FD}: finished -- nothing to do"; exit 0; fi
[ -f "\${FD}/checkpoint_latest.pth" ] || { echo "[moved] no checkpoint_latest in \${FD} -- refusing (would restart from epoch 0)"; exit 3; }
echo "[moved] host=\$(hostname) job=\${SLURM_JOB_ID} gpu=\${CUDA_VISIBLE_DEVICES:-?} resume ${name}"
bash '${cmd}' >> '${PD}/${name}.train.log' 2>&1
EOS
      common=(--account=aip-jcohen --gres=gpu:l40s:1 --cpus-per-task=16 --mem=110G --time=23:59:00 --exclude=rack02-06 --job-name="flataugmv_${name%%_auglabAugFlat*}")
      j1="$(sbatch --parsable "${common[@]}" --output="${PD}/${name}.seg1_%j.out" "${body}")"; sleep 2
      j2="$(sbatch --parsable "${common[@]}" --dependency=afterany:"${j1}" --output="${PD}/${name}.seg2_%j.out" "${body}")"; sleep 2
      printf '%s\t%s\t%s\t%s\n' "${name}" "${j1}" "${j2}" "${fd}" | tee -a "${PD}/JOBS.tsv"
    done < "${PD}/index.tsv"
  done
  exit 0
fi

if [ "${MODE}" = status ]; then
  for pk in "${MOVED[@]}"; do p="${pk%%|*}"; PD="${MV}/${p}"; echo "-- ${p}"
    [ -f "${PD}/JOBS.tsv" ] || { echo "   not submitted"; continue; }
    while IFS=$'\t' read -r name j1 j2 fd; do
      L="$(ls -t "${fd}"/training_log_*.txt 2>/dev/null | head -1)"
      last=$(grep -oE 'Epoch [0-9]+' "$L" 2>/dev/null | tail -1 | grep -oE '[0-9]+' || echo NA)
      med=$(grep -o "Epoch time: [0-9.]* s" "$L" 2>/dev/null | tail -20 | awk '{print $3}' | sort -n | awk '{a[NR]=$1} END{if(NR) print a[int((NR+1)/2)]; else print "NA"}')
      fin=no; [ -f "${fd}/checkpoint_final.pth" ] && [ ! -e "${fd}/checkpoint_latest.pth" ] && fin=yes
      printf '   %-62s epoch=%-5s s/ep(last20)=%-6s final=%s jobs=%s,%s\n' "${name%%_auglabAugFlat*}" "$last" "$med" "$fin" "$j1" "$j2"
    done < "${PD}/JOBS.tsv"
  done
  exit 0
fi
echo "unknown mode ${MODE}"; exit 1
