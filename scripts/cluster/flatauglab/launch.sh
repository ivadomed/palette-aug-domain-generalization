#!/usr/bin/env bash
# TamIA launcher of the "flat fill + Auglab" arm (flatauglab, 2026-10-28): METHOD auglabAugFlat_v26_6_2_train050_val000 = the HEADLINE
# PA+Auglab method with PA's real fill replaced by the flat fill of the paper's texture ablation (ladder rung 4.5), everything else identical
# (AugLab configs ..._ImageContrastV26_6_2FlatFillGPUTransform_train050.json + VALsynthonly_..._FlatFillGPUTransform.json, proven by
# scripts/cluster/flatauglab/make_config.py). CLONE of scripts/cluster/sem50/launch.sh (itself pa_em / ssem50 + their fixes: same 19 settings
# x folds 0-2, same pack layout, first-segment dependency for short settings, atlas own test set in queue-post, chaos eval 64000M, runtable).
# Differences vs sem50: per-setting trainer = the HEADLINE PA+Auglab run's trainer (DualVal on 13 settings, AugLabDefault / CHAOSAugLabV26_6_2
# on 6), so the verify step expects the flat VAL config on DualVal rows and none elsewhere; chaos fov_crop RUNS_FILE uses CHAOSAugLabV26_6_2.
# Every fold is RECORDED by its setting's generated train wrapper (scripts/cluster/flatauglab/make_wrappers.py = the HEADLINE PA+Auglab
# wrapper with only METHOD / config(s) / LOG_DIR / header changed; chaos t1in also drops the SINGLE_* one-fold pins, 300 -> 200 epochs),
# one RUN_ID per setting (<DATASET_NAME>_<TRAINING_CONTRAST>_<METHOD>_<TS>, TS fixed at the first record of the root). Reported checkpoint:
# checkpoint_best (val000, real validation); DualVal trainers also hard-link a ..._val100_ mirror at the end of training (not used).
#   bash scripts/cluster/flatauglab/launch.sh record [PACK...]       # record + verify, no submit
#   bash scripts/cluster/flatauglab/launch.sh submit [PACK...]       # submit the chains
#   bash scripts/cluster/flatauglab/launch.sh queue-post [GROUP...]  # per-setting predict (whole node) + eval (CPU) behind the chains
#   bash scripts/cluster/flatauglab/launch.sh status|probe-report    # epochs reached + median s/epoch per recorded fold
#   bash scripts/cluster/flatauglab/launch.sh runids <out.tsv> | runtable <out.tsv> | chaos-fov-runs
#   bash scripts/cluster/flatauglab/launch.sh queue-companions [GROUP...]  # TamIA-staged companions (mandible / chaos_fov / hcc) behind the own predict jobs
# PROBE=1 (record / submit / probe-report): pack P_probe = fold 0 of Brain T1w, Glioma T1n, HCC, Breast t1wce, PROBE_EPOCHS (default 25) epochs,
#   throwaway results base ${SCRATCH}/_flatauglab_probe, own state file, 1 h 30 job.
# FLATAUG_STATE (optional): state-file override.
# Layout = sem50 / ssem50's layout unchanged (ssem50 measured walls: Glioma packs ~34 h, Brain packs ~38.5 h; PA+Auglab per epoch ~1.02x
#   auglab_default and 0.82x SynthSeg-EM (latency probe 2026-10-15), DualVal's 2nd validation pass ~ +12 %/epoch -> <= ~43 h, inside the
#   3 x 23:59 chain). 21 long folds (Glioma 12, Brain 9) one per GPU on 6 nodes; each long GPU also carries ONE short fold (toothfairy2, ispy2,
#   chaos, pelvis); the three GPUs without a long fold carry open-ms flair x3, open-ms t1w x3, HCC x3 + two pansegdata folds each.
set -euo pipefail
cd "${FLATAUG_REPO:-/project/aip-jcohen/paulh/mri_synthesis_project}"   # FLATAUG_REPO: a code-only staging tree for a dry-run record
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"
TAG=flatauglab
METHOD=auglabAugFlat_v26_6_2_train050_val000
CFG=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2FlatFillGPUTransform_train050.json
CFG_MD5=673c0114a1dde8c7a5d0339315fe1b40
VCFG=transform_params_gpu_VALsynthonly_ImageContrastV26_6_2FlatFillGPUTransform.json
VCFG_MD5=3e4a2b74d2083950044f5eb4f1a37670
STATE="${FLATAUG_STATE:-${SCRATCH}/_packruns_${TAG}_root${PROBE:+_probe}.txt}"
MODE="${1:?usage: record|submit|queue-post|queue-companions|status|probe-report|runids|runtable|chaos-fov-runs [PACK...]}"; shift
T=benchmark/02_tasks
# setting | scripts dir | train wrapper (04_train/) | tamia env | expected trainer | epochs | DA workers | compile   (expected values typed
# here independently of the wrappers: trainer = the headline PA+Auglab run's trainer dir, epochs = policy (= the headline runs); DA/compile =
# the dataset's 04_00_common.sh). A DualVal trainer must carry the flat VAL config (VCFG), any other trainer none.
B=$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma; O=$T/brain_healthy/on-harmony/5_scripts_on-harmony
M=$T/brain_ms/open-ms/5_scripts_open-ms; C=$T/abdomen_healthy/chaos/5_scripts_chaos; TF=$T/mandible_healthy/toothfairy2/5_scripts_toothfairy2
I=$T/breast_cancer/ispy2/5_scripts_ispy2; P=$T/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic; A=$T/liver_hcc/atlas-liver-hcc/5_scripts_atlas-liver-hcc
PS=$T/pancreas_disease/pansegdata/5_scripts_pansegdata
SETTINGS=(
"brats_t1n|$B|04_66_train_t1n_${METHOD}.sh|tamia_env.sh|nnUNetTrainerBraTS2024GliomaAugLabDefault|2500|0|0"
"brats_t2w|$B|04_67_train_t2w_${METHOD}.sh|tamia_env.sh|nnUNetTrainerBraTS2024GliomaT2wAugLabDualVal|2500|0|0"
"brats_t2f|$B|04_68_train_t2f_${METHOD}.sh|tamia_env.sh|nnUNetTrainerBraTS2024GliomaT2fAugLabDualVal|2500|0|0"
"brats_t1c|$B|04_76_train_t1c_${METHOD}.sh|tamia_env.sh|nnUNetTrainerBraTS2024GliomaT1cAugLabDualVal|2500|0|0"
"chaos_t1in|$C|04_89_train_t1in_${METHOD}.sh|tamia_env_chaos.sh|nnUNetTrainerCHAOSAugLabV26_6_2|200|0|0"
"chaos_t2spir|$C|04_90_train_t2spir_${METHOD}.sh|tamia_env_chaos.sh|nnUNetTrainerCHAOSAugLabV26_6_2|200|0|0"
"onh_T1w|$O|04_88_train_t1w_${METHOD}.sh|tamia_env_onharmony.sh|nnUNetTrainerOnHarmonyAugLabDualVal|2000|0|0"
"onh_T2w|$O|04_89_train_t2w_${METHOD}.sh|tamia_env_onharmony.sh|nnUNetTrainerOnHarmonyAugLabDefault|2000|0|0"
"onh_dwi_ap|$O|04_90_train_dwi_ap_${METHOD}.sh|tamia_env_onharmony.sh|nnUNetTrainerOnHarmonyAugLabDualVal|2000|0|0"
"openms_flair|$M|04_69_train_flair_${METHOD}.sh|tamia_env_openms.sh|nnUNetTrainerOpenMSAugLabDefault|2000|8|1"
"openms_t1w|$M|04_70_train_t1w_${METHOD}.sh|tamia_env_openms.sh|nnUNetTrainerOpenMSAugLabDefault|2000|8|1"
"tf2_cbct|$TF|04_26_train_cbct_${METHOD}.sh|tamia_env_toothfairy2.sh|nnUNetTrainerToothFairy2AugLabDualVal|1000|8|1"
"ispy2_t1wce|$I|04_44_train_t1wce_${METHOD}.sh|tamia_env_ispy2.sh|nnUNetTrainerISPY2AugLabDualVal|1000|8|1"
"ispy2_t2w|$I|04_45_train_t2w_${METHOD}.sh|tamia_env_ispy2.sh|nnUNetTrainerISPY2AugLabDualVal|1000|8|1"
"pelvic_ct|$P|04_41_train_ct_${METHOD}.sh|tamia_env_totalseg-pelvic.sh|nnUNetTrainerTotalsegPelvicAugLabDualVal|200|8|1"
"pelvic_mri|$P|04_42_train_mri_${METHOD}.sh|tamia_env_totalseg-pelvic.sh|nnUNetTrainerTotalsegPelvicAugLabDualVal|200|8|1"
"atlas_t1w|$A|04_22_train_t1w_${METHOD}.sh|tamia_env_atlas-liver-hcc.sh|nnUNetTrainerAtlasLiverHCCAugLabDualVal|2000|8|1"
"panseg_t1wce|$PS|04_41_train_t1wce_${METHOD}.sh|tamia_env_pansegdata.sh|nnUNetTrainerPANSEGDATAAugLabDualVal|2000|8|1"
"panseg_t2w|$PS|04_42_train_t2w_${METHOD}.sh|tamia_env_pansegdata.sh|nnUNetTrainerPANSEGDATAAugLabDualVal|2000|8|1"
)
declare -A DSNAME=([brats]=brats2024-glioma [chaos]=chaos [onh]=on-harmony [openms]=open-ms [tf2]=toothfairy2 [ispy2]=ispy2 [pelvic]=totalseg-pelvic [atlas]=atlas-liver-hcc [panseg]=pansegdata)
# pack | "setting:fold ..." (recording order; one fold per spec so the order is deterministic) | PACK_GPU_MAP | chain length
PACKS=(
"G1|brats_t1n:0 tf2_cbct:0 brats_t1n:1 tf2_cbct:1 brats_t1n:2 tf2_cbct:2 brats_t2w:0 ispy2_t1wce:0|0 0 1 1 2 2 3 3|3"
"G2|brats_t2w:1 ispy2_t1wce:1 brats_t2w:2 ispy2_t1wce:2 brats_t2f:0 ispy2_t2w:0 brats_t2f:1 ispy2_t2w:1|0 0 1 1 2 2 3 3|3"
"G3|brats_t2f:2 ispy2_t2w:2 brats_t1c:0 chaos_t1in:0 brats_t1c:1 chaos_t1in:1 brats_t1c:2 chaos_t1in:2|0 0 1 1 2 2 3 3|3"
"B1|onh_T1w:0 chaos_t2spir:0 onh_T1w:1 chaos_t2spir:1 onh_T1w:2 chaos_t2spir:2 openms_flair:0 openms_flair:1 openms_flair:2 panseg_t1wce:0 panseg_t1wce:1|0 0 1 1 2 2 3 3 3 3 3|3"
"B2|onh_T2w:0 pelvic_ct:0 onh_T2w:1 pelvic_ct:1 onh_T2w:2 pelvic_ct:2 openms_t1w:0 openms_t1w:1 openms_t1w:2 panseg_t1wce:2 panseg_t2w:0|0 0 1 1 2 2 3 3 3 3 3|3"
"B3|onh_dwi_ap:0 pelvic_mri:0 onh_dwi_ap:1 pelvic_mri:1 onh_dwi_ap:2 pelvic_mri:2 atlas_t1w:0 atlas_t1w:1 atlas_t1w:2 panseg_t2w:1 panseg_t2w:2|0 0 1 1 2 2 3 3 3 3 3|3"
)
[ -n "${PACK_LAYOUT_FILE:-}" ] && source "${PACK_LAYOUT_FILE}"
if [ -n "${PROBE:-}" ]; then PACKS=("P_probe|onh_T1w:0 brats_t1n:0 atlas_t1w:0 ispy2_t1wce:0|0 1 2 3|1"); fi
ALL_PACKS="$(for p in "${PACKS[@]}"; do echo -n "${p%%|*} "; done)"
SEL="${*:-${ALL_PACKS}}"
setting_row() { local s; for s in "${SETTINGS[@]}"; do [ "${s%%|*}" = "$1" ] && { echo "$s"; return 0; }; done; return 1; }
pack_row() { local p; for p in "${PACKS[@]}"; do [ "${p%%|*}" = "$1" ] && { echo "$p"; return 0; }; done; return 1; }

if [ -f "${STATE}" ]; then PACKS_ROOT="$(cat "${STATE}")"
else
    [ "${MODE}" = record ] || { echo "nothing recorded yet (${STATE})"; exit 1; }
    _ts="$(date +%Y%m%d_%H%M%S)"; PACKS_ROOT="${SCRATCH}/_packruns/${TAG}${PROBE:+_probe}_${_ts}"; echo "${PACKS_ROOT}" > "${STATE}"
    mkdir -p "${PACKS_ROOT}"; echo "${_ts}" > "${PACKS_ROOT}/TS"
fi
TS="$(cat "${PACKS_ROOT}/TS")"
echo "[${TAG}] PACKS_ROOT=${PACKS_ROOT} TS=${TS} repo=$(pwd) HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo '?')"

if [ "${MODE}" = record ]; then
  { echo "repo $(pwd) HEAD $(git rev-parse HEAD 2>/dev/null || echo '?') dirty=$(git status --porcelain 2>/dev/null | wc -l)"
    echo "AugLab HEAD $(git -C sub-workspaces/auglab_workspace/AugLab rev-parse HEAD) dirty=$(git -C sub-workspaces/auglab_workspace/AugLab status --porcelain | wc -l)"
    echo "config md5 $(md5sum < sub-workspaces/auglab_workspace/AugLab/auglab/configs/${CFG} | cut -c1-32) (expect ${CFG_MD5})"
    echo "val config md5 $(md5sum < sub-workspaces/auglab_workspace/AugLab/auglab/configs/${VCFG} | cut -c1-32) (expect ${VCFG_MD5})"; } >> "${PACKS_ROOT}/LAUNCH_INFO"
  for p in ${SEL}; do
    [ -f "${PACKS_ROOT}/${p}/index.tsv" ] && { echo "[record] ${p} already recorded -- skipped"; continue; }
    IFS='|' read -r _ spec gmap chain <<<"$(pack_row "$p")"
    for sf in ${spec}; do s="${sf%%:*}"; fold="${sf#*:}"
      IFS='|' read -r name dir tw tenv etr ep eda ecmp <<<"$(setting_row "$s")"
      wrapper="${dir}/04_train/${tw}"; [ -f "${wrapper}" ] || { echo "MISSING ${wrapper}"; exit 1; }
      envf="$(grep -o 'source "$(dirname "$0")/../00_utils/env[a-z0-9_]*\.sh"' "${wrapper}" | sed 's|.*/00_utils/||; s|"$||')"
      echo "[record] ${p} <- ${s} fold ${fold} ($(basename "${wrapper}"); env ${envf}, ${tenv})"
      (
        source "${dir}/00_utils/${envf}"; source "scripts/cluster/${tenv}"
        [ -z "${PROBE:-}" ] || { export PREDICTIONS_ROOT="${SCRATCH}/_flatauglab_probe/${s}/pred"; mkdir -p "${PREDICTIONS_ROOT}"; }
        export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"
        TRAIN_FOLDS="${fold}" RUN_JOB_PACK_DIR="${PACKS_ROOT}/${p}" bash "${wrapper}" "${DATASET_NAME}_${TRAINING_CONTRAST}_${METHOD}_${TS}" < /dev/null
      )
    done
    echo "${gmap}" > "${PACKS_ROOT}/${p}/GPU_MAP"; echo "${chain}" > "${PACKS_ROOT}/${p}/CHAIN"
  done
  if [ -n "${PROBE:-}" ]; then
    python3 - "${PACKS_ROOT}" "${PROBE_EPOCHS:-25}" <<'PY'
import re, sys, glob
n = 0
for f in glob.glob(sys.argv[1] + "/*/fold*.sh"):
    s = open(f).read(); t, k = re.subn(r"NNUNET_NUM_EPOCHS=(\\?'?)\d+", r"NNUNET_NUM_EPOCHS=\g<1>" + sys.argv[2], s)
    open(f, "w").write(t); n += k
print(f"[probe] epochs patched to {sys.argv[2]} in {n} places")
PY
  fi
  echo; echo "=== verification (every recorded fold) ==="; bad=0
  for p in ${SEL}; do
    IFS='|' read -r _ spec gmap chain <<<"$(pack_row "$p")"
    exp=$(echo ${spec} | wc -w); n=$(grep -c . "${PACKS_ROOT}/$p/index.tsv")
    echo "-- $p: ${n} folds (expect ${exp}); GPU map '${gmap}', chain ${chain}"
    [ "$n" = "$exp" ] || { echo "   FOLD COUNT MISMATCH"; bad=1; }
    [ "$(echo ${gmap} | wc -w)" = "$n" ] || { echo "   GPU MAP LENGTH != folds"; bad=1; }
    i=0; read -ra G <<<"${gmap}"; read -ra SP <<<"${spec}"
    while IFS=$'\t' read -r cmd log name done_; do
      clean="$(sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$cmd")"
      get() { printf '%s\n' "$clean" | sed -n "s/^ *export $1=//p" | head -1 | tr -d "'"; }
      pre=$(get nnUNet_preprocessed); res=$(get nnUNet_results); cfgp=$(get AUGLAB_PARAMS_GPU_JSON); cfg=${cfgp##*/}; ep=$(get NNUNET_NUM_EPOCHS)
      vcfg=$(get AUGLAB_VAL_PARAMS_GPU_JSON); da=$(get nnUNet_n_proc_DA); so=$(get NNUNET_SEED_OFFSET); cmp_=$(get nnUNet_compile)
      tr=$(printf '%s\n' "$clean" | grep -o '\-tr [A-Za-z0-9_]*' | head -1 | cut -d' ' -f2)
      fl=$(printf '%s\n' "$clean" | grep -oE 'nnUNetv2_train [0-9A-Za-z_]+ 3d_fullres [0-9]+' | head -1 | awk '{print $2" fold"$4}')
      es="${SP[$i]%%:*}"; ef="${SP[$i]#*:}"
      IFS='|' read -r _ edir _ _ etr eep eda ecmp <<<"$(setting_row "${es}")"
      ok=OK
      case "$pre" in /scratch/*) [ -d "$pre" ] || ok="NO_PREPROC_DIR";; *) ok="PREPROC_NOT_SCRATCH";; esac
      case "$res" in /scratch/*) ;; *) ok="${ok},RESULTS_NOT_SCRATCH";; esac
      [[ "$res" == */auglab/* ]] || ok="${ok},NOT_AUGLAB_CATEGORY"
      [[ "$res" == *_${METHOD}_${TS} ]] || ok="${ok},RUNID($res)"
      etc="${es#*_}"; eds="${DSNAME[${es%%_*}]}"   # contrast + dataset typed from the SPEC: catches a contrast swap within a dataset
      [[ "$res" == */${etc}/auglab/${eds}_${etc}_${METHOD}_${TS} ]] || ok="${ok},CONTRAST_OR_DATASET(${eds}_${etc})"
      [[ "$name" == fold${ef}_* ]] || ok="${ok},FOLD(${name})"
      [[ "$fl" == *" fold${ef}" ]] || ok="${ok},TRAIN_CMD_FOLD(${fl})"
      [ -z "${PROBE:-}" ] || [[ "$res" == *_flatauglab_probe* ]] || ok="${ok},PROBE_NOT_THROWAWAY"
      [ "$cfg" = "${CFG}" ] || ok="${ok},WRONG_CFG($cfg)"
      [ -f "$cfgp" ] && [ "$(md5sum < "$cfgp" | cut -c1-32)" = "${CFG_MD5}" ] || ok="${ok},CFG_FILE_MISSING_OR_MD5($cfgp)"
      if [[ "$etr" == *DualVal ]]; then   # DualVal: the flat synthetic-validation config (val100 mirror only)
        [ "${vcfg##*/}" = "${VCFG}" ] && [ -f "$vcfg" ] && [ "$(md5sum < "$vcfg" | cut -c1-32)" = "${VCFG_MD5}" ] || ok="${ok},VAL_CFG(${vcfg##*/})"
      else [ -z "$vcfg" ] || ok="${ok},VAL_CFG_SET($vcfg)"; fi
      [ "$tr" = "$etr" ] || ok="${ok},TRAINER($tr)"
      [ -n "${PROBE:-}" ] || [ "$ep" = "$eep" ] || ok="${ok},EPOCHS($ep)"
      [ "${da}" = "${eda}" ] && [ "${cmp_}" = "${ecmp}" ] || ok="${ok},LOADER(${da}/${cmp_})_NOT_${eda}/${ecmp}"
      grep -q "nnUNetv2_train" "$cmd" || ok="${ok},NO_TRAIN_CMD"
      [ "${so:-0}" = 0 ] || ok="${ok},SEED_OFFSET(${so})"
      [ -e "$res" ] && [ -z "${ALLOW_EXISTING:-}" ] && ok="${ok},RESULTS_DIR_ALREADY_EXISTS"
      printf '   %-6s GPU%s %-12s %s\n      trainer=%s ep=%s DA=%s compile=%s seed_offset=%s cfg=%s vcfg=%s ds/fold=%s\n      results=%s\n' \
        "$ok" "${G[$i]}" "$es" "$name" "$tr" "$ep" "$da" "${cmp_}" "${so:-0}" "$cfg" "${vcfg##*/}" "$fl" "$res"
      [ "$ok" = OK ] || bad=1; i=$((i+1))
    done < "${PACKS_ROOT}/$p/index.tsv"
  done
  echo; echo "=== loader lines printed by train_common.sh at record time ==="; grep -h '^\[loader\]' "${PACKS_ROOT}"/*/fold*.sh 2>/dev/null | sort | uniq -c || true
  [ "$bad" = 0 ] && echo "VERIFY PASSED" || echo "VERIFY FAILED"; exit 0
fi

if [ "${MODE}" = submit ]; then
  for p in ${SEL}; do
    gmap="$(cat "${PACKS_ROOT}/${p}/GPU_MAP")"; chain="$(cat "${PACKS_ROOT}/${p}/CHAIN")"
    if [ -n "${PROBE:-}" ]; then _time="01:30:00"; _name="${TAG}probe_${p}"; else _time="${PACK_TIME:-23:59:00}"; _name="${TAG}_${p}"; fi
    PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${_time}" PACK_CHAIN="${chain}" PACK_GPU_MAP="${gmap}" \
    PACK_USE_MPS=0 PACK_JOB_NAME="${_name}" bash scripts/job_runner/run_job_pack_submit.sh "${PACKS_ROOT}/${p}" | tee "${PACKS_ROOT}/${p}/SUBMIT.log"
    grep -oE 'Submitted batch job [0-9]+' "${PACKS_ROOT}/${p}/SUBMIT.log" | awk '{print $4}' > "${PACKS_ROOT}/${p}/CHAIN_JOBS"
    echo "[submit] ${p}: chain jobs $(tr '\n' ' ' < "${PACKS_ROOT}/${p}/CHAIN_JOBS")"
    sleep 5
  done
  exit 0
fi

if [ "${MODE}" = probe-report ] || [ "${MODE}" = status ]; then
  for idx in "${PACKS_ROOT}"/*/index.tsv; do
    i=0; read -ra G <<<"$(cat "$(dirname "$idx")/GPU_MAP")"; echo "-- $(basename "$(dirname "$idx")")"
    while IFS=$'\t' read -r cmd log name done_; do
      f="${name#fold}"; fold="${f%%_*}"; clean="$(sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$cmd")"
      res="$(printf '%s\n' "$clean" | sed -n 's/^ *export nnUNet_results=//p' | head -1)"
      L="$(ls -t "${res}"/Dataset*/*/fold_${fold}/training_log_*.txt 2>/dev/null | head -1 || true)"
      ts=$(grep -o "Epoch time: [0-9.]* s" "$L" 2>/dev/null | awk '{print $3}' | tail -n +3 | sort -n || true)
      med=$(echo "$ts" | awk 'NF{a[++n]=$1} END{if(n) print a[int((n+1)/2)]; else print "NA"}')
      last=$(grep -oE "^.*Epoch [0-9]+" "$L" 2>/dev/null | tail -1 | grep -oE '[0-9]+$' || true)
      vram=$(grep -oE "max_memory_allocated[^0-9]*[0-9.]+" "$L" 2>/dev/null | tail -1 || true)
      printf '%-80s GPU%s epoch=%-5s med_s/ep=%-6s final=%s %s\n' "${name#fold}" "${G[$i]}" "${last:-NA}" "$med" "$([ -f "${done_}" ] && echo yes || echo no)" "${vram}"
      i=$((i+1))
    done < "$idx"
  done
  exit 0
fi

if [ "${MODE}" = chaos-fov-runs ]; then   # RUNS_FILE for the chaos -> amos / sliver07 crop-before-predict evaluation (06_36 + shared driver)
  out="${PACKS_ROOT}/chaos_fov_crop_flatauglab_runs.txt"
  { echo "# flatauglab arm (${METHOD}), chaos-trained, for fov_crop_predict_evaluate.sh via 06_36_fov_crop_ladder_rungs.sh (RUNS_FILE=this file)."
    echo "# Format: contrast|category|run_id|trainer|metrics_subdir (empty = flat, like the headline rows the combined config reads)."
    for tc in t1in t2spir; do rid="$(ls "${PACKS_ROOT}"/*/fold[0-9]_chaos_${tc}_${METHOD}_[0-9]*.sh 2>/dev/null | xargs -r -n1 basename | sed 's/^fold[0-9]_//; s/\.sh$//' | sort -u)"
      [ "$(echo "${rid}" | grep -c .)" = 1 ] || { echo "cannot resolve chaos ${tc} run id" >&2; exit 1; }
      echo "${tc}|auglab|${rid}|nnUNetTrainerCHAOSAugLabV26_6_2|"; done; } > "${out}"
  cat "${out}"; exit 0
fi

# ---- companions whose prepared test data is staged on TamIA (2026-10-18), each afterok behind its setting's OWN predict job (that job
# refuses without checkpoint_best + checkpoint_final for folds 0-2 and audits its outputs, so its success is the training-complete gate):
#   mandible : hanseg_sif (ct, mrt1) + pddca_sif (ct), S-I flipped inputs -- shared rung-6 predict / eval job bodies, FLATAUG_COMPANIONS=only
#   chaos_fov: amos (ct, mri) + sliver07 (ct) crop-before-predict -- 06_36 predict (whole node) + tamia_chaos_fov_eval_job.sh (CPU, audited)
#   hcc      : lld-mmri-hcc + liverhccseg -- atlas 05_35 with TEXLITE_ARM=flatauglab ATLAS_COMPANIONS_ONLY=1 (predict + eval + counts inline)
# The Vulcan-only companions (duke / ispy1 / acrin6698, totalsegmri-pancreas / msd-pancreas) go through companions_poller.sh on Vulcan.
#   bash scripts/cluster/flatauglab/launch.sh queue-companions [mandible|chaos_fov|hcc ...]     (DRY=1: print the sbatch lines only)
if [ "${MODE}" = queue-companions ]; then
  cpu_submit() { local out; if out="$(sbatch --parsable "$@" 2>/dev/null)"; then echo "${out}"; return 0; fi
                 echo "  (bare sbatch failed -- retrying with --partition=cpubase_bynode_b1)" >&2
                 sbatch --parsable --partition=cpubase_bynode_b1 "$@"; }
  sub() { if [ -n "${DRY:-}" ]; then echo "[dry] sbatch $*" >&2; echo "DRY$RANDOM"; else sbatch --parsable "$@"; fi; }
  csub() { if [ -n "${DRY:-}" ]; then echo "[dry] sbatch(cpu) $*" >&2; echo "DRY$RANDOM"; else cpu_submit "$@"; fi; }
  own_pj() { local f="${PACKS_ROOT}/post_$1/$2"; [ -s "$f" ] || { echo "no ${f} (queue-post first)" >&2; return 1; }; head -1 "$f"; }
  CO="${PACKS_ROOT}/post_companions"; mkdir -p "${CO}"
  for k in ${*:-mandible chaos_fov hcc}; do
    O="${CO}/${k}"; [ -f "${O}/QUEUED" ] && [ -z "${DRY:-}" ] && { echo "[companions] ${k}: already queued ($(cat "${O}/QUEUED")) -- skipped"; continue; }
    mkdir -p "${O}"
    case "$k" in
      mandible)
        dep="$(own_pj tf2_cbct PREDICT_JOB)"; EXP="ALL,ROSTER=scripts/cluster/flatauglab/runs.sh,OUT=${O},FLATAUG_COMPANIONS=only"
        PJ="$(sub --dependency=afterok:"${dep}" --job-name="${TAG}_predict_tf2_companions" --time=02:00:00 \
              --output="${O}/predict_job_%j.out" --export="${EXP},PACK=tf2_cbct" scripts/cluster/tamia_rung6_pv_predict_job.sh)"
        ids="${PJ}"
        for row in hanseg_sif pddca_sif; do
          EJ="$(csub --dependency=afterok:"${PJ}" --job-name="${TAG}_eval_${row}" --time=02:00:00 --cpus-per-task=4 --mem=32000M \
                --account=aip-jcohen --output="${O}/eval_${row}_%j.out" --export="${EXP},ROW=${row}" scripts/cluster/tamia_rung6_pv_eval_job.sh)"
          ids="${ids} ${EJ}"; sleep 2; done ;;
      chaos_fov)
        dep="$(own_pj chaos_t1in PREDICT_JOB):$(own_pj chaos_t2spir PREDICT_JOB)"
        bash scripts/cluster/flatauglab/launch.sh chaos-fov-runs >/dev/null; RF="${PACKS_ROOT}/chaos_fov_crop_flatauglab_runs.txt"
        [ "$(grep -c "_${METHOD}_${TS}|" "${RF}")" = 2 ] || { echo "[companions] ${RF}: expected 2 flatauglab rows"; exit 1; }
        PJ="$(sub --dependency=afterok:"${dep}" --job-name="${TAG}_chaos_fovcrop_predict" --output="${O}/predict_%j.out" \
              --export="ALL,RUNS_FILE=${RF}" benchmark/02_tasks/abdomen_healthy/chaos/5_scripts_chaos/06_evaluate/06_36_fov_crop_ladder_rungs.sh predict)"
        EJ="$(csub --dependency=afterok:"${PJ}" --job-name="${TAG}_chaos_fovcrop_eval" --output="${O}/eval_%j.out" \
              --export="ALL,RUNS_FILE=${RF}" scripts/cluster/flatauglab/tamia_chaos_fov_eval_job.sh)"
        ids="${PJ} ${EJ}" ;;
      hcc)
        dep="$(own_pj atlas_t1w JOB)"
        rid="$(ls "${PACKS_ROOT}"/*/fold[0-9]_atlas-liver-hcc_t1w_${METHOD}_[0-9]*.sh | xargs -r -n1 basename | sed 's/^fold[0-9]_//; s/\.sh$//' | sort -u)"
        [ "$(echo "${rid}" | grep -c .)" = 1 ] || { echo "[companions] cannot resolve the atlas run id"; exit 1; }
        PJ="$(sub --dependency=afterok:"${dep}" --job-name="${TAG}_hcc_companions" --output="${O}/hcc_%j.out" \
              --export="ALL,RID=${rid},TEXLITE_ARM=flatauglab,ATLAS_COMPANIONS_ONLY=1" \
              benchmark/02_tasks/liver_hcc/atlas-liver-hcc/5_scripts_atlas-liver-hcc/05_predict/05_35_tamia_texlite_predict_eval.sh)"
        ids="${PJ}" ;;
      *) echo "unknown companion group ${k}"; exit 2 ;;
    esac
    echo "[companions] ${k}: jobs ${ids} (after ${dep})"
    [ -n "${DRY:-}" ] || echo "${ids} after ${dep} $(date '+%F %T')" > "${O}/QUEUED"
  done
  exit 0
fi

if [ "${MODE}" = runids ]; then
  out="${1:?out tsv}"; tmp="$(mktemp -p "${SCRATCH}")"
  for idx in "${PACKS_ROOT}"/*/index.tsv; do p="$(basename "$(dirname "$idx")")"
    while IFS=$'\t' read -r cmd log name done_; do f="${name#fold}"; printf '%s\t%s\t%s\n' "${f%%_*}" "${f#*_}" "$p" >> "${tmp}"; done < "$idx"
  done
  { printf 'fold\trun_id\tpack\n'; sort -u "${tmp}"; } > "${out}"; rm -f "${tmp}"; cat "${out}"; exit 0
fi

# runtable <out.tsv> (pa_em 2026-10-22, kept in sem50 and here): setting, fold, run_id, pack, gpu, chain jobs, predict job, eval
# jobs -- the RUN TABLE the Vulcan companion / metrics scripts read (copy it to /scratch/paulh/flatauglab_20261028/run_ids.tsv on Vulcan).
if [ "${MODE}" = runtable ]; then
  out="${1:?out tsv}"
  { printf 'setting\tfold\trun_id\tpack\tgpu\tchain_jobs\tpredict_job\teval_job\n'
    for idx in "${PACKS_ROOT}"/*/index.tsv; do pd="$(dirname "$idx")"; pk="$(basename "$pd")"; read -ra G < "$pd/GPU_MAP"
      cj="$(tr '\n' ',' < "$pd/CHAIN_JOBS" 2>/dev/null | sed 's/,$//')"; i=0
      while IFS=$'\t' read -r cmd log name done_; do f="${name#fold}"; k="${f%%_*}"; rid="${f#*_}"
        case "$rid" in brats2024-glioma_*) s="brats_${rid#brats2024-glioma_}";; on-harmony_*) s="onh_${rid#on-harmony_}";; open-ms_*) s="openms_${rid#open-ms_}";;
          toothfairy2_*) s="tf2_${rid#toothfairy2_}";; totalseg-pelvic_*) s="pelvic_${rid#totalseg-pelvic_}";; atlas-liver-hcc_*) s="atlas_${rid#atlas-liver-hcc_}";;
          pansegdata_*) s="panseg_${rid#pansegdata_}";; *) s="$rid";; esac
        s="${s%%_${METHOD}_*}"
        pj="$(cat "${PACKS_ROOT}/post_$s/PREDICT_JOB" 2>/dev/null || cat "${PACKS_ROOT}/post_$s/JOB" 2>/dev/null || true)"
        ej="$(tr '\n' ',' < "${PACKS_ROOT}/post_$s/EVAL_JOBS" 2>/dev/null | sed 's/,$//' || true)"; [ "$s" = atlas_t1w ] && ej="(inline in ${pj})"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$s" "$k" "$rid" "$pk" "${G[$i]}" "$cj" "$pj" "$ej"; i=$((i+1))
      done < "$idx"; done; } > "${out}"
  cat "${out}"; exit 0
fi

# ---- post-training (TamIA): per setting group one whole-node predict job (shared tamia_rung6_pv_predict_job.sh, ROSTER=runs.sh) behind
# the LAST chain job of every pack holding one of its folds + one CPU eval job per row (tamia_rung6_pv_eval_job.sh). The predict job refuses
# unless every fold has checkpoint_best + checkpoint_final. Companions: queue-companions (TamIA) / companions_poller.sh (Vulcan).
# pa_em changes vs ssem50 (2026-10-22), kept in sem50 and here: (1) the atlas own test set (05_35 TEXLITE_ARM=flatauglab ATLAS_OWN_ONLY=1, whole node, predict + eval
# inline; ssem50 submitted it by hand) is queued here too, group atlas_t1w, job id in post_atlas_t1w/JOB (read by queue-companions hcc);
# (2) the SHORT settings (chaos, ispy2, toothfairy2, pelvis: 1-5 h trainings stacked on long-fold GPUs, done in the first chain segment in
# ssem50) depend on the FIRST chain job of their packs instead of the last (ssem50 released them by hand, scontrol Dependency=); the
# predict job still refuses unless all 3 folds have checkpoint_best + checkpoint_final; (3) chaos eval jobs get 64000M (32000M OOMed in ssem50).
if [ "${MODE}" = queue-post ]; then
  cpu_submit() { local out; if out="$(sbatch --parsable "$@" 2>/dev/null)"; then echo "${out}"; return 0; fi
                 echo "  (bare sbatch failed -- retrying with --partition=cpubase_bynode_b1)" >&2
                 sbatch --parsable --partition=cpubase_bynode_b1 "$@"; }
  source scripts/cluster/flatauglab/runs.sh
  groups="$(for r in "${RUNG_ROWS[@]}"; do IFS='|' read -r _ g _ <<<"$r"; echo "$g"; done | awk '!s[$0]++')"
  groups="${groups} atlas_t1w"
  [ $# -gt 0 ] && groups="$*"   # queue-post <group...>: only these setting groups (e.g. the ones whose packs are submitted)
  first_seg() { case "$1" in chaos_*|ispy2_*|tf2_cbct|pelvic_*) return 0;; *) return 1;; esac; }
  # eval job size per row (small by default so the jobs do not sit in the CPU queue; the folds are evaluated sequentially):
  eval_res() { case "$1" in brats_*|chaos_*) echo "8 64000M 03:00:00";; onh_*|ispy2_*|tf2_cbct|hanseg_sif|pddca_sif) echo "4 32000M 02:00:00";;
               *) echo "${EVAL_CPUS:-4} ${EVAL_MEM:-32000M} ${EVAL_TIME:-01:00:00}";; esac; }
  for g in ${groups}; do
    if [ "$g" = atlas_t1w ]; then prefixes="atlas-liver-hcc_t1w_${METHOD}"
    else prefixes="$(for r in "${RUNG_ROWS[@]}"; do IFS='|' read -r _ grp _ _ _ _ _ _ prefix _ <<<"$r"; [ "$grp" = "$g" ] && echo "$prefix"; true; done | sort -u)"; fi
    deps=""; for idx in "${PACKS_ROOT}"/*/index.tsv; do pd="$(dirname "$idx")"; hit=0
      for pf in ${prefixes}; do grep -q "fold[0-9]_${pf}_[0-9]" "$idx" && hit=1; true; done
      [ "$hit" = 1 ] || continue
      [ -f "${pd}/CHAIN_JOBS" ] || { echo "no CHAIN_JOBS in ${pd} (submit first)"; exit 1; }
      if first_seg "$g"; then dj="$(head -1 "${pd}/CHAIN_JOBS")"; else dj="$(tail -1 "${pd}/CHAIN_JOBS")"; fi
      deps="${deps:+${deps}:}${dj}"; done
    [ -n "${deps}" ] || { echo "[queue-post] ${g}: no pack holds its folds -- skipped"; continue; }
    if [ "$g" = atlas_t1w ]; then   # HCC own test set: 05_35 (whole node; checkpoint gate + record + predict + eval + counts inline)
      OUT="${PACKS_ROOT}/post_${g}"; mkdir -p "${OUT}"; [ -s "${OUT}/JOB" ] && { echo "[queue-post] ${g}: already queued ($(cat "${OUT}/JOB"))"; continue; }
      rid="$(rung_resolve_run atlas_t1w "atlas-liver-hcc_t1w_${METHOD}")" || { echo "[queue-post] atlas run id UNRESOLVED"; exit 1; }
      AJ="$(sbatch --parsable --dependency=afterany:"${deps}" --job-name="${TAG}_atlas_own" --output="${OUT}/predict_eval_%j.out" \
            --export="ALL,RID=${rid},TEXLITE_ARM=flatauglab,ATLAS_OWN_ONLY=1" ${A}/05_predict/05_35_tamia_texlite_predict_eval.sh)"
      echo "${AJ}" > "${OUT}/JOB"; echo "[queue-post] ${g}: 05_35 own job ${AJ} (run ${rid}, after ${deps})"; continue
    fi
    OUT="${PACKS_ROOT}/post_${g}"; mkdir -p "${OUT}"
    ( for r in "${RUNG_ROWS[@]}"; do IFS='|' read -r name grp dir _ _ _ w _ prefix _ <<<"$r"; [ "$grp" = "$g" ] || continue
        run="$(rung_resolve_run "$grp" "$prefix")" || { echo "UNRESOLVED ${name}"; exit 1; }
        [ -f "$T/${dir}/05_predict/${w}" ] || { echo "NO PREDICT WRAPPER ${dir}/05_predict/${w}"; exit 1; }
        echo "  [post] ${name}: ${run}"; done ) || { echo "[queue-post] ${g}: run table check FAILED -- not queued"; exit 1; }
    EXP="ALL,ROSTER=scripts/cluster/flatauglab/runs.sh,OUT=${OUT},FLATAUG_COMPANIONS=${FLATAUG_COMPANIONS:-0}"
    PJ="$(sbatch --parsable --dependency=afterany:"${deps}" --job-name="${TAG}_predict_${g}" --time="${PREDICT_TIME:-03:00:00}" \
          --output="${OUT}/predict_job_%j.out" --export="${EXP},PACK=${g}" scripts/cluster/tamia_rung6_pv_predict_job.sh)"
    echo "${PJ}" > "${OUT}/PREDICT_JOB"; echo "[queue-post] ${g}: predict job ${PJ} (after ${deps})"; : > "${OUT}/EVAL_JOBS"
    for r in "${RUNG_ROWS[@]}"; do IFS='|' read -r name grp _ <<<"$r"; [ "$grp" = "$g" ] || continue
      read -r ec em et <<<"$(eval_res "${name}")"
      EJ="$(cpu_submit --dependency=afterany:"${PJ}" --job-name="${TAG}_eval_${name}" --time="${et}" --cpus-per-task="${ec}" \
            --mem="${em}" --account=aip-jcohen --output="${OUT}/eval_${name}_%j.out" \
            --export="${EXP},ROW=${name}" scripts/cluster/tamia_rung6_pv_eval_job.sh)"
      echo "${EJ}" >> "${OUT}/EVAL_JOBS"; echo "[queue-post]   eval ${name}: job ${EJ}"; sleep 2
    done
  done
  exit 0
fi
echo "unknown mode ${MODE}"; exit 2
