#!/usr/bin/env bash
# TamIA launcher of the "SynthSeg-EM (+Auglab; 50%)" competitor arm (ssem50, 2026-10-18): METHOD auglabAug_synthseg_EM_train050_val000,
# 19 training settings x folds 0-2 on TamIA (pansegdata moved from the Vulcan plan to TamIA 2026-10-18 to fill the B-node GPU3 tail). Every fold is RECORDED
# by its setting's own generated train wrapper (scripts/cluster/ssem50/make_wrappers.py; plain AugLabDefault trainer, policy epochs,
# dataset loader defaults), one RUN_ID per setting for the whole arm (<DATASET_NAME>_<TRAINING_CONTRAST>_<METHOD>_<TS>, TS fixed at
# the first record of the root), so the same setting's folds in different packs are ONE run. Clone of pa_variants/launch.sh.
#   bash scripts/cluster/ssem50/launch.sh record [PACK...]       # record + verify, no submit
#   bash scripts/cluster/ssem50/launch.sh submit [PACK...]       # submit the chains
#   bash scripts/cluster/ssem50/launch.sh queue-post [GROUP...]  # per-setting predict (whole node) + eval (CPU) behind the chains
#   bash scripts/cluster/ssem50/launch.sh status|probe-report    # epochs reached + median s/epoch per recorded fold
#   bash scripts/cluster/ssem50/launch.sh runids <out.tsv>       # setting, fold, run_id, pack
#   bash scripts/cluster/ssem50/launch.sh chaos-fov-runs         # write <root>/chaos_fov_crop_ssem50_runs.txt (amos / sliver07 fov_crop)
#   bash scripts/cluster/ssem50/launch.sh queue-companions [GROUP...]  # TamIA-staged companions (mandible / chaos_fov / hcc) behind the own predict jobs
# PROBE=1 (record / submit / probe-report): pack P_probe = fold 0 of Brain T1w, Glioma T1n, HCC, Breast t1wce, one per GPU,
#   PROBE_EPOCHS (default 25) epochs, throwaway results base ${SCRATCH}/_ssem50_probe, own state file, 1 h 30 job.
# SSEM50_STATE (optional): state-file override (e.g. a dry-run root recorded from a staging tree).
# Layout (PACKS below, draft 2026-10-18 from the latency probe; finalize after the sizing probe): 21 long folds (Glioma 12 x ~33 h,
#   Brain 9 x ~39 h at 1/GPU, H100) one per GPU on 6 nodes; each long GPU also carries ONE short fold (toothfairy2, ispy2, chaos,
#   pelvis: 1-5 h) that ends early; the three GPUs without a long fold carry open-ms flair x3, open-ms t1w x3, HCC x3 (~9 h each,
#   3 per GPU) + two pansegdata folds each (2026-10-18). Chain 3 x 23:59 per node (a chained job exits at once when every done-marker exists).
set -euo pipefail
cd "${SSEM50_REPO:-/project/aip-jcohen/paulh/mri_synthesis_project}"   # SSEM50_REPO: a code-only staging tree for a dry-run record
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"
TAG=ssem50
METHOD=auglabAug_synthseg_EM_train050_val000
CFG=transform_params_gpu_default01-23_auglabAug_Synthseg_EM_train050.json
CFG_MD5=befe7706707a64a8b9abf4ee82f5e67a
STATE="${SSEM50_STATE:-${SCRATCH}/_packruns_${TAG}_root${PROBE:+_probe}.txt}"
MODE="${1:?usage: record|submit|queue-post|queue-companions|status|probe-report|runids|chaos-fov-runs [PACK...]}"; shift
T=benchmark/02_tasks
# setting | scripts dir | train wrapper (04_train/) | tamia env | expected trainer | epochs | DA workers | compile   (expected values typed
# here independently of the wrappers: trainer/epochs = the source synthseg_EM run's debug.json; DA/compile = the dataset's 04_00_common.sh)
B=$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma; O=$T/brain_healthy/on-harmony/5_scripts_on-harmony
M=$T/brain_ms/open-ms/5_scripts_open-ms; C=$T/abdomen_healthy/chaos/5_scripts_chaos; TF=$T/mandible_healthy/toothfairy2/5_scripts_toothfairy2
I=$T/breast_cancer/ispy2/5_scripts_ispy2; P=$T/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic; A=$T/liver_hcc/atlas-liver-hcc/5_scripts_atlas-liver-hcc
PS=$T/pancreas_disease/pansegdata/5_scripts_pansegdata
SETTINGS=(
"brats_t1n|$B|04_62_train_t1n_${METHOD}.sh|tamia_env.sh|nnUNetTrainerBraTS2024GliomaAugLabDefault|2500|0|0"
"brats_t2w|$B|04_63_train_t2w_${METHOD}.sh|tamia_env.sh|nnUNetTrainerBraTS2024GliomaT2wAugLabDefault|2500|0|0"
"brats_t2f|$B|04_64_train_t2f_${METHOD}.sh|tamia_env.sh|nnUNetTrainerBraTS2024GliomaT2fAugLabDefault|2500|0|0"
"brats_t1c|$B|04_65_train_t1c_${METHOD}.sh|tamia_env.sh|nnUNetTrainerBraTS2024GliomaT1cAugLabDefault|2500|0|0"
"chaos_t1in|$C|04_83_train_t1in_${METHOD}.sh|tamia_env_chaos.sh|nnUNetTrainerCHAOSAugLabDefault|200|0|0"
"chaos_t2spir|$C|04_84_train_t2spir_${METHOD}.sh|tamia_env_chaos.sh|nnUNetTrainerCHAOSAugLabDefault|200|0|0"
"onh_T1w|$O|04_79_train_t1w_${METHOD}.sh|tamia_env_onharmony.sh|nnUNetTrainerOnHarmonyAugLabDefault|2000|0|0"
"onh_T2w|$O|04_80_train_t2w_${METHOD}.sh|tamia_env_onharmony.sh|nnUNetTrainerOnHarmonyAugLabDefault|2000|0|0"
"onh_dwi_ap|$O|04_81_train_dwi_ap_${METHOD}.sh|tamia_env_onharmony.sh|nnUNetTrainerOnHarmonyAugLabDefault|2000|0|0"
"openms_flair|$M|04_63_train_flair_${METHOD}.sh|tamia_env_openms.sh|nnUNetTrainerOpenMSAugLabDefault|2000|8|1"
"openms_t1w|$M|04_64_train_t1w_${METHOD}.sh|tamia_env_openms.sh|nnUNetTrainerOpenMSAugLabDefault|2000|8|1"
"tf2_cbct|$TF|04_23_train_cbct_${METHOD}.sh|tamia_env_toothfairy2.sh|nnUNetTrainerToothFairy2AugLabDefault|1000|8|1"
"ispy2_t1wce|$I|04_38_train_t1wce_${METHOD}.sh|tamia_env_ispy2.sh|nnUNetTrainerISPY2AugLabDefault|1000|8|1"
"ispy2_t2w|$I|04_39_train_t2w_${METHOD}.sh|tamia_env_ispy2.sh|nnUNetTrainerISPY2AugLabDefault|1000|8|1"
"pelvic_ct|$P|04_35_train_ct_${METHOD}.sh|tamia_env_totalseg-pelvic.sh|nnUNetTrainerTotalsegPelvicAugLabDefault|200|8|1"
"pelvic_mri|$P|04_36_train_mri_${METHOD}.sh|tamia_env_totalseg-pelvic.sh|nnUNetTrainerTotalsegPelvicAugLabDefault|200|8|1"
"atlas_t1w|$A|04_19_train_t1w_${METHOD}.sh|tamia_env_atlas-liver-hcc.sh|nnUNetTrainerAtlasLiverHCCAugLabDefault|2000|8|1"
"panseg_t1wce|$PS|04_35_train_t1wce_${METHOD}.sh|tamia_env_pansegdata.sh|nnUNetTrainerPANSEGDATAAugLabDefault|2000|8|1"
"panseg_t2w|$PS|04_36_train_t2w_${METHOD}.sh|tamia_env_pansegdata.sh|nnUNetTrainerPANSEGDATAAugLabDefault|2000|8|1"
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
    echo "config md5 $(md5sum < sub-workspaces/auglab_workspace/AugLab/auglab/configs/${CFG} | cut -c1-32) (expect ${CFG_MD5})"; } >> "${PACKS_ROOT}/LAUNCH_INFO"
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
        [ -z "${PROBE:-}" ] || { export PREDICTIONS_ROOT="${SCRATCH}/_ssem50_probe/${s}/pred"; mkdir -p "${PREDICTIONS_ROOT}"; }
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
      [ -z "${PROBE:-}" ] || [[ "$res" == *_ssem50_probe* ]] || ok="${ok},PROBE_NOT_THROWAWAY"
      [ "$cfg" = "${CFG}" ] || ok="${ok},WRONG_CFG($cfg)"
      [ -f "$cfgp" ] && [ "$(md5sum < "$cfgp" | cut -c1-32)" = "${CFG_MD5}" ] || ok="${ok},CFG_FILE_MISSING_OR_MD5($cfgp)"
      [ -z "$vcfg" ] || ok="${ok},VAL_CFG_SET($vcfg)"
      [ "$tr" = "$etr" ] || ok="${ok},TRAINER($tr)"
      [ -n "${PROBE:-}" ] || [ "$ep" = "$eep" ] || ok="${ok},EPOCHS($ep)"
      [ "${da}" = "${eda}" ] && [ "${cmp_}" = "${ecmp}" ] || ok="${ok},LOADER(${da}/${cmp_})_NOT_${eda}/${ecmp}"
      grep -q "nnUNetv2_train" "$cmd" || ok="${ok},NO_TRAIN_CMD"
      [ "${so:-0}" = 0 ] || ok="${ok},SEED_OFFSET(${so})"
      [ -e "$res" ] && [ -z "${ALLOW_EXISTING:-}" ] && ok="${ok},RESULTS_DIR_ALREADY_EXISTS"
      printf '   %-6s GPU%s %-12s %s\n      trainer=%s ep=%s DA=%s compile=%s seed_offset=%s cfg=%s ds/fold=%s\n      results=%s\n' \
        "$ok" "${G[$i]}" "$es" "$name" "$tr" "$ep" "$da" "${cmp_}" "${so:-0}" "$cfg" "$fl" "$res"
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
  out="${PACKS_ROOT}/chaos_fov_crop_ssem50_runs.txt"
  { echo "# ssem50 arm (${METHOD}), chaos-trained, for fov_crop_predict_evaluate.sh via 06_36_fov_crop_ladder_rungs.sh (RUNS_FILE=this file)."
    echo "# Format: contrast|category|run_id|trainer|metrics_subdir (empty = flat, like the headline rows the combined config reads)."
    for tc in t1in t2spir; do rid="$(ls "${PACKS_ROOT}"/*/fold[0-9]_chaos_${tc}_${METHOD}_[0-9]*.sh 2>/dev/null | xargs -r -n1 basename | sed 's/^fold[0-9]_//; s/\.sh$//' | sort -u)"
      [ "$(echo "${rid}" | grep -c .)" = 1 ] || { echo "cannot resolve chaos ${tc} run id" >&2; exit 1; }
      echo "${tc}|auglab|${rid}|nnUNetTrainerCHAOSAugLabDefault|"; done; } > "${out}"
  cat "${out}"; exit 0
fi

# ---- companions whose prepared test data is staged on TamIA (2026-10-18), each afterok behind its setting's OWN predict job (that job
# refuses without checkpoint_best + checkpoint_final for folds 0-2 and audits its outputs, so its success is the training-complete gate):
#   mandible : hanseg_sif (ct, mrt1) + pddca_sif (ct), S-I flipped inputs -- shared rung-6 predict / eval job bodies, SSEM50_COMPANIONS=only
#   chaos_fov: amos (ct, mri) + sliver07 (ct) crop-before-predict -- 06_36 predict (whole node) + tamia_chaos_fov_eval_job.sh (CPU, audited)
#   hcc      : lld-mmri-hcc + liverhccseg -- atlas 05_35 with TEXLITE_ARM=ssem50 ATLAS_COMPANIONS_ONLY=1 (predict + eval + counts inline)
# The Vulcan-only companions (duke / ispy1 / acrin6698, totalsegmri-pancreas / msd-pancreas) go through companions_poller.sh on Vulcan.
#   bash scripts/cluster/ssem50/launch.sh queue-companions [mandible|chaos_fov|hcc ...]     (DRY=1: print the sbatch lines only)
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
        dep="$(own_pj tf2_cbct PREDICT_JOB)"; EXP="ALL,ROSTER=scripts/cluster/ssem50/runs.sh,OUT=${O},SSEM50_COMPANIONS=only"
        PJ="$(sub --dependency=afterok:"${dep}" --job-name="${TAG}_predict_tf2_companions" --time=02:00:00 \
              --output="${O}/predict_job_%j.out" --export="${EXP},PACK=tf2_cbct" scripts/cluster/tamia_rung6_pv_predict_job.sh)"
        ids="${PJ}"
        for row in hanseg_sif pddca_sif; do
          EJ="$(csub --dependency=afterok:"${PJ}" --job-name="${TAG}_eval_${row}" --time=02:00:00 --cpus-per-task=4 --mem=32000M \
                --account=aip-jcohen --output="${O}/eval_${row}_%j.out" --export="${EXP},ROW=${row}" scripts/cluster/tamia_rung6_pv_eval_job.sh)"
          ids="${ids} ${EJ}"; sleep 2; done ;;
      chaos_fov)
        dep="$(own_pj chaos_t1in PREDICT_JOB):$(own_pj chaos_t2spir PREDICT_JOB)"
        bash scripts/cluster/ssem50/launch.sh chaos-fov-runs >/dev/null; RF="${PACKS_ROOT}/chaos_fov_crop_ssem50_runs.txt"
        [ "$(grep -c "_${METHOD}_${TS}|" "${RF}")" = 2 ] || { echo "[companions] ${RF}: expected 2 ssem50 rows"; exit 1; }
        PJ="$(sub --dependency=afterok:"${dep}" --job-name="${TAG}_chaos_fovcrop_predict" --output="${O}/predict_%j.out" \
              --export="ALL,RUNS_FILE=${RF}" benchmark/02_tasks/abdomen_healthy/chaos/5_scripts_chaos/06_evaluate/06_36_fov_crop_ladder_rungs.sh predict)"
        EJ="$(csub --dependency=afterok:"${PJ}" --job-name="${TAG}_chaos_fovcrop_eval" --output="${O}/eval_%j.out" \
              --export="ALL,RUNS_FILE=${RF}" scripts/cluster/ssem50/tamia_chaos_fov_eval_job.sh)"
        ids="${PJ} ${EJ}" ;;
      hcc)
        dep="$(own_pj atlas_t1w JOB)"
        rid="$(ls "${PACKS_ROOT}"/*/fold[0-9]_atlas-liver-hcc_t1w_${METHOD}_[0-9]*.sh | xargs -r -n1 basename | sed 's/^fold[0-9]_//; s/\.sh$//' | sort -u)"
        [ "$(echo "${rid}" | grep -c .)" = 1 ] || { echo "[companions] cannot resolve the atlas run id"; exit 1; }
        PJ="$(sub --dependency=afterok:"${dep}" --job-name="${TAG}_hcc_companions" --output="${O}/hcc_%j.out" \
              --export="ALL,RID=${rid},TEXLITE_ARM=ssem50,ATLAS_COMPANIONS_ONLY=1" \
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

# ---- post-training (TamIA): per setting group one whole-node predict job (shared tamia_rung6_pv_predict_job.sh, ROSTER=runs.sh) behind
# the LAST chain job of every pack holding one of its folds + one CPU eval job per row (tamia_rung6_pv_eval_job.sh). The predict job refuses
# unless every fold has checkpoint_best + checkpoint_final. Atlas (HCC) and the chaos fov_crop / breast / liver companions are NOT queued
# here: see VERIFICATION.md (atlas has no eval kind in the shared eval job; companions run after the checkpoints are fetched).
if [ "${MODE}" = queue-post ]; then
  cpu_submit() { local out; if out="$(sbatch --parsable "$@" 2>/dev/null)"; then echo "${out}"; return 0; fi
                 echo "  (bare sbatch failed -- retrying with --partition=cpubase_bynode_b1)" >&2
                 sbatch --parsable --partition=cpubase_bynode_b1 "$@"; }
  source scripts/cluster/ssem50/runs.sh
  groups="$(for r in "${RUNG_ROWS[@]}"; do IFS='|' read -r _ g _ <<<"$r"; echo "$g"; done | awk '!s[$0]++')"
  [ $# -gt 0 ] && groups="$*"   # queue-post <group...>: only these setting groups (e.g. the ones whose packs are submitted)
  # eval job size per row (small by default so the jobs do not sit in the CPU queue; the folds are evaluated sequentially):
  eval_res() { case "$1" in brats_*) echo "8 64000M 03:00:00";; onh_*|ispy2_*|tf2_cbct|hanseg_sif|pddca_sif) echo "4 32000M 02:00:00";;
               *) echo "${EVAL_CPUS:-4} ${EVAL_MEM:-32000M} ${EVAL_TIME:-01:00:00}";; esac; }
  for g in ${groups}; do
    prefixes="$(for r in "${RUNG_ROWS[@]}"; do IFS='|' read -r _ grp _ _ _ _ _ _ prefix _ <<<"$r"; [ "$grp" = "$g" ] && echo "$prefix"; true; done | sort -u)"
    deps=""; for idx in "${PACKS_ROOT}"/*/index.tsv; do pd="$(dirname "$idx")"; hit=0
      for pf in ${prefixes}; do grep -q "fold[0-9]_${pf}_[0-9]" "$idx" && hit=1; true; done
      [ "$hit" = 1 ] || continue
      [ -f "${pd}/CHAIN_JOBS" ] || { echo "no CHAIN_JOBS in ${pd} (submit first)"; exit 1; }
      deps="${deps:+${deps}:}$(tail -1 "${pd}/CHAIN_JOBS")"; done
    [ -n "${deps}" ] || { echo "[queue-post] ${g}: no pack holds its folds -- skipped"; continue; }
    OUT="${PACKS_ROOT}/post_${g}"; mkdir -p "${OUT}"
    ( for r in "${RUNG_ROWS[@]}"; do IFS='|' read -r name grp dir _ _ _ w _ prefix _ <<<"$r"; [ "$grp" = "$g" ] || continue
        run="$(rung_resolve_run "$grp" "$prefix")" || { echo "UNRESOLVED ${name}"; exit 1; }
        [ -f "$T/${dir}/05_predict/${w}" ] || { echo "NO PREDICT WRAPPER ${dir}/05_predict/${w}"; exit 1; }
        echo "  [post] ${name}: ${run}"; done ) || { echo "[queue-post] ${g}: run table check FAILED -- not queued"; exit 1; }
    EXP="ALL,ROSTER=scripts/cluster/ssem50/runs.sh,OUT=${OUT},SSEM50_COMPANIONS=${SSEM50_COMPANIONS:-0}"
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
