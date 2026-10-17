#!/usr/bin/env bash
# TamIA launcher of the PA variants round 2 (2026-10-16): 4 arms (ctrl = headline PA config unchanged, p075, edge2x, alphawide;
# scripts/cluster/pa_variants/make_wrappers.py) x the 3 DEVELOPMENT settings (on-harmony T1w, open-ms flair, chaos t2spir), packed
# onto whole-node 4xH100 chains. Every fold is recorded by its arm's own wrapper (plain val000 trainer, dataset loader defaults,
# policy epochs), one RUN_ID per (arm, setting) for the whole round (<dataset>_<contrast>_<METHOD>_<TS>), so a fold 1-2 extension
# recorded later into another pack of the same root is the SAME run. Clone of tamia_pack_texlite.sh (record / verify / submit /
# probe / post-training flow); predict + eval through the shared tamia_rung6_pv_{predict,eval}_job.sh with ROSTER=runs.sh.
#   bash scripts/cluster/pa_variants/launch.sh record [PACK...]       # record + verify, no submit
#   bash scripts/cluster/pa_variants/launch.sh submit [PACK...]       # submit the chains
#   bash scripts/cluster/pa_variants/launch.sh queue-post [PACK...]   # predict (whole node) + eval (CPU) behind the chains
#   bash scripts/cluster/pa_variants/launch.sh probe-report|status    # s/epoch + epochs reached per recorded fold
#   bash scripts/cluster/pa_variants/launch.sh runids <out.tsv>       # setting, fold, run_id, pack, arm
# PROBE=1 (record/submit/probe-report): the S0 layout for PROBE_EPOCHS (default 12) epochs into a throwaway results base
#   (${SCRATCH}/_pa_variants_probe), one 60-min job, own state file; read with probe-report, then cancel/forget.
# Packs (PACKS below; add a row for the fold 1-2 extension, then `record <pack>` + `submit <pack>` + `queue-post <pack>`):
#   S0 = fold-0 screen: GPU g = on-harmony + open-ms + chaos of arm g (on-harmony ~60 s/epoch shared -> ~34 h = chain 2 x 24 h;
#   open-ms ~10 h and chaos ~3 h finish inside chain job 1, so their post jobs wait on chain job 1 only).
# PA_ROUND=3 (2026-10-17): round 3 = Auglab weakening (redistoff, structoff, structoffredistoff + ctrl81 = the unchanged headline
#   config as round 3's own control; AugLab 38d9f01 configs), same protocol EXCEPT the loader: every round-3 run trains with 8
#   data-loader workers + torch compile (on-harmony / chaos through the wrappers' explicit override; verify checks 8/1). Own state file / pack root / TS (TAG pa_variants_r3), pack R3S0 with the S0 layout.
#   Every mode takes PA_ROUND=3 (record, submit, queue-post, status, runids); runs.sh resolves round-3 RUN_IDs from that root.
#   Design: /scratch/paulh/review_20261017/weaken/DESIGN.md (Vulcan).
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"
PA_ROUND="${PA_ROUND:-2}"; export PA_ROUND
TAG=pa_variants; [ "${PA_ROUND}" = 2 ] || TAG="pa_variants_r${PA_ROUND}"
STATE="${SCRATCH}/_packruns_${TAG}_root${PROBE:+_probe}.txt"
MODE="${1:?usage: record|submit|queue-post|probe-report|status|runids [PACK...]}"; shift
T=benchmark/02_tasks
O=$T/brain_healthy/on-harmony/5_scripts_on-harmony; M=$T/brain_ms/open-ms/5_scripts_open-ms; C=$T/abdomen_healthy/chaos/5_scripts_chaos
declare -A METH=([ctrl]=auglabAugCtrl_v26_6_2_train050_val000 [p075]=auglabAugP075_v26_6_2_train075_val000
                 [edge2x]=auglabAugEdge2x_v26_6_2_train050_val000 [alphawide]=auglabAugAlphaWide_v26_6_2_train050_val000
                 [unsharpoff]=auglabAugUnsharpOff_v26_6_2_train050_val000 [redistoff]=auglabAugRedistOff_v26_6_2_train050_val000
                 [edgehalf]=auglabAugEdgeHalf_v26_6_2_train050_val000 [texhalf]=auglabAugTexHalf_v26_6_2_train050_val000
                 [structoff]=auglabAugStructOff_v26_6_2_train050_val000 [structoffredistoff]=auglabAugStructOffRedistOff_v26_6_2_train050_val000
                 [ctrl81]=auglabAugCtrl81_v26_6_2_train050_val000)
declare -A CFG=([ctrl]=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json
                [p075]=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train075.json
                [edge2x]=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_edge2x.json
                [alphawide]=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_alphawide.json
                [unsharpoff]=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_unsharpoff.json
                [redistoff]=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_redistoff.json
                [edgehalf]=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_edgehalf.json
                [texhalf]=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_texhalf.json
                [structoff]=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_structoff.json
                [structoffredistoff]=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_structoff_redistoff.json
                [ctrl81]=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json)
# train wrapper numbers per setting, in arm order (round 2: ctrl p075 edge2x alphawide; round 3: redistoff structoff structoffredistoff ctrl81)
if [ "${PA_ROUND}" = 3 ]; then
declare -A NUM=([onh_t1w]="04_73 04_76 04_77 04_78" [openms_flair]="04_57 04_60 04_61 04_62" [chaos_t2spir]="04_77 04_80 04_81 04_82")
ARMS_ORDER=(redistoff structoff structoffredistoff ctrl81)
else
declare -A NUM=([onh_t1w]="04_68 04_69 04_70 04_71" [openms_flair]="04_52 04_53 04_54 04_55" [chaos_t2spir]="04_72 04_73 04_74 04_75")
ARMS_ORDER=(ctrl p075 edge2x alphawide)
fi
# setting | scripts dir | contrast tag in the wrapper name | tamia env | expected trainer | epochs
SETTINGS=(
"onh_t1w|$O|t1w|tamia_env_onharmony.sh|nnUNetTrainerOnHarmonyAugLabDefault|2000"
"openms_flair|$M|flair|tamia_env_openms.sh|nnUNetTrainerOpenMSAugLabDefault|2000"
"chaos_t2spir|$C|t2spir|tamia_env_chaos.sh|nnUNetTrainerCHAOSAugLabV26_6_2|200"
)
# pack | "arm:setting:folds ..." (recording order) | PACK_GPU_MAP (one GPU per recorded fold) | chain length
if [ "${PA_ROUND}" = 3 ]; then PACKS=(
"R3S0|redistoff:onh_t1w:0 redistoff:openms_flair:0 redistoff:chaos_t2spir:0 structoff:onh_t1w:0 structoff:openms_flair:0 structoff:chaos_t2spir:0 structoffredistoff:onh_t1w:0 structoffredistoff:openms_flair:0 structoffredistoff:chaos_t2spir:0 ctrl81:onh_t1w:0 ctrl81:openms_flair:0 ctrl81:chaos_t2spir:0|0 0 0 1 1 1 2 2 2 3 3 3|2"
); else PACKS=(
"S0|ctrl:onh_t1w:0 ctrl:openms_flair:0 ctrl:chaos_t2spir:0 p075:onh_t1w:0 p075:openms_flair:0 p075:chaos_t2spir:0 edge2x:onh_t1w:0 edge2x:openms_flair:0 edge2x:chaos_t2spir:0 alphawide:onh_t1w:0 alphawide:openms_flair:0 alphawide:chaos_t2spir:0|0 0 0 1 1 1 2 2 2 3 3 3|2"
); fi
[ -n "${PACK_LAYOUT_FILE:-}" ] && source "${PACK_LAYOUT_FILE}"
if [ -n "${PROBE:-}" ]; then PACKS=("P_probe|${PACKS[0]#*|}"); PACKS=("${PACKS[0]%|*}|1"); fi
ALL_PACKS="$(for p in "${PACKS[@]}"; do echo -n "${p%%|*} "; done)"
SEL="${*:-${ALL_PACKS}}"
setting_row() { local s; for s in "${SETTINGS[@]}"; do [ "${s%%|*}" = "$1" ] && { echo "$s"; return 0; }; done; return 1; }
pack_row() { local p; for p in "${PACKS[@]}"; do [ "${p%%|*}" = "$1" ] && { echo "$p"; return 0; }; done; return 1; }
arm_idx() { local i; for i in "${!ARMS_ORDER[@]}"; do [ "${ARMS_ORDER[$i]}" = "$1" ] && { echo "$i"; return 0; }; done; return 1; }
wrapper_of() {   # <arm> <setting> -> path of the arm's train wrapper
  local dir tc n; IFS='|' read -r _ dir tc _ _ _ <<<"$(setting_row "$2")"; read -ra n <<<"${NUM[$2]}"
  echo "${dir}/04_train/${n[$(arm_idx "$1")]}_train_${tc}_${METH[$1]}.sh"; }

if [ -f "${STATE}" ]; then PACKS_ROOT="$(cat "${STATE}")"
else
    [ "${MODE}" = record ] || { echo "nothing recorded yet (${STATE})"; exit 1; }
    _ts="$(date +%Y%m%d_%H%M%S)"; PACKS_ROOT="${SCRATCH}/_packruns/${TAG}${PROBE:+_probe}_${_ts}"; echo "${PACKS_ROOT}" > "${STATE}"
    mkdir -p "${PACKS_ROOT}"; echo "${_ts}" > "${PACKS_ROOT}/TS"
fi
TS="$(cat "${PACKS_ROOT}/TS")"
echo "[${TAG}] PACKS_ROOT=${PACKS_ROOT} TS=${TS}"

if [ "${MODE}" = record ]; then
  for p in ${SEL}; do
    [ -f "${PACKS_ROOT}/${p}/index.tsv" ] && { echo "[record] ${p} already recorded -- skipped"; continue; }
    IFS='|' read -r _ spec gmap chain <<<"$(pack_row "$p")"
    for asf in ${spec}; do arm="${asf%%:*}"; sf="${asf#*:}"; s="${sf%%:*}"; folds="${sf#*:}"
      IFS='|' read -r name dir tc tenv etr ep <<<"$(setting_row "$s")"
      wrapper="$(wrapper_of "${arm}" "${s}")"
      envf="$(grep -o 'source "$(dirname "$0")/../00_utils/env[a-z0-9_]*\.sh"' "${wrapper}" | sed 's|.*/00_utils/||; s|"$||')"
      echo "[record] ${p} <- ${arm} ${s} folds ${folds} ($(basename "${wrapper}"); env ${envf}, ${tenv})"
      (
        source "${dir}/00_utils/${envf}"; source "scripts/cluster/${tenv}"
        [ -z "${PROBE:-}" ] || { export PREDICTIONS_ROOT="${SCRATCH}/_pa_variants_probe/pred"; mkdir -p "${PREDICTIONS_ROOT}"; }
        export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"
        TRAIN_FOLDS="${folds//,/ }" RUN_JOB_PACK_DIR="${PACKS_ROOT}/${p}" bash "${wrapper}" "${DATASET_NAME}_${TRAINING_CONTRAST}_${METH[$arm]}_${TS}" < /dev/null
      )
    done
    echo "${gmap}" > "${PACKS_ROOT}/${p}/GPU_MAP"; echo "${chain}" > "${PACKS_ROOT}/${p}/CHAIN"
  done
  if [ -n "${PROBE:-}" ]; then
    python3 - "${PACKS_ROOT}" "${PROBE_EPOCHS:-12}" <<'PY'
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
    exp=0; for asf in ${spec}; do f="${asf##*:}"; exp=$(( exp + $(echo "$f" | tr ',' '\n' | grep -c .) )); done
    n=$(grep -c . "${PACKS_ROOT}/$p/index.tsv"); echo "-- $p: ${n} folds (expect ${exp}); GPU map '${gmap}', chain ${chain}"
    [ "$n" = "$exp" ] || { echo "   FOLD COUNT MISMATCH"; bad=1; }
    [ "$(echo ${gmap} | wc -w)" = "$n" ] || { echo "   GPU MAP LENGTH != folds"; bad=1; }
    i=0; read -ra G <<<"${gmap}"; declare -a exp_rows=()
    for asf in ${spec}; do for f in $(echo "${asf##*:}" | tr ',' ' '); do exp_rows+=("${asf%%:*}|${asf#*:}"); done; done
    while IFS=$'\t' read -r cmd log name done_; do
      clean="$(sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$cmd")"
      get() { printf '%s\n' "$clean" | sed -n "s/^ *export $1=//p" | head -1; }
      pre=$(get nnUNet_preprocessed); res=$(get nnUNet_results); cfgp=$(get AUGLAB_PARAMS_GPU_JSON); cfg=${cfgp##*/}; ep=$(get NNUNET_NUM_EPOCHS)
      vcfg=$(get AUGLAB_VAL_PARAMS_GPU_JSON); da=$(get nnUNet_n_proc_DA); so=$(get NNUNET_SEED_OFFSET); so="${so//\'/}"
      cmp_=$(get nnUNet_compile); cmp_="${cmp_//\'/}"
      tr=$(printf '%s\n' "$clean" | grep -o '\-tr [A-Za-z0-9_]*' | head -1 | cut -d' ' -f2)
      erow="${exp_rows[$i]}"; earm="${erow%%|*}"; es="${erow#*|}"; es="${es%%:*}"
      IFS='|' read -r _ _ _ _ etr eep <<<"$(setting_row "${es}")"
      ok=OK
      case "$pre" in /scratch/*) [ -d "$pre" ] || ok="NO_PREPROC_DIR";; *) ok="PREPROC_NOT_SCRATCH";; esac
      case "$res" in /scratch/*) ;; *) ok="${ok},RESULTS_NOT_SCRATCH";; esac
      [[ "$res" == */auglab/* ]] || ok="${ok},NOT_AUGLAB_CATEGORY"
      [[ "$res" == *_${METH[$earm]}_${TS} ]] || ok="${ok},RUNID_NOT_${earm}"
      [ -z "${PROBE:-}" ] || [[ "$res" == *_pa_variants_probe* ]] || ok="${ok},PROBE_NOT_THROWAWAY"
      [ "$cfg" = "${CFG[$earm]}" ] || ok="${ok},WRONG_CFG($cfg)"
      [ -f "$cfgp" ] || ok="${ok},CFG_FILE_MISSING($cfgp)"
      [ -z "$vcfg" ] || ok="${ok},VAL_CFG_SET($vcfg)"
      [ "$tr" = "$etr" ] || ok="${ok},TRAINER($tr)"
      [ -n "${PROBE:-}" ] || [ "$ep" = "$eep" ] || [ "$ep" = "'$eep'" ] || ok="${ok},EPOCHS($ep)"
      grep -q "nnUNetv2_train" "$cmd" || ok="${ok},NO_TRAIN_CMD"
      [ "${so:-0}" = 0 ] || ok="${ok},SEED_OFFSET(${so})"
      if [ "${PA_ROUND}" = 3 ]; then [ "${da}" = 8 ] && [ "${cmp_}" = 1 ] || ok="${ok},LOADER(${da}/${cmp_})_NOT_8/1"; fi
      [ -e "$res" ] && ok="${ok},RESULTS_DIR_ALREADY_EXISTS"
      printf '   %-6s GPU%s %s\n      trainer=%s ep=%s DA=%s compile=%s seed_offset=%s cfg=%s\n      results=%s\n' "$ok" "${G[$i]}" "$name" "$tr" "$ep" "$da" "${cmp_}" "${so:-0}" "$cfg" "$res"
      [ "$ok" = OK ] || bad=1; i=$((i+1))
    done < "${PACKS_ROOT}/$p/index.tsv"
  done
  echo; echo "=== loader lines printed by train_common.sh at record time ==="; grep -h '^\[loader\]' "${PACKS_ROOT}"/*/fold*.sh 2>/dev/null | sort | uniq -c || true
  [ "$bad" = 0 ] && echo "VERIFY PASSED" || echo "VERIFY FAILED"; exit 0
fi

if [ "${MODE}" = submit ]; then
  for p in ${SEL}; do
    gmap="$(cat "${PACKS_ROOT}/${p}/GPU_MAP")"; chain="$(cat "${PACKS_ROOT}/${p}/CHAIN")"
    if [ -n "${PROBE:-}" ]; then _time="01:00:00"; _name="${TAG}probe_${p}"; else _time="${PACK_TIME:-23:59:00}"; _name="${TAG}_${p}"; fi
    PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${_time}" PACK_CHAIN="${chain}" PACK_GPU_MAP="${gmap}" \
    PACK_USE_MPS=0 PACK_JOB_NAME="${_name}" bash scripts/job_runner/run_job_pack_submit.sh "${PACKS_ROOT}/${p}" | tee "${PACKS_ROOT}/${p}/SUBMIT.log"
    grep -oE 'Submitted batch job [0-9]+' "${PACKS_ROOT}/${p}/SUBMIT.log" | awk '{print $4}' > "${PACKS_ROOT}/${p}/CHAIN_JOBS"
    echo "[submit] ${p}: chain jobs $(tr '\n' ' ' < "${PACKS_ROOT}/${p}/CHAIN_JOBS")"
    sleep 5
  done
  exit 0
fi

if [ "${MODE}" = probe-report ] || [ "${MODE}" = status ]; then   # s/epoch (median, epochs >= 3) + epochs reached, per fold
  for idx in "${PACKS_ROOT}"/*/index.tsv; do
    i=0; read -ra G <<<"$(cat "$(dirname "$idx")/GPU_MAP")"; echo "-- $(basename "$(dirname "$idx")")"
    while IFS=$'\t' read -r cmd log name done_; do
      f="${name#fold}"; fold="${f%%_*}"; clean="$(sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$cmd")"
      res="$(printf '%s\n' "$clean" | sed -n 's/^ *export nnUNet_results=//p' | head -1)"
      L="$(ls -t "${res}"/Dataset*/*/fold_${fold}/training_log_*.txt 2>/dev/null | head -1 || true)"
      ts=$(grep -o "Epoch time: [0-9.]* s" "$L" 2>/dev/null | awk '{print $3}' | tail -n +3 | sort -n || true)
      n=$(echo "$ts" | grep -c . || true); med=$(echo "$ts" | awk '{a[NR]=$1} END{if(NR) print a[int((NR+1)/2)]; else print "NA"}')
      last=$(grep -oE "^.*Epoch [0-9]+" "$L" 2>/dev/null | tail -1 | grep -oE '[0-9]+$' || true)
      cfgj=$(ls "${res}"/Dataset*/*/fold_${fold}/transform_params_gpu_used_for_training.json 2>/dev/null | head -1 || true)
      fin=$([ -f "${done_}" ] && echo yes || echo no)
      printf '%-72s GPU%s epoch=%-5s med_s/ep=%-6s final=%s cfg_saved=%s\n' "${name#fold}" "${G[$i]}" "${last:-NA}" "$med" "$fin" "${cfgj:+yes}"
      i=$((i+1))
    done < "$idx"
  done
  exit 0
fi

if [ "${MODE}" = runids ]; then   # setting <TAB> fold <TAB> run_id <TAB> pack <TAB> arm
  out="${1:?out tsv}"; tmp="$(mktemp -p "${SCRATCH}")"
  for idx in "${PACKS_ROOT}"/*/index.tsv; do p="$(basename "$(dirname "$idx")")"
    while IFS=$'\t' read -r cmd log name done_; do f="${name#fold}"; rid="${f#*_}"; arm=""
      for a in "${ARMS_ORDER[@]}"; do [[ "${rid}" == *"_${METH[$a]}_"* ]] && arm="$a"; done
      case "${rid}" in on-harmony_T1w_*) s=onh_t1w ;; open-ms_flair_*) s=openms_flair ;; chaos_t2spir_*) s=chaos_t2spir ;; *) s=UNKNOWN ;; esac
      printf '%s\t%s\t%s\t%s\t%s\n' "$s" "${f%%_*}" "${rid}" "$p" "${arm}" >> "${tmp}"; done < "$idx"
  done
  { printf 'setting\tfold\trun_id\tpack\tarm\n'; sort -u "${tmp}"; } > "${out}"; rm -f "${tmp}"; cat "${out}"; exit 0
fi

# ---- post-training: per (pack, setting) one whole-node predict job (shared tamia_rung6_pv_predict_job.sh, ROSTER=runs.sh,
# RUNG_FOLDS = the pack's folds, PA_ARMS = the pack's arms) + one CPU eval job per row (tamia_rung6_pv_eval_job.sh).
# Dependency: on-harmony waits for the LAST chain job; open-ms / chaos (done inside chain job 1) wait for chain job 1 only.
# The predict job refuses to run unless every listed fold has checkpoint_best + checkpoint_final.
# Eval jobs: 4 CPU / 32000 MB / 1 h by default (EVAL_CPUS / EVAL_MEM / EVAL_TIME): bigger CPU requests sat at Priority for days on
# TamIA in round 2 (2026-10-16), and the fold-0 evaluations fit in this.
if [ "${MODE}" = queue-post ]; then
    cpu_submit() { local out; if out="$(sbatch --parsable "$@" 2>/dev/null)"; then echo "${out}"; return 0; fi
                   echo "  (bare sbatch failed -- retrying with --partition=cpubase_bynode_b1)" >&2
                   sbatch --parsable --partition=cpubase_bynode_b1 "$@"; }
    for p in ${SEL}; do
      IFS='|' read -r _ spec _ _ <<<"$(pack_row "$p")"
      CJ=(); [ -f "${PACKS_ROOT}/${p}/CHAIN_JOBS" ] && mapfile -t CJ < "${PACKS_ROOT}/${p}/CHAIN_JOBS"
      [ "${#CJ[@]}" -gt 0 ] || { echo "no CHAIN_JOBS for ${p} (submit first)"; exit 1; }
      for s in onh_t1w openms_flair chaos_t2spir; do
        arms=""; folds=""
        for asf in ${spec}; do sf="${asf#*:}"; [ "${sf%%:*}" = "$s" ] || continue
          arms="${arms} ${asf%%:*}"; folds="$(echo "${sf#*:}" | tr ',' ' ')"; done
        [ -n "${arms}" ] || continue
        arms="$(echo ${arms} | tr ' ' '\n' | sort -u | tr '\n' ' ' | sed 's/ $//')"
        case "$s" in onh_t1w) dep="${CJ[-1]}" ;; *) dep="${CJ[0]}" ;; esac
        OUT="${PACKS_ROOT}/post_${p}_${s}"; mkdir -p "${OUT}"
        # the run table must resolve every row before anything is queued
        ( export PA_ARMS="${arms}"; source scripts/cluster/pa_variants/runs.sh
          for r in "${RUNG_ROWS[@]}"; do IFS='|' read -r name pack dir _ _ _ w _ prefix _ <<<"$r"; [ "$pack" = "$s" ] || continue
            run="$(rung_resolve_run "$pack" "$prefix")" || { echo "UNRESOLVED ${name}"; exit 1; }
            [ -f "$T/${dir}/05_predict/${w}" ] || { echo "NO PREDICT WRAPPER ${dir}/05_predict/${w}"; exit 1; }
            echo "  [post] ${name}: ${run}"; done ) || { echo "[queue-post] ${p}/${s}: run table check FAILED -- not queued"; exit 1; }
        EXP="ALL,ROSTER=scripts/cluster/pa_variants/runs.sh,PA_ROUND=${PA_ROUND},PA_ARMS=${arms},RUNG_FOLDS=${folds},OUT=${OUT}"
        PJ="$(sbatch --parsable --dependency=afterany:"${dep}" --job-name="${TAG}_predict_${p}_${s}" --time=03:00:00 \
              --output="${OUT}/predict_job_%j.out" --export="${EXP},PACK=${s}" scripts/cluster/tamia_rung6_pv_predict_job.sh)"
        echo "${PJ}" > "${OUT}/PREDICT_JOB"; echo "[queue-post] ${p}/${s}: predict job ${PJ} (after ${dep}; arms '${arms}', folds '${folds}')"
        : > "${OUT}/EVAL_JOBS"
        for a in ${arms}; do
          EJ="$(cpu_submit --dependency=afterany:"${PJ}" --job-name="${TAG}_eval_${a}_${s}" --time="${EVAL_TIME:-01:00:00}" --cpus-per-task="${EVAL_CPUS:-4}" \
                --mem="${EVAL_MEM:-32000M}" --account=aip-jcohen --output="${OUT}/eval_${a}_%j.out" \
                --export="${EXP},ROW=${a}_${s}" scripts/cluster/tamia_rung6_pv_eval_job.sh)"
          echo "${EJ}" >> "${OUT}/EVAL_JOBS"; echo "[queue-post]   eval ${a}_${s}: job ${EJ}"; sleep 2
        done
      done
    done
    exit 0
fi
echo "unknown mode ${MODE}"; exit 2
