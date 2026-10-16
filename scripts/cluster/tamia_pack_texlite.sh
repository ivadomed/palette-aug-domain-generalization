#!/usr/bin/env bash
# TamIA launcher: PALETTE + REDUCED Auglab ("texlite", config
# transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_texlite.json) on a subset of training
# settings x folds 0 1 2, packed onto whole-node 4xH100 chains. Each fold is recorded by its dataset's
# *_auglabAugTexlite_v26_6_2_train050_val000.sh wrapper (scripts/cluster/auglab_texlite/make_wrappers.py: a copy of that
# setting's headline OURS wrapper with ONLY the train config swapped; plain val000 trainer where the headline was DualVal).
# Modelled on tamia_pack_rung7_auglab_pv.sh (same record / verify / submit / probe / post-training flow).
#   bash scripts/cluster/tamia_pack_texlite.sh record [PACK...]      # record + verify, no submit
#   bash scripts/cluster/tamia_pack_texlite.sh submit [PACK...]      # submit chains (resume-safe)
#   bash scripts/cluster/tamia_pack_texlite.sh postcheck|queue-post [PACK...]   # predict + eval behind the chains
# PROBE=1 (record/submit): ONE pack "P_probe" = fold 0 of every setting, PROBE_EPOCHS epochs (default 12), throwaway results
#   under ${SCRATCH}/_texlite_probe, one 60-min job; read it with `probe-report`, cancel the job once the numbers are in.
# Run ids: ${SCRATCH}/_packruns/texlite_<ts>/<pack>/index.tsv; table written by `runids` to <out tsv>.
# TEXLITE_ARM=keepres (any mode): second arm "texlite_keepres" (config ..._train050_texlite_keepres.json: GaussianBlur and
#   SimulateLowRes kept as in the headline config), METHOD auglabAugTexliteKeepres_v26_6_2_train050_val000, its own wrappers
#   (one number up), its own pack root/state file (_packruns_texlite_keepres_root.txt) and job names (texlite_keepres_*), so
#   it never touches the texlite launch. Unset = the texlite arm.
# TEXLITE_ARM=edgeonly: third arm (config ..._train050_edgeonly.json = headline with only Scharr + RandomConv off), METHOD
#   auglabAugEdgeonly_v26_6_2_train050_val000, wrappers two numbers after the texlite ones, root/jobs texlite_edgeonly_*.
# SCOPE (2026-10-15 evening): the operator set is chosen on the DEVELOPMENT settings only = on-harmony T1w, open-ms flair, chaos
#   t2spir. Glioma t1n, mandible cbct and HCC t1w are HELD OUT: their texlite pack A (620790-2) and keepres pack A (621069-71) and
#   post jobs were cancelled, keepres pack B (621072-3, never started) was replaced by pack D_dev. Packs per arm below.
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"
ARM="${TEXLITE_ARM:-}"; export TEXLITE_ARM="${ARM}"   # read by tamia_texlite_runs.sh (also inside the post jobs)
TAG="texlite${ARM:+_${ARM}}"                          # state file / pack root / job-name prefix
STATE="${SCRATCH}/_packruns_${TAG}_root${PROBE:+_probe}.txt"
MODE="${1:?usage: record|submit|postcheck|queue-post|probe-report|runids [PACK...]}"; shift
case "${ARM}" in
  "") NEWCFG=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_texlite.json
      NEWM=auglabAugTexlite_v26_6_2_train050_val000
      WB=04_34; WM=04_49; WA=04_17; WT=04_21; WC=04_69; WO=04_65; PA=05_12 ;;
  keepres) NEWCFG=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_texlite_keepres.json
      NEWM=auglabAugTexliteKeepres_v26_6_2_train050_val000
      WB=04_35; WM=04_50; WA=04_18; WT=04_22; WC=04_70; WO=04_66; PA=05_13 ;;
  edgeonly) NEWCFG=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_edgeonly.json
      NEWM=auglabAugEdgeonly_v26_6_2_train050_val000
      WB=none; WM=04_51; WA=none; WT=none; WC=04_71; WO=04_67; PA=none ;;   # development settings only
  *) echo "unknown TEXLITE_ARM='${ARM}' (empty, keepres or edgeonly)"; exit 2 ;;
esac
T=benchmark/02_tasks
if [ -d "$T/liver_hcc/atlas-liver-hcc" ]; then AD="$T/liver_hcc/atlas-liver-hcc/5_scripts_atlas-liver-hcc"
else AD="benchmark/03_archive/atlas-liver-hcc/5_scripts_atlas-liver-hcc"; fi   # TamIA checkout (pre-2026-10-14 layout)
B=$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma; C=$T/abdomen_healthy/chaos/5_scripts_chaos
M=$T/brain_ms/open-ms/5_scripts_open-ms; TF=$T/mandible_healthy/toothfairy2/5_scripts_toothfairy2
O=$T/brain_healthy/on-harmony/5_scripts_on-harmony

# setting | scripts dir | texlite wrapper | headline OURS wrapper | tamia env | expected trainer | epochs
SETTINGS=(
"brats_t1n|$B|${WB}_train_t1n_${NEWM}.sh|04_28_train_t1n_auglabAug_v26_6_2_train050_val000.sh|tamia_env.sh|nnUNetTrainerBraTS2024GliomaAugLabDefault|2500"
"openms_flair|$M|${WM}_train_flair_${NEWM}.sh|04_22_train_auglabAug_v26_6_2_train050_val000.sh|tamia_env_openms.sh|nnUNetTrainerOpenMSAugLabDefault|2000"
"atlas_t1w|$AD|${WA}_train_t1w_${NEWM}.sh|04_06_train_auglabAug_v26_6_2_dualval.sh|tamia_env_atlas-liver-hcc.sh|nnUNetTrainerAtlasLiverHCCAugLabDefault|2000"
"tf2_cbct|$TF|${WT}_train_cbct_${NEWM}.sh|04_06_train_auglabAug_v26_6_2_dualval.sh|tamia_env_toothfairy2.sh|nnUNetTrainerToothFairy2AugLabDefault|1000"
"chaos_t2spir|$C|${WC}_train_t2spir_${NEWM}.sh|04_44_train_t2spir_auglabAug_v26_6_2_train050_val000.sh|tamia_env_chaos.sh|nnUNetTrainerCHAOSAugLabV26_6_2|200"
"onh_t1w|$O|${WO}_train_t1w_${NEWM}.sh|04_26_train_t1w_auglabAug_v26_6_2_dualval.sh|tamia_env_onharmony.sh|nnUNetTrainerOnHarmonyAugLabDefault|2000"
)
# Packs: pack | "setting:folds setting:folds ..." | PACK_GPU_MAP (one GPU per recorded fold, in recording order) | chain length
# Sized from the sizing probe (PROBE=1 + probe-report); PACK_LAYOUT_FILE overrides this default.
# Probe 2026-10-15 (12 epochs, 1 fold/GPU except tf2+chaos sharing): brats 47.5 s/epoch at ~11% GPU utilisation (data-loading
# bound, DA_WORKERS=0 as in the headline run), open-ms 14.9, atlas 28.8, tf2 52.4 (shared), chaos 30.2 (shared); rung-7 brats
# PA-Aug at 3 folds/GPU ran 58.5 s/epoch -> pair folds per GPU instead of one fold per GPU.
# Development settings only (scope 2026-10-15 evening). texlite: B (running since 17:54); its on-harmony folds run on Vulcan.
# keepres / edgeonly: ONE 9-fold node each,
# on-harmony + open-ms paired on GPUs 0-2, chaos x3 on GPU 3 (the 2026-10-15 pack-A pattern).
case "${ARM}" in
  # texlite on-harmony T1w: pack C_onh (621108-9, one GPU idle for ~20 h) was cancelled before it started; those 3 folds train
  # per-GPU on VULCAN with the same RUN_ID (scripts/cluster/auglab_texlite/vulcan_onh_texlite.sh).
  "") PACKS=("B_openms_chaos|openms_flair:0,1,2 chaos_t2spir:0,1,2|0 1 2 3 3 3|2") ;;
  *)  PACKS=("D_dev|onh_t1w:0,1,2 openms_flair:0,1,2 chaos_t2spir:0,1,2|0 1 2 0 1 2 3 3 3|2") ;;
esac
[ -n "${PACK_LAYOUT_FILE:-}" ] && source "${PACK_LAYOUT_FILE}"
if [ -n "${PROBE:-}" ]; then PACKS=("P_probe|brats_t1n:0 openms_flair:0 atlas_t1w:0 tf2_cbct:0 chaos_t2spir:0|0 1 2 3 3|1"); fi
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
# One RUN_ID per setting for the whole launch (<dataset>_<contrast>_<METHOD>_<TS>, train_common.sh's format), passed to the
# wrapper explicitly, so a setting whose folds sit in two packs is still ONE run.
TS="$(cat "${PACKS_ROOT}/TS")"
echo "[${TAG}] PACKS_ROOT=${PACKS_ROOT} METHOD=${NEWM}"

if [ "${MODE}" = record ]; then
  for p in ${SEL}; do
    [ -f "${PACKS_ROOT}/${p}/index.tsv" ] && { echo "[record] ${p} already recorded -- skipped"; continue; }
    IFS='|' read -r _ spec gmap chain <<<"$(pack_row "$p")"
    for sf in ${spec}; do s="${sf%%:*}"; folds="${sf#*:}"
      IFS='|' read -r name dir w orig tenv etr ep <<<"$(setting_row "$s")"
      wrapper="${dir}/04_train/${w}"
      envf="$(grep -o 'source "$(dirname "$0")/../00_utils/env[a-z0-9_]*\.sh"' "${wrapper}" | sed 's|.*/00_utils/||; s|"$||')"
      echo "[record] ${p} <- ${s} folds ${folds} (${w}; env ${envf}, ${tenv})"
      (
        source "${dir}/00_utils/${envf}"; source "scripts/cluster/${tenv}"
        [ -z "${PROBE:-}" ] || { export PREDICTIONS_ROOT="${SCRATCH}/_texlite_probe/pred"; mkdir -p "${PREDICTIONS_ROOT}"; }
        export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"
        TRAIN_FOLDS="${folds//,/ }" RUN_JOB_PACK_DIR="${PACKS_ROOT}/${p}" bash "${wrapper}" "${DATASET_NAME}_${TRAINING_CONTRAST}_${NEWM}_${TS}" < /dev/null
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
    exp=0; for sf in ${spec}; do f="${sf#*:}"; exp=$(( exp + $(echo "$f" | tr ',' '\n' | grep -c .) )); done
    n=$(grep -c . "${PACKS_ROOT}/$p/index.tsv"); echo "-- $p: ${n} folds (expect ${exp}); GPU map '${gmap}', chain ${chain}"
    [ "$n" = "$exp" ] || { echo "   FOLD COUNT MISMATCH"; bad=1; }
    [ "$(echo ${gmap} | wc -w)" = "$n" ] || { echo "   GPU MAP LENGTH != folds"; bad=1; }
    while IFS=$'\t' read -r cmd log name done_; do
      clean="$(sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$cmd")"
      get() { printf '%s\n' "$clean" | sed -n "s/^ *export $1=//p" | head -1; }
      pre=$(get nnUNet_preprocessed); res=$(get nnUNet_results); cfgp=$(get AUGLAB_PARAMS_GPU_JSON); cfg=${cfgp##*/}; ep=$(get NNUNET_NUM_EPOCHS)
      vcfg=$(get AUGLAB_VAL_PARAMS_GPU_JSON)
      tr=$(printf '%s\n' "$clean" | grep -o '\-tr [A-Za-z0-9_]*' | head -1 | cut -d' ' -f2)
      ok=OK
      case "$pre" in /scratch/*) [ -d "$pre" ] || ok="NO_PREPROC_DIR";; *) ok="PREPROC_NOT_SCRATCH";; esac
      case "$res" in /scratch/*) ;; *) ok="${ok},RESULTS_NOT_SCRATCH";; esac
      [[ "$res" == */auglab/* ]] || ok="${ok},NOT_AUGLAB_CATEGORY"
      [[ "$res" == *${NEWM}_* ]] || ok="${ok},RUNID_NOT_${TAG}"
      [ -z "${PROBE:-}" ] || [[ "$res" == *_texlite_probe* ]] || ok="${ok},PROBE_NOT_THROWAWAY"
      [ "$cfg" = "${NEWCFG}" ] || ok="${ok},WRONG_CFG($cfg)"
      [ -f "$cfgp" ] || ok="${ok},CFG_FILE_MISSING($cfgp)"
      [ -z "$vcfg" ] || ok="${ok},VAL_CFG_SET($vcfg)"
      grep -q "nnUNetv2_train" "$cmd" || ok="${ok},NO_TRAIN_CMD"
      [ -e "$res" ] && ok="${ok},RESULTS_DIR_ALREADY_EXISTS"
      printf '   %-6s %s\n      trainer=%s ep=%s cfg=%s\n      results=%s\n' "$ok" "$name" "$tr" "$ep" "$cfg" "$res"
      [ "$ok" = OK ] || bad=1
    done < "${PACKS_ROOT}/$p/index.tsv"
  done
  echo; echo "=== wrapper pairs: ${TAG} vs headline OURS (trainer, epoch default, config) ==="
  for r in "${SETTINGS[@]}"; do IFS='|' read -r name dir w orig tenv etr eep <<<"$r"
    a="${dir}/04_train/$w"; b="${dir}/04_train/$orig"
    [ -f "$a" ] || continue   # (held-out settings have no wrapper for every arm and are no longer launched)
    ta=$(grep -o '^TRAINER="[^"]*"' "$a" | cut -d'"' -f2); tb=$(grep -o '^TRAINER="[^"]*"' "$b" | cut -d'"' -f2)
    ea=$(grep -oE 'NUM_EPOCHS(:-|=)[0-9]+' "$a" | head -1 | grep -oE '[0-9]+$'); eb=$(grep -oE 'NUM_EPOCHS(:-|=)[0-9]+' "$b" | head -1 | grep -oE '[0-9]+$')
    s=OK; [ "$ta" = "$etr" ] || s="TRAINER(${ta}!=${etr})"; [ "$ea" = "$eb" ] && [ "${ea}" = "$eep" ] || s="${s},EPOCHS(${ea} vs ${eb}, expect ${eep})"
    [ "$s" = OK ] || bad=1
    printf '   %-4s %-14s %s (headline trainer %s)\n' "$s" "$name" "$ta" "$tb"
  done
  [ "$bad" = 0 ] && echo "VERIFY PASSED" || echo "VERIFY FAILED"; exit 0
fi

if [ "${MODE}" = submit ]; then
  for p in ${SEL}; do
    gmap="$(cat "${PACKS_ROOT}/${p}/GPU_MAP")"; chain="$(cat "${PACKS_ROOT}/${p}/CHAIN")"
    if [ -n "${PROBE:-}" ]; then _time="01:00:00"; _name="${TAG}probe_${p}"; else _time="${PACK_TIME:-23:59:00}"; _name="${TAG}_${p}"; fi
    PACK_GPU_TYPE=h100 PACK_NODE_GPUS=4 PACK_TIME="${_time}" PACK_CHAIN="${chain}" PACK_GPU_MAP="${gmap}" \
    PACK_USE_MPS=0 PACK_JOB_NAME="${_name}" bash scripts/job_runner/run_job_pack_submit.sh "${PACKS_ROOT}/${p}"
    sleep 5
  done
  exit 0
fi

if [ "${MODE}" = probe-report ]; then   # median s/epoch (epochs >= 3) per recorded fold, from nnU-Net's own training log
  for idx in "${PACKS_ROOT}"/*/index.tsv; do
    i=0; read -ra G <<<"$(cat "$(dirname "$idx")/GPU_MAP")"
    while IFS=$'\t' read -r cmd log name done_; do
      f="${name#fold}"; fold="${f%%_*}"; clean="$(sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$cmd")"
      res="$(printf '%s\n' "$clean" | sed -n 's/^ *export nnUNet_results=//p' | head -1)"
      L="$(ls -t "${res}"/Dataset*/*/fold_${fold}/training_log_*.txt 2>/dev/null | head -1)"
      ts=$(grep -o "Epoch time: [0-9.]* s" "$L" 2>/dev/null | awk '{print $3}' | tail -n +3 | sort -n)
      n=$(echo "$ts" | grep -c .); med=$(echo "$ts" | awk '{a[NR]=$1} END{if(NR) print a[int((NR+1)/2)]; else print "NA"}')
      ep=$(grep -c "^.*Epoch [0-9]" "$L" 2>/dev/null || true)
      cfg=$(ls "${res}"/Dataset*/*/fold_${fold}/transform_params_gpu_used_for_training.json 2>/dev/null | head -1)
      printf '%-80s GPU%s  epochs_logged=%s  median_s_per_epoch(ep>=3,n=%s)=%s  cfg_saved=%s\n' "$name" "${G[$i]}" "${ep:-0}" "$n" "$med" "${cfg:+yes}"
      i=$((i+1))
    done < "$idx"
  done
  exit 0
fi

if [ "${MODE}" = runids ]; then   # write the RUN_ID table: setting <TAB> fold <TAB> run_id <TAB> pack
  out="${1:?out tsv}"; : > "${out}"
  for idx in "${PACKS_ROOT}"/*/index.tsv; do p="$(basename "$(dirname "$idx")")"
    while IFS=$'\t' read -r cmd log name done_; do f="${name#fold}"; printf '%s\t%s\t%s\n' "${f#*_}" "${f%%_*}" "$p" >> "${out}"; done < "$idx"
  done
  sort -u "${out}" -o "${out}"; cat "${out}"; exit 0
fi

# ---- post-training: predict + evaluate (run table scripts/cluster/tamia_texlite_runs.sh; the rung-6/7 predict/eval job
# bodies, ROSTER=that table). The liver setting goes through its own whole-node predict+eval job
# (atlas-liver-hcc 05_predict/05_35_tamia_texlite_predict_eval.sh: own test + lld-mmri-hcc + liverhccseg, as its rung-4.5 job).
if [ "${MODE}" = postcheck ] || [ "${MODE}" = queue-post ]; then
    source scripts/cluster/tamia_texlite_runs.sh
    bad=0
    for r in "${RUNG_ROWS[@]}"; do IFS='|' read -r name pack dir envf tenv tc wrapper args prefix pexp ekind eexp <<<"$r"
        run="$(rung_resolve_run "${pack}" "${prefix}")" || { echo "FAIL  ${name}: cannot resolve run id for ${prefix}"; bad=1; continue; }
        w="$T/$dir/05_predict/${wrapper}"; ok=OK
        [ -f "$w" ] || ok="NO_PREDICT_WRAPPER"
        wt=$(grep -o '^TRAINER="[^"]*"' "$w" 2>/dev/null | cut -d'"' -f2); wc=$(grep -o '^CATEGORY="[^"]*"' "$w" 2>/dev/null | cut -d'"' -f2)
        tt=$(sed -e 's/\\n/\n/g' "$(ls "${PACKS_ROOT}"/*/fold0_"${run}".sh | head -1)" | grep -o '\-tr [A-Za-z0-9_]*' | head -1 | cut -d' ' -f2)
        [ "$wt" = "$tt" ] || ok="${ok},TRAINER_MISMATCH(predict=${wt} train=${tt})"
        [ "$wc" = "auglab" ] || ok="${ok},CATEGORY(${wc})"
        [ "$ok" = OK ] || bad=1
        printf '%-6s %-13s %-14s predict=%s trainer=%s run=%s\n' "${ok%%,*}" "$name" "$ekind" "$wrapper" "$wt" "$run"
        [ "$ok" = OK ] || echo "         PROBLEMS: ${ok}"
    done
    # liver row (only when a pack of this arm holds atlas_t1w folds; held out since 2026-10-15 evening)
    if printf '%s\n' "${PACKS[@]}" | grep -q ' *atlas_t1w:\|[|]atlas_t1w:'; then
    lrun="$(rung_resolve_run atlas_t1w "atlas-liver-hcc_t1w_${NEWM}" 2>/dev/null || true)"
    for w in "${AD}/05_predict/${PA}_predict_t1w_${NEWM}.sh" "${AD}/05_predict/05_35_tamia_texlite_predict_eval.sh"; do [ -f "$w" ] || { echo "FAIL  liver: missing $w"; bad=1; }; done
    echo "liver  run=${lrun:-UNRESOLVED} (predict+eval job ${AD}/05_predict/05_35_tamia_texlite_predict_eval.sh)"; [ -n "${lrun}" ] || bad=1
    fi
    [ "${MODE}" = postcheck ] && { [ "$bad" = 0 ] && echo "POSTCHECK PASSED" || echo "POSTCHECK FAILED"; exit 0; }
    [ "$bad" = 0 ] || { echo "postcheck failed -- not queueing"; exit 1; }
    cpu_submit() { local out; if out="$(sbatch --parsable "$@" 2>/dev/null)"; then echo "${out}"; return 0; fi
                   echo "  (bare sbatch failed -- retrying with --partition=cpubase_bynode_b1)" >&2
                   sbatch --parsable --partition=cpubase_bynode_b1 "$@"; }
    # post groups = settings; each waits for the LAST chain job of every pack holding one of its folds
    deps_for() { local s="$1" pk spec ids="" j; for pk in "${PACKS[@]}"; do IFS='|' read -r pn spec _ <<<"$pk"
                     [[ " ${spec} " == *" ${s}:"* ]] || continue
                     j="$(squeue -u "${USER}" -h -n "${TAG}_${pn}" -o %i | sort -n | tail -1)"; [ -n "$j" ] && ids="${ids}:${j}"; done
                 [ -n "${ids}" ] && echo "--dependency=afterany${ids}" || true; }
    GROUPS_SEL="${POST_GROUPS:-$(for pk in "${PACKS[@]}"; do IFS='|' read -r _ spec _ <<<"$pk"; for sf in ${spec}; do echo -n "${sf%%:*} "; done; done)}"
    for p in ${GROUPS_SEL}; do
        last="$(deps_for "$p")"; dep=(); [ -n "${last}" ] && dep=("${last}")
        OUT="${PACKS_ROOT}/post_${p}"; mkdir -p "${OUT}"
        has_rows=0; for r in "${RUNG_ROWS[@]}"; do IFS='|' read -r name pack _ <<<"$r"; [ "$pack" = "$p" ] && has_rows=1; done
        if [ "${has_rows}" = 1 ]; then
          PJ="$(sbatch --parsable "${dep[@]}" --job-name="${TAG}_predict_${p}" --time=06:00:00 \
                --output="${OUT}/predict_job_%j.out" --export="ALL,ROSTER=scripts/cluster/tamia_texlite_runs.sh,TEXLITE_ARM=${ARM},PACK=${p},OUT=${OUT}" scripts/cluster/tamia_rung6_pv_predict_job.sh)"
          echo "${PJ}" > "${OUT}/PREDICT_JOB"; echo "[queue-post] ${p}: predict job ${PJ} (${last:-no pending training job})"
          for r in "${RUNG_ROWS[@]}"; do IFS='|' read -r name pack _ <<<"$r"; [ "$pack" = "$p" ] || continue
              EJ="$(cpu_submit --dependency=afterany:"${PJ}" --job-name="${TAG}_eval_${name}" --time=05:00:00 --cpus-per-task=16 \
                    --mem=96G --account=aip-jcohen --output="${OUT}/eval_${name}_%j.out" \
                    --export="ALL,ROSTER=scripts/cluster/tamia_texlite_runs.sh,TEXLITE_ARM=${ARM},ROW=${name},OUT=${OUT}" scripts/cluster/tamia_rung6_pv_eval_job.sh)"
              echo "${EJ}" >> "${OUT}/EVAL_JOBS"; echo "[queue-post]   eval ${name}: job ${EJ}"; sleep 2
          done
        fi
        if [ "${p}" = atlas_t1w ]; then
          LJ="$(sbatch --parsable "${dep[@]}" --job-name="${TAG}_liver_predict_eval" --output="${OUT}/liver_predict_eval_%j.out" \
                --export="ALL,RID=${lrun},TEXLITE_ARM=${ARM}" "${AD}/05_predict/05_35_tamia_texlite_predict_eval.sh")"
          echo "${LJ}" > "${OUT}/LIVER_JOB"; echo "[queue-post] liver predict+eval job ${LJ} (run ${lrun})"
        fi
    done
    exit 0
fi
echo "unknown mode ${MODE}"; exit 2
