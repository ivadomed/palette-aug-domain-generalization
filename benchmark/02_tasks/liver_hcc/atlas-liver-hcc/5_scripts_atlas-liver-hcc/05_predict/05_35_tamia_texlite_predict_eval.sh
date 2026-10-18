#!/bin/bash
#SBATCH --job-name=atlashcc_texlite_pred_eval
#SBATCH --account=aip-jcohen
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
# atlas-liver-hcc, PALETTE + REDUCED Auglab ("texlite", scripts/cluster/tamia_pack_texlite.sh): PREDICT + EVALUATE on one TamIA
# whole-node H100 job for atlas own (t1w) and the two cross-dataset sources of the HCC combined table (lld-mmri-hcc,
# liverhccseg). Clone of 05_34 (rung 4.5) with: RUN_ID from the environment (RID, exported by the launcher's queue-post),
# the texlite predict wrappers, companion metrics FLAT next to the headline OURS run (not ablations/), every metrics dir on
# $SCRATCH (the TamIA /project file-count quota is nearly full), lld-mmri-malignant skipped (not in the combined table).
# Usage (on TamIA): RID=<run id> sbatch --dependency=afterany:<training chain jobs> --output=<log> 05_35_tamia_texlite_predict_eval.sh
# TEXLITE_ARM=keepres: the second arm ("texlite_keepres", METHOD auglabAugTexliteKeepres_...): only the predict wrappers change
# (05_13 / companion 05_17 instead of 05_12 / 05_16); unset = the texlite arm exactly as queued 2026-10-15.
# ATLAS_OWN_ONLY=1: atlas own test set only. ATLAS_COMPANIONS_ONLY=1 (ssem50, 2026-10-18): lld-mmri-hcc + liverhccseg only (queued
# afterok behind the own-only job by scripts/cluster/ssem50/launch.sh queue-companions); the checkpoint gate still runs.
set -uo pipefail
ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
if [ -d "${ROOT}/benchmark/02_tasks/liver_hcc" ]; then AR="${ROOT}/benchmark/02_tasks/liver_hcc"; else AR="${ROOT}/benchmark/03_archive"; fi   # TamIA checkout: pre-2026-10-14 layout
A="${AR}/atlas-liver-hcc/5_scripts_atlas-liver-hcc"
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
RID="${RID:?RID (texlite run id) required}"
case "${TEXLITE_ARM:-}" in
  "") W_OWN=05_12_predict_t1w_auglabAugTexlite_v26_6_2_train050_val000.sh; W_CROSS=05_16_predict_atlas_auglabAugTexlite_v26_6_2_train050_val000.sh ;;
  keepres) W_OWN=05_13_predict_t1w_auglabAugTexliteKeepres_v26_6_2_train050_val000.sh; W_CROSS=05_17_predict_atlas_auglabAugTexliteKeepres_v26_6_2_train050_val000.sh ;;
  ssem50) W_OWN=05_36_predict_t1w_auglabAug_synthseg_EM_train050_val000.sh; W_CROSS=05_24_predict_atlas_auglabAug_synthseg_EM_train050_val000.sh ;;   # SynthSeg-EM (+Auglab; 50%) arm, scripts/cluster/ssem50/
  *) echo "[texlite] unknown TEXLITE_ARM='${TEXLITE_ARM}'"; exit 2 ;;
esac
CAT=auglab
[ "${ATLAS_OWN_ONLY:-0}" = 1 ] && [ "${ATLAS_COMPANIONS_ONLY:-0}" = 1 ] && { echo "[texlite] ATLAS_OWN_ONLY and ATLAS_COMPANIONS_ONLY are exclusive"; exit 2; }
PACK_DIR="${SCRATCH}/atlas-liver-hcc/_packruns/texlite${TEXLITE_ARM:+_${TEXLITE_ARM}}${ATLAS_COMPANIONS_ONLY:+_companions}_predict_eval_$(date +%Y%m%d_%H%M%S)"; mkdir -p "${PACK_DIR}"
echo "[texlite] host=$(hostname) job=${SLURM_JOB_ID:-?} PACK_DIR=${PACK_DIR}"

if [ "${EVAL_ATLAS_ONLY:-0}" != 1 ]; then
# ---- 0. training must be complete
RDIR="${SCRATCH}/atlas-liver-hcc/8_results/01_predictions/atlas_liver_hcc_model/t1w/${CAT}/${RID}"
for k in 0 1 2; do
  ls "${RDIR}"/Dataset*/*/fold_${k}/checkpoint_best.pth >/dev/null 2>&1 && ls "${RDIR}"/Dataset*/*/fold_${k}/checkpoint_final.pth >/dev/null 2>&1 \
    || { echo "[texlite] FAILED: fold ${k} training incomplete (no checkpoint_best/final)"; exit 2; }
done
echo "[texlite] checkpoints OK"

# ---- 1. record fold cmds: name | scripts dir | tamia env file | wrapper | items
record() {   # $1 = scripts dir, $2 = env file, $3 = wrapper, rest = items
  local S="$1" E="$2" W="$3"; shift 3
  ( source "${S}/00_utils/env.sh"; source "${ROOT}/scripts/cluster/${E}"
    cd "${S}/05_predict"; RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${W}" "${RID}" all "$@" )
}
echo "[texlite] ==== PHASE 1: record ===="
[ "${ATLAS_COMPANIONS_ONLY:-0}" = 1 ] || record "${A}" tamia_env_atlas-liver-hcc.sh "${W_OWN}"
HCC_ITEMS="t2wi dwi ce-pre ce-art ce-pv ce-del inphase outphase"
# ATLAS_OWN_ONLY=1 (ssem50, 2026-10-18): atlas own test set only; the companions are predicted later from synced checkpoints
if [ "${ATLAS_OWN_ONLY:-0}" != 1 ]; then
record "${AR}/lld-mmri-hcc/5_scripts_lld-mmri-hcc" tamia_env_lld-mmri-hcc_atlascross.sh "${W_CROSS}" ${HCC_ITEMS}
record "${AR}/liverhccseg/5_scripts_liverhccseg" tamia_env_liverhccseg_atlascross.sh "${W_CROSS}" ce-pre_T1w ce-art_T1w ce-pv_T1w ce-del_T1w
fi
NEXP=9; [ "${ATLAS_OWN_ONLY:-0}" = 1 ] && NEXP=3; [ "${ATLAS_COMPANIONS_ONLY:-0}" = 1 ] && NEXP=6
N=$(grep -c . "${PACK_DIR}/index.tsv" 2>/dev/null || echo 0)
echo "[texlite] recorded ${N} fold-predict commands (expect ${NEXP})"
[ "${N}" = "${NEXP}" ] || { echo "[texlite] FAILED: recorded ${N} != ${NEXP}"; exit 3; }

# ---- 2. worker pool, 3 concurrent tasks per GPU
echo "[texlite] ==== PHASE 2: predict ===="
mapfile -t LINES < "${PACK_DIR}/index.tsv"
SLOTS=9; G=4; T0=$(date +%s)
worker() {
  local slot="$1" idx="$1" gpu=$(( $1 % G ))
  while [ "${idx}" -lt "${N}" ]; do
    IFS=$'\t' read -r cmdfile log name donefile <<< "${LINES[$idx]}"
    local plog="${cmdfile%.sh}.log" t0=$(date +%s)
    echo "[texlite] slot${slot}(GPU${gpu}): launch '${name}'"
    CUDA_VISIBLE_DEVICES=${gpu} bash "${cmdfile}" > "${plog}" 2>&1
    echo "[texlite] slot${slot}(GPU${gpu}): '${name}' exit=$? ($(( $(date +%s) - t0 ))s)"
    idx=$(( idx + SLOTS ))
  done
}
pids=(); for s in $(seq 0 $((SLOTS - 1))); do worker "$s" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
echo "[texlite] predict done ($(( $(date +%s) - T0 ))s)"

fi   # EVAL_ATLAS_ONLY=1 skips phases 0-2 (re-run of the atlas-own eval only)
# ---- 3. evaluate (direct python, same 06_00_evaluate.py + label map as the ladder scripts)
echo "[texlite] ==== PHASE 3: evaluate ===="
EVALUATE_PY="${A}/06_evaluate/06_00_evaluate.py"; PY="${ROOT}/.venv/bin/python"
SUMMARIZE="${ROOT}/benchmark/00_commun_scripts/00_03_evaluate/summarize_fold.py"
bad=0
eval_item() {   # $1 pred_dir  $2 gt_dir  $3 eval_dir  $4 item  $5 expected N cases  $6.. extra evaluate args
  # rows = cases x ROWS_PER_CASE (atlas own scores 2 labels, liver+tumour, so 2 rows/case; companions score tumour only)
  local pd="$1" gt="$2" ed="$3" item="$4" exp="$5" rpc="${ROWS_PER_CASE:-1}"; shift 5
  local n; n=$(find "${pd}" -name '*.nii.gz' 2>/dev/null | wc -l)
  [ "${n}" -eq "${exp}" ] || { echo "[texlite] BAD pred count ${n}/${exp}: ${pd}"; return 1; }
  mkdir -p "${ed}"
  "${PY}" "${EVALUATE_PY}" --pred_dir "${pd}" --gt_dir "${gt}" --name "${item}" --out_csv "${ed}/${item}_metrics.csv" --workers 4 "$@" > "${ed}/${item}_eval.log" 2>&1
  local rows=$(( $(wc -l < "${ed}/${item}_metrics.csv") - 1 ))
  echo "[texlite] eval ${ed} ${item}: ${rows}/$(( exp * rpc ))"
  [ "${rows}" -eq $(( exp * rpc )) ]
}
export -f eval_item; export EVALUATE_PY PY

# atlas own: flat layout, scratch metrics
[ "${ATLAS_COMPANIONS_ONLY:-0}" = 1 ] || (
  source "${A}/00_utils/env.sh"; source "${ROOT}/scripts/cluster/tamia_env_atlas-liver-hcc.sh"
  export ROWS_PER_CASE=2
  for f in 0 1 2; do
    eval_item "${PREDICTIONS_ROOT}/atlas_liver_hcc_model/t1w/${CAT}/${RID}/fold${f}/t1w" "${nnUNet_raw}/Dataset080_AtlasLiverHCC/labelsTs_t1w" \
      "${METRICS_ROOT}/atlas_liver_hcc_model/t1w/${CAT}_${RID}/fold${f}" t1w 12 \
      --dataset_json "${nnUNet_raw}/Dataset080_AtlasLiverHCC/dataset.json" --labels liver tumour || exit 1
    "${PY}" "${SUMMARIZE}" "${METRICS_ROOT}/atlas_liver_hcc_model/t1w/${CAT}_${RID}/fold${f}" "${RID}" "${f}" --group-col contrast --groups-word Contrasts \
      --title-suffix " | ATLAS liver-HCC T1w CE-MRI | liver+tumour" --groups t1w > /dev/null 2>&1
  done
) || bad=1

companion() {   # $1 name  $2 env file  $3 N  $4 title  rest = items
  local d="$1" E="$2" Ncase="$3" title="$4"; shift 4
  (
    source "${AR}/${d}/5_scripts_${d}/00_utils/env.sh"; source "${ROOT}/scripts/cluster/${E}"
    export METRICS_ROOT="${SCRATCH}/${d}/8_results_${d}/02_metrics"   # scratch (the companion env.sh default is /project)
    for f in 0 1 2; do
      ed="${METRICS_ROOT}/atlas_liver_hcc_model/t1w/${CAT}_${RID}/fold${f}"
      for item in "$@"; do
        eval_item "${PREDICTIONS_ROOT}/atlas_liver_hcc_model/t1w/${CAT}/${RID}/fold${f}/${item}" "${nnUNet_raw}/labelsTs_${item}" "${ed}" "${item}" "${Ncase}" \
          --label_map '{"tumour": [2, 1]}' || exit 1
      done
      "${PY}" "${SUMMARIZE}" "${ed}" "${RID}" "${f}" --group-col contrast --groups-word Contrasts --title-suffix " | ${title} | tumour only (PALETTE + reduced Auglab)" --groups "$@" > /dev/null 2>&1
    done
    echo "[texlite] ${d} metrics -> ${METRICS_ROOT}/atlas_liver_hcc_model/t1w/${CAT}_${RID}"
  ) || bad=1
}
if [ "${EVAL_ATLAS_ONLY:-0}" != 1 ] && [ "${ATLAS_OWN_ONLY:-0}" != 1 ]; then
companion lld-mmri-hcc tamia_env_lld-mmri-hcc_atlascross.sh 157 "LLD-MMRI-HCC MRI" ${HCC_ITEMS}
companion liverhccseg tamia_env_liverhccseg_atlascross.sh 14 "LiverHccSeg MRI" ce-pre_T1w ce-art_T1w ce-pv_T1w ce-del_T1w
fi
echo "[texlite] ==== DONE bad=${bad} ===="
exit "${bad}"
