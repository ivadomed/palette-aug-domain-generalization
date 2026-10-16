#!/bin/bash
#SBATCH --job-name=atlashcc_r45_pred_eval
#SBATCH --account=aip-jcohen
#SBATCH --time=05:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=/scratch/p/paulh/atlas-liver-hcc/_packruns/r45_predict_eval_%j.out
# atlas-liver-hcc ladder rung 4.5 (FLAT fill): PREDICT + EVALUATE on one TamIA whole-node H100 job for atlas own (t1w) and the
# three cross-dataset companions (lld-mmri-hcc, lld-mmri-malignant, liverhccseg). Same recipe as the per-dataset
# 05_20/05_21/05_23 TamIA pack scripts (record fold cmds with RUN_JOB_PACK_DIR, run them in a worker pool, then evaluate with
# 06_00_evaluate.py and merge with summarize_fold.py), merged into one job because only 3 folds x 4 datasets = 12 tasks.
# Companion metrics land in <dataset>/8_results_<dataset>/02_metrics/atlas_liver_hcc_model/t1w/ablations/auglab_<RID>/ (next to rung 4),
# atlas own in the flat layout. Queued afterany the training chain; refuses to run if a fold lacks checkpoint_best.
# Usage (on TamIA): sbatch --dependency=afterany:<last training chain job> 05_34_...sh
set -uo pipefail
ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
AR="${ROOT}/benchmark/02_tasks/liver_hcc"
A="${AR}/atlas-liver-hcc/5_scripts_atlas-liver-hcc"
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
RID="atlas-liver-hcc_t1w_baseline_kmeans_label_remap_voronoi_flatfill_20261010_073719"
CAT=auglab
PACK_DIR="${SCRATCH}/atlas-liver-hcc/_packruns/r45_predict_eval_$(date +%Y%m%d_%H%M%S)"; mkdir -p "${PACK_DIR}"
echo "[r45] host=$(hostname) job=${SLURM_JOB_ID:-?} PACK_DIR=${PACK_DIR}"

if [ "${EVAL_ATLAS_ONLY:-0}" != 1 ]; then
# ---- 0. training must be complete
RDIR="${SCRATCH}/atlas-liver-hcc/8_results/01_predictions/atlas_liver_hcc_model/t1w/${CAT}/${RID}"
for k in 0 1 2; do
  ls "${RDIR}"/Dataset*/*/fold_${k}/checkpoint_best.pth >/dev/null 2>&1 && ls "${RDIR}"/Dataset*/*/fold_${k}/checkpoint_final.pth >/dev/null 2>&1 \
    || { echo "[r45] FAILED: fold ${k} training incomplete (no checkpoint_best/final)"; exit 2; }
done
echo "[r45] checkpoints OK"

# ---- 1. record fold cmds: name | scripts dir | tamia env file | wrapper | items
record() {   # $1 = scripts dir, $2 = env file, $3 = wrapper, rest = items
  local S="$1" E="$2" W="$3"; shift 3
  ( source "${S}/00_utils/env.sh"; source "${ROOT}/scripts/cluster/${E}"
    cd "${S}/05_predict"; RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${W}" "${RID}" all "$@" )
}
echo "[r45] ==== PHASE 1: record ===="
record "${A}" tamia_env_atlas-liver-hcc.sh 05_11_predict_baseline_kmeans_label_remap_voronoi_flatfill.sh
HCC_ITEMS="t2wi dwi ce-pre ce-art ce-pv ce-del inphase outphase"
record "${AR}/lld-mmri-hcc/5_scripts_lld-mmri-hcc" tamia_env_lld-mmri-hcc_atlascross.sh 05_15_predict_atlas_baseline_kmeans_label_remap_voronoi_flatfill.sh ${HCC_ITEMS}
record "${AR}/lld-mmri-malignant/5_scripts_lld-mmri-malignant" tamia_env_lld-mmri-malignant_atlascross.sh 05_15_predict_atlas_baseline_kmeans_label_remap_voronoi_flatfill.sh ${HCC_ITEMS}
record "${AR}/liverhccseg/5_scripts_liverhccseg" tamia_env_liverhccseg_atlascross.sh 05_15_predict_atlas_baseline_kmeans_label_remap_voronoi_flatfill.sh ce-pre_T1w ce-art_T1w ce-pv_T1w ce-del_T1w
N=$(grep -c . "${PACK_DIR}/index.tsv" 2>/dev/null || echo 0)
echo "[r45] recorded ${N} fold-predict commands (expect 12)"
[ "${N}" = 12 ] || { echo "[r45] FAILED: recorded ${N} != 12"; exit 3; }

# ---- 2. worker pool, 3 concurrent tasks per GPU
echo "[r45] ==== PHASE 2: predict ===="
mapfile -t LINES < "${PACK_DIR}/index.tsv"
SLOTS=12; G=4; T0=$(date +%s)
worker() {
  local slot="$1" idx="$1" gpu=$(( $1 % G ))
  while [ "${idx}" -lt "${N}" ]; do
    IFS=$'\t' read -r cmdfile log name donefile <<< "${LINES[$idx]}"
    local plog="${cmdfile%.sh}.log" t0=$(date +%s)
    echo "[r45] slot${slot}(GPU${gpu}): launch '${name}'"
    CUDA_VISIBLE_DEVICES=${gpu} bash "${cmdfile}" > "${plog}" 2>&1
    echo "[r45] slot${slot}(GPU${gpu}): '${name}' exit=$? ($(( $(date +%s) - t0 ))s)"
    idx=$(( idx + SLOTS ))
  done
}
pids=(); for s in $(seq 0 $((SLOTS - 1))); do worker "$s" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
echo "[r45] predict done ($(( $(date +%s) - T0 ))s)"

fi   # EVAL_ATLAS_ONLY=1 skips phases 0-2 (re-run of the atlas-own eval only)
# ---- 3. evaluate (direct python, same 06_00_evaluate.py + label map as the ladder scripts)
echo "[r45] ==== PHASE 3: evaluate ===="
EVALUATE_PY="${A}/06_evaluate/06_00_evaluate.py"; PY="${ROOT}/.venv/bin/python"
SUMMARIZE="${ROOT}/benchmark/00_commun_scripts/00_03_evaluate/summarize_fold.py"
bad=0
eval_item() {   # $1 pred_dir  $2 gt_dir  $3 eval_dir  $4 item  $5 expected N cases  $6.. extra evaluate args
  # rows = cases x ROWS_PER_CASE (atlas own scores 2 labels, liver+tumour, so 2 rows/case; companions score tumour only)
  local pd="$1" gt="$2" ed="$3" item="$4" exp="$5" rpc="${ROWS_PER_CASE:-1}"; shift 5
  local n; n=$(find "${pd}" -name '*.nii.gz' 2>/dev/null | wc -l)
  [ "${n}" -eq "${exp}" ] || { echo "[r45] BAD pred count ${n}/${exp}: ${pd}"; return 1; }
  mkdir -p "${ed}"
  "${PY}" "${EVALUATE_PY}" --pred_dir "${pd}" --gt_dir "${gt}" --name "${item}" --out_csv "${ed}/${item}_metrics.csv" --workers 4 "$@" > "${ed}/${item}_eval.log" 2>&1
  local rows=$(( $(wc -l < "${ed}/${item}_metrics.csv") - 1 ))
  echo "[r45] eval ${ed} ${item}: ${rows}/$(( exp * rpc ))"
  [ "${rows}" -eq $(( exp * rpc )) ]
}
export -f eval_item; export EVALUATE_PY PY

# atlas own: flat layout, scratch metrics
(
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
    for f in 0 1 2; do
      ed="${METRICS_ROOT}/atlas_liver_hcc_model/t1w/ablations/${CAT}_${RID}/fold${f}"
      for item in "$@"; do
        eval_item "${PREDICTIONS_ROOT}/atlas_liver_hcc_model/t1w/${CAT}/${RID}/fold${f}/${item}" "${nnUNet_raw}/labelsTs_${item}" "${ed}" "${item}" "${Ncase}" \
          --label_map '{"tumour": [2, 1]}' || exit 1
      done
      "${PY}" "${SUMMARIZE}" "${ed}" "${RID}" "${f}" --group-col contrast --groups-word Contrasts --title-suffix " | ${title} | tumour only (causal-ablation ladder rung)" --groups "$@" > /dev/null 2>&1
    done
    echo "[r45] ${d} metrics -> ${METRICS_ROOT}/atlas_liver_hcc_model/t1w/ablations/${CAT}_${RID}"
  ) || bad=1
}
if [ "${EVAL_ATLAS_ONLY:-0}" != 1 ]; then
companion lld-mmri-hcc tamia_env_lld-mmri-hcc_atlascross.sh 157 "LLD-MMRI-HCC MRI" ${HCC_ITEMS}
companion lld-mmri-malignant tamia_env_lld-mmri-malignant_atlascross.sh 109 "LLD-MMRI-malignant MRI" ${HCC_ITEMS}
companion liverhccseg tamia_env_liverhccseg_atlascross.sh 14 "LiverHccSeg MRI" ce-pre_T1w ce-art_T1w ce-pv_T1w ce-del_T1w
fi
echo "[r45] ==== DONE bad=${bad} ===="
exit "${bad}"
