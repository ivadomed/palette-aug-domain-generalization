#!/usr/bin/env bash
# Evaluate one brats2024-glioma-trained run on BraTS-SSA 2024 (all 4 contrasts, folds 0 1 2), scoring the
# 3 labels the pre-treatment cohort has (NCR SNFH ET; RC excluded) -- the archived 06_01_evaluate_run.sh
# body, plus (a) any of the 4 training families and (b) CKPT_TAG, the project's uniform hook for a
# non-default prediction set: predictions fold{k}/<tag>/<item>, metrics <cat>_<rid>_<tag> (SRCSM's
# source-matched predictions use CKPT_TAG=srcmatch, exactly like the glioma test set's own metrics).
# Runs INLINE (call it from inside a Slurm job). Usage: FAMILY=t2f [CKPT_TAG=srcmatch] bash eval_run.sh <RUN_ID> <CATEGORY>
set -euo pipefail
case "${FAMILY:?}" in
  t1n) export BRATS_DATASET_ID=051 BRATS_DS_NAME=Dataset051_BraTS2024GliomaT1n ;;
  t2w) export BRATS_DATASET_ID=052 BRATS_DS_NAME=Dataset052_BraTS2024GliomaT2w ;;
  t2f) export BRATS_DATASET_ID=053 BRATS_DS_NAME=Dataset053_BraTS2024GliomaT2f ;;
  t1c) export BRATS_DATASET_ID=054 BRATS_DS_NAME=Dataset054_BraTS2024GliomaT1c ;;
esac
export BRATS_TRAINING_CONTRAST="${FAMILY}"
unset nnUNet_raw nnUNet_preprocessed nnUNet_results PREDICTIONS_ROOT METRICS_ROOT BRATS_DATASET_JSON
BSSA_REPO="${BSSA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"   # repo root (Vulcan or Killarney checkout)
source "${BSSA_REPO}"/benchmark/03_archive/brats-ssa2024/5_scripts_brats-ssa2024/00_utils/env.sh
cd "${PROJECT_ROOT}"
RUN_ID="${1:?RUN_ID}"; CATEGORY="${2:?CATEGORY}"
TAG="${CKPT_TAG:-}"
ITEMS="t1n t1c t2w t2f"
DJ="${BRATS_NNUNET_RAW}/${BRATS_DS_NAME}/dataset.json"
PRED_BASE="${PREDICTIONS_ROOT}/${BRATS_MODEL_TYPE}/${BRATS_TRAINING_CONTRAST}/${CATEGORY}/${RUN_ID}"
OUT_BASE="${METRICS_ROOT}/${BRATS_MODEL_TYPE}/${BRATS_TRAINING_CONTRAST}/${CATEGORY}_${RUN_ID}${TAG:+_${TAG}}"
[ -f "${DJ}" ] || { echo "ERROR: no dataset.json ${DJ}" >&2; exit 1; }
mkdir -p "${OUT_BASE}/_logs"
for F in 0 1 2; do
  for item in ${ITEMS}; do
    PRED_DIR="${PRED_BASE}/fold${F}/${TAG:+${TAG}/}${item}"
    n=$(ls "${PRED_DIR}" 2>/dev/null | command grep -c '\.nii\.gz$' || true)
    [ "${n}" = 95 ] || { echo "ERROR: ${PRED_DIR} has ${n} predictions, expected 95" >&2; exit 1; }
    echo "[$(date '+%H:%M:%S')] eval ${RUN_ID}${TAG:+ (${TAG})} fold${F} ${item}"
    .venv/bin/python benchmark/03_archive/brats-ssa2024/5_scripts_brats-ssa2024/06_evaluate/06_00_evaluate_brats_ssa.py \
      --pred_dir "${PRED_DIR}" --gt_dir "${nnUNet_raw}/labelsTs_${item}" --dataset_json "${DJ}" \
      --labels NCR SNFH ET --name "${item}" --out_csv "${OUT_BASE}/fold${F}/${item}_metrics.csv" --workers "${EVAL_WORKERS:-8}"
  done
  .venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/summarize_fold.py "${OUT_BASE}/fold${F}" "${RUN_ID}" "${F}" \
    --groups ${ITEMS} --group-col contrast --groups-word Contrasts
  rows=$(( $(wc -l < "${OUT_BASE}/fold${F}/eval_all.csv") - 1 ))
  [ "${rows}" = 1140 ] || { echo "ERROR: ${OUT_BASE}/fold${F}/eval_all.csv has ${rows} rows, expected 1140" >&2; exit 1; }
done
echo "-> ${OUT_BASE}"
