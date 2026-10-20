#!/usr/bin/env bash
# BraTS-SSA 2024 cross-dataset predict shim for ANY of the four brats2024-glioma training families.
# Same contract as the archived benchmark/03_archive/brats-ssa2024/5_scripts_brats-ssa2024/05_predict/
# 05_01_predict_common.sh (which only knows t1n/t2w and hardcodes a 45-min limit), delegating to the shared
# driver benchmark/00_commun_scripts/00_02_predict/predict_common.sh (checkpoint_best, --disable_tta).
#
# Env in: FAMILY (t1n|t2w|t2f|t1c), METHOD, TRAINER, CATEGORY; optional PREDICT_INPUT_SUFFIX /
#         PREDICT_OUTPUT_SUBDIR (SRCSM source matching), PREDICT_TIME (default 01:30:00).
# Usage:  bash predict_shim.sh <RUN_ID> [FOLD|all] [ITEM ...]
set -euo pipefail
case "${FAMILY:?FAMILY (t1n|t2w|t2f|t1c) required}" in
  t1n) export BRATS_DATASET_ID=051 BRATS_DS_NAME=Dataset051_BraTS2024GliomaT1n ;;
  t2w) export BRATS_DATASET_ID=052 BRATS_DS_NAME=Dataset052_BraTS2024GliomaT2w ;;
  t2f) export BRATS_DATASET_ID=053 BRATS_DS_NAME=Dataset053_BraTS2024GliomaT2f ;;
  t1c) export BRATS_DATASET_ID=054 BRATS_DS_NAME=Dataset054_BraTS2024GliomaT1c ;;
  *) echo "bad FAMILY ${FAMILY}" >&2; exit 1 ;;
esac
export BRATS_TRAINING_CONTRAST="${FAMILY}"
unset nnUNet_raw nnUNet_preprocessed nnUNet_results PREDICTIONS_ROOT METRICS_ROOT BRATS_DATASET_JSON
BSSA_REPO="${BSSA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"   # repo root (Vulcan or Killarney checkout)
source "${BSSA_REPO}"/benchmark/03_archive/brats-ssa2024/5_scripts_brats-ssa2024/00_utils/env.sh
cd "${PROJECT_ROOT}"

SOURCE_PREFIX="BRATS"
PREDICT_MODE="cross"
PREDICT_JOB_PREFIX="brats_ssa_predict_${FAMILY}"
PREDICT_LOG_PREFIX="brats_ssa_predict"
PREDICT_ITEMS_DEFAULT="t1n t1c t2w t2f"
PREDICT_FOLD_DEFAULT="all"
PREDICT_TIME="${PREDICT_TIME:-01:30:00}"
PREDICT_EXTRA_FLAGS=""

source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_02_predict/predict_common.sh" "$@"
