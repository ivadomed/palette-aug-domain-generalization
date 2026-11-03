#!/usr/bin/env bash
# Predict a LOW-DATA texture-step run (04_91) on the on-harmony cross-contrast test set (T1w T2w bold dwi_ap epi_ap
# gre_echo1_mag, all folds), checkpoint_best, no TTA, RAS -> native resample inside each predict job (05_01's
# PREDICT_POST_ITEM_CMD), exactly as every other on-harmony run; only the trainer class differs (+LowData).
#   LOWDATA_CONTRAST=<T1w|T2w|dwi_ap> bash 05_97_predict_lowdata_texture_step.sh <RUN_ID> [FOLD] [ITEM ...]
set -euo pipefail
case "${LOWDATA_CONTRAST:-T1w}" in
  T1w) source "$(dirname "$0")/../00_utils/env.sh" ;;
  T2w) source "$(dirname "$0")/../00_utils/env_t2w.sh" ;;
  dwi_ap) source "$(dirname "$0")/../00_utils/env_dwi.sh" ;;
  *) echo "ERROR: LOWDATA_CONTRAST must be T1w, T2w or dwi_ap (got '${LOWDATA_CONTRAST}')" >&2; exit 1 ;;
esac
case "${1:?RUN_ID required}" in *"_${TRAINING_CONTRAST}_lowdata_n"*) ;;
  *) echo "ERROR: ${1} is not a ${TRAINING_CONTRAST} low-data run" >&2; exit 1 ;; esac
METHOD="lowdata"
TRAINER="nnUNetTrainerOnHarmonyAugLabDefaultLowData"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
