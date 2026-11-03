#!/usr/bin/env bash
# Predict a LOW-DATA texture-step run (04_93) on chaos's own test items (t1in t1out t2spir ct). Same as the rung-4.5 / rung-5
# predict wrappers (05_53 / 05_54, 05_38) except the trainer class (+LowData), which nnUNetv2_predict needs to find the
# checkpoint folder; the arm is read from the RUN_ID.
#   LOWDATA_CONTRAST=<t1in|t2spir> bash 05_79_predict_lowdata_texture_step.sh <RUN_ID> [FOLD] [MODALITY ...]
set -euo pipefail
case "${LOWDATA_CONTRAST:-t1in}" in
  t1in)   source "$(dirname "$0")/../00_utils/env.sh"; DATASET_ID="60" ;;
  t2spir) source "$(dirname "$0")/../00_utils/env_t2spir.sh"; DATASET_ID="61" ;;
  *) echo "ERROR: LOWDATA_CONTRAST must be t1in or t2spir (got '${LOWDATA_CONTRAST}')" >&2; exit 1 ;;
esac
RUN_ID_ARG="${1:?RUN_ID required}"
case "${RUN_ID_ARG}" in
  *"_${TRAINING_CONTRAST}_lowdata_n"*"_flat_"*)    TRAINER="nnUNetTrainerCHAOSAugLabDefaultLowData" ;;
  *"_${TRAINING_CONTRAST}_lowdata_n"*"_texture_"*) TRAINER="nnUNetTrainerCHAOSAugLabDualValLowData" ;;
  *) echo "ERROR: ${RUN_ID_ARG} is not a ${TRAINING_CONTRAST} low-data texture-step run" >&2; exit 1 ;;
esac
METHOD="lowdata"
CATEGORY="nnUNet"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
