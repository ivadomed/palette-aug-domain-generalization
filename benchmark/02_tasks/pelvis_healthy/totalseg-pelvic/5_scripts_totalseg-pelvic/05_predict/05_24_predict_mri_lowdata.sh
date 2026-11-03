#!/usr/bin/env bash
# Predict a LOW-DATA run (04_35_train_mri_lowdata.sh) over ct/mri. Same as 05_22 / the rung-5 predict except
# the trainer class (+LowData), which nnUNetv2_predict needs to find the checkpoint folder.
# Usage: bash 05_24_predict_mri_lowdata.sh <RUN_ID> [FOLD] [items...]
#   CT-trained run (LOWDATA_CONTRAST=ct at training): LOWDATA_CONTRAST=ct bash 05_24_predict_mri_lowdata.sh <RUN_ID> ...
#   on a host where fold F must not land on slot/GPU F (tassan): SLOT=<s> GPU=<s> bash 05_24_... <RUN_ID> <F>
set -euo pipefail
case "${LOWDATA_CONTRAST:-mri}" in
  mri) source "$(dirname "$0")/../00_utils/env_mri.sh" ;;
  ct)  export TRAINING_CONTRAST="ct"; source "$(dirname "$0")/../00_utils/env.sh" ;;
  *) echo "ERROR: LOWDATA_CONTRAST must be mri or ct (got '${LOWDATA_CONTRAST}')" >&2; exit 1 ;;
esac
RUN_ID_ARG="${1:?RUN_ID required}"
case "${RUN_ID_ARG}" in *"_${TRAINING_CONTRAST}_lowdata_n"*) ;; *) echo "ERROR: ${RUN_ID_ARG} is not a ${TRAINING_CONTRAST} low-data run" >&2; exit 1 ;; esac
METHOD="lowdata"
TRAINER="nnUNetTrainerTotalsegPelvicAugLabDefaultLowData"
CATEGORY="auglab"
DATASET_ID="130"   # fixed -- imagesTs_* consolidated under Dataset130, see 05_01 header
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
