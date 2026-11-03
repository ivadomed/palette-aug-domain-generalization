#!/usr/bin/env bash
# Predict a LOW-DATA texture-step run (04_45) on pansegdata's held-out test set (t1wce / t2w, all folds). Same as the rung-4.5 /
# rung-5 predict wrappers except the trainer class (+LowData), which nnUNetv2_predict needs to find the checkpoint folder.
#   LOWDATA_CONTRAST=<t1wce|t2w> bash 05_42_predict_lowdata_texture_step.sh <RUN_ID> [FOLD] [ITEM ...]
# External companions (totalsegmri-pancreas, msd-pancreas): their roster shim 05_01_predict_pansegdata_common.sh with the
# explicit RUN_ID (scripts/cluster/lowdata_vulcan/post_run.sh).
set -euo pipefail
case "${LOWDATA_CONTRAST:-t1wce}" in
  t1wce) source "$(dirname "$0")/../00_utils/env.sh"; DATASET_ID="${DATASET_ID_T1WCE}" ;;
  t2w)   source "$(dirname "$0")/../00_utils/env_t2w.sh"; DATASET_ID="${DATASET_ID_T2W}" ;;
  *) echo "ERROR: LOWDATA_CONTRAST must be t1wce or t2w (got '${LOWDATA_CONTRAST}')" >&2; exit 1 ;;
esac
case "${1:?RUN_ID required}" in *"_${TRAINING_CONTRAST}_lowdata_n"*) ;;
  *) echo "ERROR: ${1} is not a ${TRAINING_CONTRAST} low-data run" >&2; exit 1 ;; esac
METHOD="lowdata"
TRAINER="nnUNetTrainerPANSEGDATAAugLabDefaultLowData"
CATEGORY="auglab"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
