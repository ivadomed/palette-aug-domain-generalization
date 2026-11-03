#!/usr/bin/env bash
# LOW-DATA texture step (on-harmony): train ONE arm at ONE data level N.
#   LOWDATA_CONTRAST=<T1w|T2w|dwi_ap> bash 04_91_train_lowdata_texture_step.sh <flat|texture> <N> [RUN_ID]
#     flat    = rung 4.5 -- trainer/config/epochs of 04_62 (T1w) / 04_63 (T2w) / 04_64 (dwi_ap)
#     texture = rung 5, PALETTE alone val000 -- as 04_53 (T1w) / 04_55 (T2w) / 04_57 (dwi_ap)
#     (both arms: nnUNetTrainerOnHarmonyAugLabDefault, validation on real images, auglab category, 2000 epochs)
#     LOWDATA_CONTRAST = T1w (default: env.sh, Dataset031_OnHarmonyT1w31) | T2w (env_t2w.sh, Dataset032_OnHarmonyT2w31)
#                        | dwi_ap (env_dwi.sh, Dataset033_OnHarmonyDWI31)
#     N       = training SESSIONS kept per fold, from N different subjects (LowDataMixin + LOWDATA_SUBJECT_RE, see
#               02_nnunet/OnHarmonyLowDataTrainers.py; fixed seed, same subset for both arms; val unchanged)
#     RUN_ID  = resume an existing run (default on-harmony_<contrast>_lowdata_n<NN>_<arm>_<timestamp>)
# Only differences from the full-data wrappers: trainer class (+LowData) and the RUN_ID (the full-data wrappers' LOG_DIR in
# /tmp is replaced by 8_results_on-harmony/_logs/). Loader = dataset defaults (04_00_common.sh). Arm block:
# benchmark/00_commun_scripts/00_01_train/lowdata_arm.sh.
set -euo pipefail
case "${LOWDATA_CONTRAST:-T1w}" in
  T1w) source "$(dirname "$0")/../00_utils/env.sh"; DATASET_ID="031" ;;
  T2w) source "$(dirname "$0")/../00_utils/env_t2w.sh"; export DATASET_ID="032" ;;
  dwi_ap) source "$(dirname "$0")/../00_utils/env_dwi.sh"; export DATASET_ID="033" ;;
  *) echo "ERROR: LOWDATA_CONTRAST must be T1w, T2w or dwi_ap (got '${LOWDATA_CONTRAST}')" >&2; exit 1 ;;
esac
TRAINER="nnUNetTrainerOnHarmonyAugLabDefaultLowData"
export NNUNET_NUM_EPOCHS=2000
AUGLAB_CONFIGS_DIR="$(cd "${PROJECT_ROOT}/sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_01_train/lowdata_arm.sh" "$@"
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_on-harmony_${TRAINING_CONTRAST}_${METHOD}"
mkdir -p "${LOG_DIR}"
echo "[lowdata] contrast=${TRAINING_CONTRAST} dataset=${DATASET_ID} arm=${ARM} N=${N} trainer=${TRAINER} config=${CFG} epochs=${NNUNET_NUM_EPOCHS}"
source "$(dirname "$0")/04_00_common.sh" "${RUN_ID}"
