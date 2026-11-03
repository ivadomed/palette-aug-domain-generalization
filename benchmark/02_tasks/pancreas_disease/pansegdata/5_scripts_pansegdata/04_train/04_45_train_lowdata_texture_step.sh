#!/usr/bin/env bash
# LOW-DATA texture step (pansegdata): train ONE arm at ONE data level N.
#   LOWDATA_CONTRAST=<t1wce|t2w> bash 04_45_train_lowdata_texture_step.sh <flat|texture> <N> [RUN_ID]
#     flat    = rung 4.5 -- trainer/config/epochs of 04_33 (t1wce) / 04_34 (t2w)
#     texture = rung 5, PALETTE alone val000 -- as 04_18 (t1wce) / 04_22 (t2w)
#     (both arms: nnUNetTrainerPANSEGDATAAugLabDefault, validation on real images, auglab category, 2000 epochs)
#     LOWDATA_CONTRAST = t1wce (default: env.sh, Dataset150) | t2w (env_t2w.sh, Dataset151)
#     N       = training subjects kept per fold (LowDataMixin: fixed-seed nested subset, same for both arms; val unchanged)
#     RUN_ID  = resume an existing run (default pansegdata_<contrast>_lowdata_n<NN>_<arm>_<timestamp>)
# Only differences from the full-data wrappers: trainer class (+LowData, 02_nnunet/PANSEGDATALowDataTrainers.py) and the
# RUN_ID. Loader = dataset defaults (04_00_common.sh). Arm block: benchmark/00_commun_scripts/00_01_train/lowdata_arm.sh.
set -euo pipefail
case "${LOWDATA_CONTRAST:-t1wce}" in
  t1wce) source "$(dirname "$0")/../00_utils/env.sh"; DATASET_ID="${DATASET_ID_T1WCE}" ;;
  t2w)   source "$(dirname "$0")/../00_utils/env_t2w.sh"; DATASET_ID="${DATASET_ID_T2W}" ;;
  *) echo "ERROR: LOWDATA_CONTRAST must be t1wce or t2w (got '${LOWDATA_CONTRAST}')" >&2; exit 1 ;;
esac
TRAINER="nnUNetTrainerPANSEGDATAAugLabDefaultLowData"
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"
AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_01_train/lowdata_arm.sh" "$@"
export AUGLAB_VAL_PARAMS_GPU_JSON=""
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_pansegdata_${TRAINING_CONTRAST}_${METHOD}"
mkdir -p "${LOG_DIR}"
echo "[lowdata] contrast=${TRAINING_CONTRAST} dataset=${DATASET_ID} arm=${ARM} N=${N} trainer=${TRAINER} config=${CFG} epochs=${NNUNET_NUM_EPOCHS}"
source "$(dirname "$0")/04_00_common.sh" "${RUN_ID}"
