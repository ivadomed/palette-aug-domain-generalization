#!/usr/bin/env bash
# LOW-DATA texture step (chaos): train ONE arm at ONE data level N.
#   LOWDATA_CONTRAST=<t1in|t2spir> bash 04_93_train_lowdata_texture_step.sh <flat|texture> <N> [RUN_ID]
#     flat    = rung 4.5 -- trainer/config/epochs of 04_67 (t1in) / 04_68 (t2spir): nnUNetTrainerCHAOSAugLabDefault
#     texture = rung 5, PALETTE alone val000 -- as 04_48 (t1in) / 04_63 (t2spir): nnUNetTrainerCHAOSAugLabDualVal with the
#               synth-only validation config (checkpoint_best = clean validation; the val100 mirror is skipped, no _val000_)
#     LOWDATA_CONTRAST = t1in (default: env.sh, Dataset060) | t2spir (env_t2spir.sh, Dataset061)
#     N       = training subjects kept per fold (LowDataMixin: fixed-seed nested subset, same for both arms; val unchanged)
#     RUN_ID  = resume an existing run (default chaos_<contrast>_lowdata_n<NN>_<arm>_<timestamp>)
# Only differences from the full-data wrappers: trainer class (+LowData, 02_nnunet/CHAOSLowDataTrainers.py) and the RUN_ID.
# nnUNet category (no NNUNET_RESULTS_BASE), 200 epochs, dataset loader defaults (04_00_common.sh). Arm block:
# benchmark/00_commun_scripts/00_01_train/lowdata_arm.sh.
set -euo pipefail
case "${LOWDATA_CONTRAST:-t1in}" in
  t1in)   source "$(dirname "$0")/../00_utils/env.sh"; DATASET_ID="060" ;;
  t2spir) source "$(dirname "$0")/../00_utils/env_t2spir.sh"; DATASET_ID="061" ;;
  *) echo "ERROR: LOWDATA_CONTRAST must be t1in or t2spir (got '${LOWDATA_CONTRAST}')" >&2; exit 1 ;;
esac
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"
AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_01_train/lowdata_arm.sh" "$@"
case "${ARM}" in
  flat)    TRAINER="nnUNetTrainerCHAOSAugLabDefaultLowData"; export AUGLAB_VAL_PARAMS_GPU_JSON="" ;;
  texture) TRAINER="nnUNetTrainerCHAOSAugLabDualValLowData"
           export AUGLAB_VAL_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json" ;;
esac
LOG_DIR="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/_logs/nnunet_chaos_${TRAINING_CONTRAST}_${METHOD}"
mkdir -p "${LOG_DIR}"
echo "[lowdata] contrast=${TRAINING_CONTRAST} dataset=${DATASET_ID} arm=${ARM} N=${N} trainer=${TRAINER} config=${CFG} epochs=${NNUNET_NUM_EPOCHS}"
source "$(dirname "$0")/04_00_common.sh" "${RUN_ID}"
