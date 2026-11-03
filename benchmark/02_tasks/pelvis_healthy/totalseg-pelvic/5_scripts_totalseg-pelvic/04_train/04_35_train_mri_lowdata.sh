#!/usr/bin/env bash
# LOW-DATA texture step (totalseg-pelvic, MRI-trained): train ONE arm at ONE data level N.
#   bash 04_35_train_mri_lowdata.sh <flat|texture> <N> [RUN_ID]
#     flat    = rung 4.5, flat fill  -- same config/trainer/epochs as 04_34_train_mri_baseline_kmeans_label_remap_voronoi_flatfill.sh
#     texture = rung 5, PALETTE alone (val000) -- same as 04_29_train_mri_v26_6_2_train050_val000.sh
#     N       = training subjects kept per fold (LowDataMixin: fixed-seed nested subset, same for both arms; val unchanged)
#     RUN_ID  = resume an existing run (default: totalseg-pelvic_mri_lowdata_n<NN>_<arm>_<timestamp>)
# The only differences from the full-data wrappers are the trainer class (+LowData, see
# 02_nnunet/TotalsegPelvicLowDataTrainers.py) and the RUN_ID marker. RUN_IDs deliberately avoid the headline method
# substrings (flatfill_, v26_6_2_train050_val000) so no headline glob can pick a low-data run up.
# Epochs unchanged (200, policy). Folds / placement / loader overrides come from the caller's env (TRAIN_FOLDS,
# FOLD_SLOT_GPU, DA_WORKERS + ALLOW_METHOD_LOADER_OVERRIDE=1), as for every wrapper.
set -euo pipefail
source "$(dirname "$0")/../00_utils/env_mri.sh"

ARM="${1:?arm required (flat|texture)}"
N="${2:?N required}"
NN="$(printf '%02d' "${N}")"

TRAINER="nnUNetTrainerTotalsegPelvicAugLabDefaultLowData"
DATASET_ID="131"
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"
AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
case "${ARM}" in
  flat)    CFG="transform_params_gpu_baseline_kmeans_label_remap_voronoi_flatfill_spatialDA_train050.json" ;;
  texture) CFG="transform_params_gpu_v26_6_2_synth_spatialDA_train050.json" ;;
  *) echo "ERROR: unknown arm '${ARM}' (flat|texture)" >&2; exit 1 ;;
esac
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/${CFG}"
export AUGLAB_VAL_PARAMS_GPU_JSON=""
export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

METHOD="lowdata_n${NN}_${ARM}"
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_totalseg_pelvic_mri_${METHOD}"
mkdir -p "${LOG_DIR}"
RUN_ID="${3:-${DATASET_NAME}_${TRAINING_CONTRAST}_${METHOD}_$(date +%Y%m%d_%H%M%S)}"
case "${RUN_ID}" in *"_lowdata_n${NN}_"*) ;; *) echo "ERROR: RUN_ID ${RUN_ID} lacks _lowdata_n${NN}_" >&2; exit 1 ;; esac

echo "[lowdata] arm=${ARM} N=${N} trainer=${TRAINER} config=${CFG} epochs=${NNUNET_NUM_EPOCHS}"
source "$(dirname "$0")/04_00_common.sh" "${RUN_ID}"
