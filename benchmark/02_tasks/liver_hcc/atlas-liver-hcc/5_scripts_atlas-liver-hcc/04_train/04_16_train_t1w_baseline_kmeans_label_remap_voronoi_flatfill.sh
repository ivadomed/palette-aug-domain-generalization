#!/usr/bin/env bash
# GENERATED from 04_15 (rung 4 lblvor): only the transform config (-> *_flatfill_*: ladder rung 4.5, FLAT fill = PALETTE partition at region target means, no noise, no texture; scripts/cluster/rung45_flat) and METHOD/LOG_DIR suffix _flatfill changed.
# and METHOD/LOG_DIR suffix _flatfill changed. Archive depth: AugLab hop is 5 levels.
source "$(dirname "$0")/../00_utils/env.sh"

METHOD="baseline_kmeans_label_remap_voronoi_flatfill"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDefault"
DATASET_ID="080"
LOG_DIR="${RESULTS_DIR}/_logs/nnunet_atlas-liver-hcc_baseline_kmeans_label_remap_voronoi_flatfill"
export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-2000}"

AUGLAB_CONFIGS_DIR="$(cd "$(dirname "$0")/../../../../../../sub-workspaces/auglab_workspace/AugLab/auglab/configs" && pwd)"
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/transform_params_gpu_baseline_kmeans_label_remap_voronoi_flatfill_spatialDA_train050.json"
export AUGLAB_VAL_PARAMS_GPU_JSON=""

export NNUNET_RESULTS_BASE="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab"

source "$(dirname "$0")/04_00_common.sh" "$@"
