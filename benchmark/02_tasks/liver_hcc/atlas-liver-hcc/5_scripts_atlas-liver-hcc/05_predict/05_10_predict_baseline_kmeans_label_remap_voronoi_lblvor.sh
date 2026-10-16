#!/usr/bin/env bash
# Predict atlas-liver-hcc's own held-out T1w with the rung-5 val000 retrain (see companions' 05_13).
# Usage: bash 05_09_predict_v26_6_2_train050_val000.sh <RUN_ID> [...]
set -euo pipefail
source "$(dirname "$0")/../00_utils/env.sh"
METHOD="baseline_kmeans_label_remap_voronoi_lblvor"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDefault"
DATASET_ID="80"
CATEGORY="auglab"
export PREDICT_FOLDS="0 1 2"
source "$(dirname "$0")/05_01_predict_common.sh" "$@"
