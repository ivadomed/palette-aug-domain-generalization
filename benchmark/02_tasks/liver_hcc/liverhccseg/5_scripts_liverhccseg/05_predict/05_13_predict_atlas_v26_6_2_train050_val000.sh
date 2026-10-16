#!/usr/bin/env bash
# Predict with the atlas-liver-hcc ladder rung 5 RETRAINED on val000 (v26_6_2 alone, real-intensity fill,
# checkpoint_best chosen on REAL validation images -- same trainer/category as rung 4). Clone of
# 05_11 (rung 4) with only METHOD changed. RUN_ID is required (no default: timestamp is set at launch).
# Usage: bash 05_13_predict_atlas_v26_6_2_train050_val000.sh <RUN_ID> [FOLD] [ITEMS...]
set -euo pipefail
METHOD="v26_6_2_train050_val000"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:?RUN_ID required}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
