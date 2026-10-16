#!/usr/bin/env bash
# Predict with the atlas-liver-hcc ladder rung 4 RETRAINED (lblvor noise fill; Voronoi cells inside labels) --
# checkpoint_best chosen on REAL validation images -- same trainer/category as rung 4). Clone of
# 05_11 (rung 4) with only METHOD changed. RUN_ID is required (no default: timestamp is set at launch).
# Usage: bash 05_14_predict_atlas_baseline_kmeans_label_remap_voronoi_lblvor.sh <RUN_ID> [FOLD] [ITEMS...]
set -euo pipefail
METHOD="baseline_kmeans_label_remap_voronoi_lblvor"
TRAINER="nnUNetTrainerAtlasLiverHCCAugLabDefault"
CATEGORY="auglab"
RUN_ID="${1:?RUN_ID required}"
source "$(dirname "$0")/05_01_predict_common.sh" "$RUN_ID" "${@:2}"
