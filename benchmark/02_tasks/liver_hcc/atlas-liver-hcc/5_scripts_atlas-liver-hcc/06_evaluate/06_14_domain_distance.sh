#!/usr/bin/env bash
# Model-free domain distance: atlas-liver-hcc's TRAINING images vs each
# lld-mmri-hcc evaluation contrast.
#
# Why this exists (2026-09-01): the baseline (non-contrast-agnostic) model ranks
# 1st of 7 methods on `outphase` and on `ce-art`, but 5th-7th on t2wi/dwi/inphase
# -- baseline winning only where data is in-domain. `ce-art` is known in-domain;
# `outphase` was classified OOD. Deciding that from Dice would be circular, so
# this measures domain membership from the IMAGES ALONE and validates itself on
# the four CE phases already known to be in-domain.
#
# Output: 8_results_atlas-liver-hcc/02_metrics/atlas_liver_hcc_model/t1w/domain_distance/
#
# Usage:  bash 06_14_domain_distance.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../00_utils/env.sh"

SHARED="${PROJECT_ROOT}/benchmark/00_commun_scripts/00_04_analysis/domain_distance.py"
ATLAS_RAW="${PROJECT_ROOT}/benchmark/02_tasks/liver_hcc/atlas-liver-hcc/0_raw_atlas-liver-hcc/train/imagesTr"
LLD_BIDS="${PROJECT_ROOT}/benchmark/02_tasks/liver_hcc/lld-mmri-hcc/1_BIDS_lld-mmri-hcc/lld-mmri-hcc"
OUT="${PROJECT_ROOT}/benchmark/02_tasks/liver_hcc/atlas-liver-hcc/8_results_atlas-liver-hcc/02_metrics/atlas_liver_hcc_model/t1w/domain_distance"

mkdir -p "${OUT}"

run_job --name atlas_domain_dist --gpus 0 --cpus 4 --mem 16G --time 01:00:00 \
        --log "${OUT}/domain_distance.log" --wait -- \
    "${PROJECT_ROOT}/.venv/bin/python" "${SHARED}" \
        --ref-name atlas-liver-hcc-train \
        --ref-glob "${ATLAS_RAW}/im*.nii.gz" \
        --test "ce-art=${LLD_BIDS}/sub-*/anat/sub-*_ce-art_T1w.nii.gz" \
        --test "ce-pre=${LLD_BIDS}/sub-*/anat/sub-*_ce-pre_T1w.nii.gz" \
        --test "ce-pv=${LLD_BIDS}/sub-*/anat/sub-*_ce-pv_T1w.nii.gz" \
        --test "ce-del=${LLD_BIDS}/sub-*/anat/sub-*_ce-del_T1w.nii.gz" \
        --test "inphase=${LLD_BIDS}/sub-*/anat/sub-*_inphase_T1w.nii.gz" \
        --test "outphase=${LLD_BIDS}/sub-*/anat/sub-*_outphase_T1w.nii.gz" \
        --test "t2wi=${LLD_BIDS}/sub-*/anat/sub-*_T2w.nii.gz" \
        --test "dwi=${LLD_BIDS}/sub-*/anat/sub-*_dwi.nii.gz" \
        --known-in-domain ce-art,ce-pre,ce-pv,ce-del \
        --max-per-set 60 \
        --out "${OUT}"

echo "done -> ${OUT}/domain_distance.md"

# --- Stage 2: cohort-controlled ------------------------------------------------
# Stage 1 anchors on ATLAS itself, so its domain classifier saturates (AUC>0.9 for
# EVERY contrast): it separates cohort (Dijon vs the LLD-MMRI cohort -- different
# scanners, protocols, resolution) rather than contrast, and its own internal
# validation correctly refuses interpretation.
#
# Stage 2 removes that confound. ATLAS's training cohort is 33/60 arterial, so
# lld-mmri-hcc's own `ce-art` is the best available in-domain ANCHOR -- and every
# other lld contrast is the same patients, same scanners, same site, with contrast
# as the only variable. "Which contrast is closest to the in-domain anchor" is then
# answerable without any cross-cohort term. Known-IND set is the remaining CE
# phases, which must rank near the anchor for the measure to be interpretable.
OUT2="${OUT}/anchored_on_ce_art"
mkdir -p "${OUT2}"

run_job --name atlas_domain_dist_anchored --gpus 0 --cpus 4 --mem 16G --time 01:00:00 \
        --log "${OUT2}/domain_distance.log" --wait -- \
    "${PROJECT_ROOT}/.venv/bin/python" "${SHARED}" \
        --ref-name lld-ce-art-in-domain-anchor \
        --ref-glob "${LLD_BIDS}/sub-*/anat/sub-*_ce-art_T1w.nii.gz" \
        --test "ce-pv=${LLD_BIDS}/sub-*/anat/sub-*_ce-pv_T1w.nii.gz" \
        --test "ce-del=${LLD_BIDS}/sub-*/anat/sub-*_ce-del_T1w.nii.gz" \
        --test "ce-pre=${LLD_BIDS}/sub-*/anat/sub-*_ce-pre_T1w.nii.gz" \
        --test "inphase=${LLD_BIDS}/sub-*/anat/sub-*_inphase_T1w.nii.gz" \
        --test "outphase=${LLD_BIDS}/sub-*/anat/sub-*_outphase_T1w.nii.gz" \
        --test "t2wi=${LLD_BIDS}/sub-*/anat/sub-*_T2w.nii.gz" \
        --test "dwi=${LLD_BIDS}/sub-*/anat/sub-*_dwi.nii.gz" \
        --known-in-domain ce-pv,ce-del,ce-pre \
        --max-per-set 100 \
        --out "${OUT2}"

echo "done -> ${OUT2}/domain_distance.md"
