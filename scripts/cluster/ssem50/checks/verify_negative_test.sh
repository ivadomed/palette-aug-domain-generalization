# Negative test of launch.sh verify (QC must respond): re-verify the recorded B1 pack against a spec with onh_T1w swapped for onh_T2w
# (same trainer + epochs + loader) -- must print CONTRAST_OR_DATASET and VERIFY FAILED. Nothing is recorded or submitted.
export SCRATCH=/scratch/p/paulh; T=$SCRATCH/review_20261018/ssem50/tree; L=$SCRATCH/review_20261018/ssem50/neg_layout.sh
cat > $L <<'EOL'
PACKS=("B1|onh_T2w:0 chaos_t2spir:0 onh_T1w:1 chaos_t2spir:1 onh_T1w:2 chaos_t2spir:2 openms_flair:0 openms_flair:1 openms_flair:2|0 0 1 1 2 2 3 3 3|3")
EOL
cd $T; PACK_LAYOUT_FILE=$L SSEM50_REPO=$T SSEM50_STATE=$SCRATCH/review_20261018/ssem50/dryrun_state_main.txt bash scripts/cluster/ssem50/launch.sh record B1 2>&1 | grep -E "skipped|^\s+[A-Z]|VERIFY" | head -6
