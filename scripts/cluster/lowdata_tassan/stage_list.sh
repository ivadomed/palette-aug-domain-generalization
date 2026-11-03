#!/bin/bash
# Run ON VULCAN: write the stage list of the low-data texture-step experiment (totalseg-pelvic, MRI-trained) for tassan:
# <VSTAGE>/Pelvis.files (entries), Pelvis.sizes ("<bytes>\t<repo-relative file>", the format transfer.sh pull reads),
# Pelvis.ready. Contents: nnU-Net preprocessed Dataset131 (3d_fullres only + plans, fingerprint, dataset.json, splits,
# gt_segmentations), raw Dataset131 dataset.json, raw Dataset130 dataset.json + test images imagesTs_{ct,mri} (labels stay
# on Vulcan: evaluation runs here), the MRI split files the trainer checks (splits_final.json, test_cases.json), and the
# two full-data runs' transform_params_gpu_used_for_training.json (tassan checks its low-data runs use the same configs).
#   bash scripts/cluster/lowdata_tassan/stage_list.sh
# On tassan: VSTAGE=/scratch/paulh/lowdata_tassan bash benchmark/00_commun_scripts/00_04_analysis/texture_removal/transfer.sh pull Pelvis
set -euo pipefail
REPO=/project/aip-jcohen/paulh/mri_synthesis_project
VSTAGE="${VSTAGE:-/scratch/paulh/lowdata_tassan}"
D=benchmark/02_tasks/pelvis_healthy/totalseg-pelvic
PRE="${D}/2_nnUNet_totalseg-pelvic/preprocessed/Dataset131_TotalsegPelvic_MRI"
RAW="${D}/2_nnUNet_totalseg-pelvic/raw"
RUNS="${D}/8_results_totalseg-pelvic/01_predictions/totalseg_pelvic_model/mri/auglab"
mkdir -p "${VSTAGE}"
cd "${REPO}"
entries=(
  "${PRE}/dataset.json" "${PRE}/dataset_fingerprint.json" "${PRE}/nnUNetPlans.json" "${PRE}/splits_final.json"
  "${PRE}/gt_segmentations" "${PRE}/nnUNetPlans_3d_fullres"
  "${RAW}/Dataset131_TotalsegPelvic_MRI/dataset.json"
  "${RAW}/Dataset130_TotalsegPelvic_CT/dataset.json" "${RAW}/Dataset130_TotalsegPelvic_CT/imagesTs_ct"
  "${RAW}/Dataset130_TotalsegPelvic_CT/imagesTs_mri"
  "${D}/4_splits_totalseg-pelvic/mri/splits_final.json" "${D}/4_splits_totalseg-pelvic/mri/test_cases.json"
)
for r in totalseg-pelvic_mri_baseline_kmeans_label_remap_voronoi_flatfill_20261008_191257 \
         totalseg-pelvic_mri_v26_6_2_train050_val000_20261005_223058; do
  entries+=("$(ls -d "${RUNS}/${r}"/Dataset131_TotalsegPelvic_MRI/*/fold_0/transform_params_gpu_used_for_training.json)")
done
printf '%s\n' "${entries[@]}" > "${VSTAGE}/Pelvis.files"
: > "${VSTAGE}/Pelvis.sizes.tmp"
for e in "${entries[@]}"; do
  [ -e "${e}" ] || { echo "ERROR: missing ${e}" >&2; exit 1; }
  find -L "${e}" -type f -printf '%s\t%p\n' >> "${VSTAGE}/Pelvis.sizes.tmp"
done
sort -k2 "${VSTAGE}/Pelvis.sizes.tmp" > "${VSTAGE}/Pelvis.sizes"; rm "${VSTAGE}/Pelvis.sizes.tmp"
n_ct=$(ls "${RAW}/Dataset130_TotalsegPelvic_CT/imagesTs_ct" | wc -l); n_mri=$(ls "${RAW}/Dataset130_TotalsegPelvic_CT/imagesTs_mri" | wc -l)
echo "main $(git rev-parse --short HEAD); test cases ct=${n_ct} mri=${n_mri}; $(wc -l < "${VSTAGE}/Pelvis.sizes") files, $(awk -F'\t' '{s+=$1} END {printf "%.2f", s/1e9}' "${VSTAGE}/Pelvis.sizes") GB; $(date -u +%FT%TZ)" > "${VSTAGE}/Pelvis.ready"
cat "${VSTAGE}/Pelvis.ready"
