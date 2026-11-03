#!/bin/bash
# Run ON VULCAN: write the stage list of the low-data texture-step experiment (totalseg-pelvic, MRI-trained) for tassan:
# <VSTAGE>/Pelvis.files (entries), Pelvis.sizes ("<bytes>\t<repo-relative file>", the format transfer.sh pull reads),
# Pelvis.ready. Contents: nnU-Net preprocessed Dataset131 (3d_fullres only + plans, fingerprint, dataset.json, splits,
# gt_segmentations), raw Dataset131 dataset.json, raw Dataset130 dataset.json + test images imagesTs_{ct,mri} (labels stay
# on Vulcan: evaluation runs here), the MRI split files the trainer checks (splits_final.json, test_cases.json), and the
# two full-data runs' transform_params_gpu_used_for_training.json (tassan checks its low-data runs use the same configs).
#   bash scripts/cluster/lowdata_tassan/stage_list.sh
# On tassan: VSTAGE=/scratch/paulh/lowdata_tassan bash benchmark/00_commun_scripts/00_04_analysis/texture_removal/transfer.sh pull Pelvis
# CT-trained direction: LOWDATA_CONTRAST=ct bash stage_list.sh -> PelvisCT.{files,sizes,ready}, same contents with the CT
# side (preprocessed Dataset130, raw Dataset130 dataset.json + both test-image dirs, CT split files, the two CT full-data
# runs' used configs); on tassan: ... transfer.sh pull PelvisCT
# pansegdata (training only on tassan, run_panseg.sh): LOWDATA_TASK=panseg bash stage_list.sh -> Pancreas.{files,sizes,ready}:
# preprocessed Dataset150 + Dataset151 (3d_fullres + plans, fingerprint, dataset.json, splits, gt_segmentations), both raw
# dataset.json, 4_splits_pansegdata/{splits_final,test_cases}.json, the four full-data rung-4.5 / rung-5 runs' used configs and
# the Vulcan low-data run table (RUN_IDs + job ids; gitignored, so it travels here). No test images (prediction runs on Vulcan).
#   on tassan: ... transfer.sh pull Pancreas
set -euo pipefail
REPO=/project/aip-jcohen/paulh/mri_synthesis_project
VSTAGE="${VSTAGE:-/scratch/paulh/lowdata_tassan}"
write_list() {   # $1 list name, rest: entries -> <VSTAGE>/<name>.{files,sizes}
  local L="$1"; shift
  printf '%s\n' "$@" > "${VSTAGE}/${L}.files"; : > "${VSTAGE}/${L}.sizes.tmp"
  for e in "$@"; do
    [ -e "${e}" ] || { echo "ERROR: missing ${e}" >&2; exit 1; }
    find -L "${e}" -type f -printf '%s\t%p\n' >> "${VSTAGE}/${L}.sizes.tmp"
  done
  sort -k2 "${VSTAGE}/${L}.sizes.tmp" > "${VSTAGE}/${L}.sizes"; rm "${VSTAGE}/${L}.sizes.tmp"
}
if [ "${LOWDATA_TASK:-pelvis}" = panseg ]; then
  D=benchmark/02_tasks/pancreas_disease/pansegdata
  mkdir -p "${VSTAGE}"; cd "${REPO}"
  entries=("${D}/4_splits_pansegdata/splits_final.json" "${D}/4_splits_pansegdata/test_cases.json"
           "${D}/8_results_pansegdata/_launch/RUN_IDS_lowdata_vulcan.tsv")
  for DS in Dataset150_PanSegData_T1WCE Dataset151_PanSegData_T2W; do
    P="${D}/2_nnUNet_pansegdata/preprocessed/${DS}"
    entries+=("${P}/dataset.json" "${P}/dataset_fingerprint.json" "${P}/nnUNetPlans.json" "${P}/splits_final.json"
              "${P}/gt_segmentations" "${P}/nnUNetPlans_3d_fullres" "${D}/2_nnUNet_pansegdata/raw/${DS}/dataset.json")
  done
  for r in t1wce/auglab/pansegdata_t1wce_baseline_kmeans_label_remap_voronoi_flatfill_20261009_080433 \
           t1wce/auglab/pansegdata_t1wce_v26_6_2_train050_val000_20261005_223237 \
           t2w/auglab/pansegdata_t2w_baseline_kmeans_label_remap_voronoi_flatfill_20261009_080507 \
           t2w/auglab/pansegdata_t2w_v26_6_2_train050_val000_20261005_223311; do
    entries+=("$(ls -d "${D}/8_results_pansegdata/01_predictions/pansegdata_model/${r}"/Dataset*/nnUNetTrainerPANSEGDATAAugLabDefault__nnUNetPlans__3d_fullres/fold_0/transform_params_gpu_used_for_training.json)")
  done
  write_list Pancreas "${entries[@]}"
  echo "main $(git rev-parse --short HEAD); $(wc -l < "${VSTAGE}/Pancreas.sizes") files, $(awk -F'\t' '{s+=$1} END {printf "%.2f", s/1e9}' "${VSTAGE}/Pancreas.sizes") GB; $(date -u +%FT%TZ)" > "${VSTAGE}/Pancreas.ready"
  cat "${VSTAGE}/Pancreas.ready"; exit 0
fi
C="${LOWDATA_CONTRAST:-mri}"
case "${C}" in
  mri) DSNAME=Dataset131_TotalsegPelvic_MRI; LIST=Pelvis
       REFS="totalseg-pelvic_mri_baseline_kmeans_label_remap_voronoi_flatfill_20261008_191257 totalseg-pelvic_mri_v26_6_2_train050_val000_20261005_223058" ;;
  ct)  DSNAME=Dataset130_TotalsegPelvic_CT; LIST=PelvisCT
       REFS="totalseg-pelvic_ct_baseline_kmeans_label_remap_voronoi_flatfill_20261008_191224 totalseg-pelvic_ct_v26_6_2_train050_val000_20261005_223025" ;;
  *) echo "LOWDATA_CONTRAST must be mri or ct" >&2; exit 1 ;;
esac
D=benchmark/02_tasks/pelvis_healthy/totalseg-pelvic
PRE="${D}/2_nnUNet_totalseg-pelvic/preprocessed/${DSNAME}"
RAW="${D}/2_nnUNet_totalseg-pelvic/raw"
RUNS="${D}/8_results_totalseg-pelvic/01_predictions/totalseg_pelvic_model/${C}/auglab"
mkdir -p "${VSTAGE}"
cd "${REPO}"
entries=(
  "${PRE}/dataset.json" "${PRE}/dataset_fingerprint.json" "${PRE}/nnUNetPlans.json" "${PRE}/splits_final.json"
  "${PRE}/gt_segmentations" "${PRE}/nnUNetPlans_3d_fullres"
  "${RAW}/Dataset130_TotalsegPelvic_CT/dataset.json" "${RAW}/Dataset130_TotalsegPelvic_CT/imagesTs_ct"
  "${RAW}/Dataset130_TotalsegPelvic_CT/imagesTs_mri"
  "${D}/4_splits_totalseg-pelvic/${C}/splits_final.json" "${D}/4_splits_totalseg-pelvic/${C}/test_cases.json"
)
[ "${C}" = mri ] && entries=("${entries[@]:0:6}" "${RAW}/${DSNAME}/dataset.json" "${entries[@]:6}")   # MRI raw dataset.json (CT's is listed above)
for r in ${REFS}; do
  entries+=("$(ls -d "${RUNS}/${r}"/${DSNAME}/*/fold_0/transform_params_gpu_used_for_training.json)")
done
printf '%s\n' "${entries[@]}" > "${VSTAGE}/${LIST}.files"
: > "${VSTAGE}/${LIST}.sizes.tmp"
for e in "${entries[@]}"; do
  [ -e "${e}" ] || { echo "ERROR: missing ${e}" >&2; exit 1; }
  find -L "${e}" -type f -printf '%s\t%p\n' >> "${VSTAGE}/${LIST}.sizes.tmp"
done
sort -k2 "${VSTAGE}/${LIST}.sizes.tmp" > "${VSTAGE}/${LIST}.sizes"; rm "${VSTAGE}/${LIST}.sizes.tmp"
n_ct=$(ls "${RAW}/Dataset130_TotalsegPelvic_CT/imagesTs_ct" | wc -l); n_mri=$(ls "${RAW}/Dataset130_TotalsegPelvic_CT/imagesTs_mri" | wc -l)
echo "main $(git rev-parse --short HEAD); test cases ct=${n_ct} mri=${n_mri}; $(wc -l < "${VSTAGE}/${LIST}.sizes") files, $(awk -F'\t' '{s+=$1} END {printf "%.2f", s/1e9}' "${VSTAGE}/${LIST}.sizes") GB; $(date -u +%FT%TZ)" > "${VSTAGE}/${LIST}.ready"
cat "${VSTAGE}/${LIST}.ready"
