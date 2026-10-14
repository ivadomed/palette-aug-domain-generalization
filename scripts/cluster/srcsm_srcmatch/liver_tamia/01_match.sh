#!/bin/bash
#SBATCH --job-name=sm_liver_match
#SBATCH --account=aip-jcohen
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=150G
#SBATCH --output=/scratch/p/paulh/srcsm_match/_status/liver_tamia_match_%j.out
# SRCSM + test-time source matching, liver setting (atlas-liver-hcc t1w SRCSM model), TamIA port of
# scripts/cluster/srcsm_srcmatch/run_setting.sh's `liver` key, step 1/3 (CPU): average CDF of the 48 atlas training
# images (imagesTr, staged from Vulcan to $SCRATCH/srcsm_match/liver_t1w_src/imagesTr; the 12 test cases are NOT in it),
# then histogram-match every test input the model is scored on. Matched images: $SCRATCH/srcsm_match/liver_t1w/,
# symlink <input>_srcmatch_t1w next to each scratch-resident TamIA input dir.
#   sbatch scripts/cluster/srcsm_srcmatch/liver_tamia/01_match.sh
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH=/scratch/p/paulh
S="${SCRATCH}"
TR="${S}/srcsm_match/liver_t1w_src/imagesTr"
[ "$(ls "${TR}"/*_0000.nii.gz | wc -l)" = 48 ] || { echo "FAILED: expected 48 training images in ${TR}"; exit 2; }
specs=("${S}/atlas-liver-hcc/2_nnUNet/raw/Dataset080_AtlasLiverHCC/imagesTs_t1w:mr")
for it in ce-pre ce-art ce-pv ce-del t2wi dwi inphase outphase; do specs+=("${S}/lld-mmri-hcc/2_nnUNet_lld-mmri-hcc/raw/imagesTs_${it}:mr"); done
for it in ce-pre_T1w ce-art_T1w ce-pv_T1w ce-del_T1w; do specs+=("${S}/liverhccseg/2_nnUNet_liverhccseg/raw/imagesTs_${it}:mr"); done
echo "[match] host=$(hostname) job=${SLURM_JOB_ID:-?} $(date)"
bash benchmark/00_commun_scripts/00_02_predict/srcsm_match_setting.sh liver_t1w t1w "${TR}" mr "${specs[@]}"
# every matched dir must hold exactly as many volumes as its source
bad=0
for sp in "${specs[@]}"; do IN="${sp%:*}"; a=$(ls "${IN}" | grep -c '_0000.nii.gz$'); b=$(ls "${IN}_srcmatch_t1w/" | grep -c '_0000.nii.gz$')
  echo "[match] $(basename "$(dirname "${IN}")")/$(basename "${IN}"): ${b}/${a}"; [ "${a}" = "${b}" ] || bad=1; done
echo "[match] DONE bad=${bad} $(date)"; exit ${bad}
