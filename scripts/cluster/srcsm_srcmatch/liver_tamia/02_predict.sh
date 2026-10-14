#!/bin/bash
#SBATCH --job-name=sm_liver_predict
#SBATCH --account=aip-jcohen
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=/scratch/p/paulh/srcsm_match/_status/liver_tamia_predict_%j.out
# Step 2/3 (one whole-node H100 job): predict the atlas-liver-hcc SRCSM model (checkpoint_best, folds 0 1 2) on the
# source-matched inputs of step 1, through each dataset's own predict wrapper (PREDICT_INPUT_SUFFIX=_srcmatch_t1w,
# PREDICT_OUTPUT_SUBDIR=srcmatch -> fold{k}/srcmatch/<item>). Fold commands are recorded in node-pack mode
# (RUN_JOB_PACK_DIR; one pack dir per item group so the per-fold job names never collide) and run in a worker pool,
# 3 concurrent commands per GPU, same recipe as the atlas 05_34 TamIA pack script.
#   sbatch --dependency=afterok:<match job> scripts/cluster/srcsm_srcmatch/liver_tamia/02_predict.sh
set -uo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project; cd "${ROOT}"
export SCRATCH=/scratch/p/paulh
AR="${ROOT}/benchmark/03_archive"          # TamIA layout (Vulcan moved these to benchmark/02_tasks/liver_hcc)
RID=atlas-liver-hcc_t1w_srcsm_20260804_062605
CK="${SCRATCH}/atlas-liver-hcc/8_results/01_predictions/atlas_liver_hcc_model/t1w/auglab/${RID}/Dataset080_AtlasLiverHCC/nnUNetTrainerAtlasLiverHCCAugLabDefault__nnUNetPlans__3d_fullres"
for k in 0 1 2; do [ -s "${CK}/fold_${k}/checkpoint_best.pth" ] || { echo "FAILED: no checkpoint_best fold ${k}"; exit 2; }; done
P="${SCRATCH}/srcsm_match/_packruns/liver_predict_$(date +%Y%m%d_%H%M%S)"; mkdir -p "${P}"
echo "[pred] host=$(hostname) job=${SLURM_JOB_ID:-?} PACK=${P}"
export PREDICT_INPUT_SUFFIX=_srcmatch_t1w PREDICT_OUTPUT_SUBDIR=srcmatch
record() {   # $1 group tag, $2 dataset, $3 tamia env file, $4 wrapper, rest = items
  local tag="$1" d="$2" E="$3" W="$4"; shift 4
  ( source "${AR}/${d}/5_scripts_${d}/00_utils/env.sh"; source "${ROOT}/scripts/cluster/${E}"
    cd "${AR}/${d}/5_scripts_${d}/05_predict"; RUN_JOB_PACK_DIR="${P}/${tag}" bash "${W}" "${RID}" all "$@" ) > "${P}/record_${tag}.log" 2>&1
  cat "${P}/${tag}/index.tsv" >> "${P}/index.tsv"
}
record atlas atlas-liver-hcc tamia_env_atlas-liver-hcc.sh 05_06_predict_srcsm.sh
record lld1 lld-mmri-hcc tamia_env_lld-mmri-hcc_atlascross.sh 05_06_predict_atlas_srcsm.sh t2wi dwi
record lld2 lld-mmri-hcc tamia_env_lld-mmri-hcc_atlascross.sh 05_06_predict_atlas_srcsm.sh ce-pre ce-art
record lld3 lld-mmri-hcc tamia_env_lld-mmri-hcc_atlascross.sh 05_06_predict_atlas_srcsm.sh ce-pv ce-del
record lld4 lld-mmri-hcc tamia_env_lld-mmri-hcc_atlascross.sh 05_06_predict_atlas_srcsm.sh inphase outphase
record hccseg liverhccseg tamia_env_liverhccseg_atlascross.sh 05_06_predict_atlas_srcsm.sh ce-pre_T1w ce-art_T1w ce-pv_T1w ce-del_T1w
N=$(grep -c . "${P}/index.tsv" 2>/dev/null || echo 0)
echo "[pred] recorded ${N} fold commands (expect 18)"
[ "${N}" = 18 ] || { echo "[pred] FAILED: recorded ${N} != 18"; cat "${P}"/record_*.log; exit 3; }
grep -L -- "_srcmatch_t1w" "${P}"/*/*.sh | grep -v index && { echo "[pred] FAILED: a recorded command does not read the matched inputs"; exit 3; }
grep -l -E -- "(-i|-o)[^/]{0,8}/project/" "${P}"/*/*.sh && { echo "[pred] FAILED: a recorded -i/-o path is under /project (must be scratch)"; exit 3; }
grep -h -o -E -- "-o[^/]{0,8}/scratch/p/paulh/[^ ']*" "${P}"/*/*.sh | sed 's/^/[pred] out: /'
grep -h -E "→ fold" "${P}"/record_*.log | sed 's/^/[pred] /'
mapfile -t LINES < "${P}/index.tsv"
SLOTS=12; G=4; T0=$(date +%s)
worker() {
  local slot="$1" idx="$1" gpu=$(( $1 % G ))
  while [ "${idx}" -lt "${N}" ]; do
    IFS=$'\t' read -r cmdfile log name donefile <<< "${LINES[$idx]}"
    local t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=${gpu} bash "${cmdfile}" > "${cmdfile%.sh}.log" 2>&1
    echo "[pred] slot${slot}(GPU${gpu}): '$(basename "$(dirname "${cmdfile}")")/${name}' exit=$? ($(( $(date +%s) - t0 ))s)"
    idx=$(( idx + SLOTS ))
  done
}
pids=(); for s in $(seq 0 $((SLOTS - 1))); do worker "$s" & pids+=($!); done
for p in "${pids[@]}"; do wait "$p"; done
echo "[pred] predict done ($(( $(date +%s) - T0 ))s)"
# completeness: every fold/item must have one prediction per input case
bad=0
chk() { local pd="$1" n="$2"; local m; m=$(ls "${pd}"/*.nii.gz 2>/dev/null | wc -l); echo "[pred] ${m}/${n} ${pd}"; [ "${m}" = "${n}" ] || bad=1; }
for k in 0 1 2; do
  chk "${SCRATCH}/atlas-liver-hcc/8_results/01_predictions/atlas_liver_hcc_model/t1w/auglab/${RID}/fold${k}/srcmatch/t1w" 12
  for it in t2wi dwi ce-pre ce-art ce-pv ce-del inphase outphase; do
    chk "${SCRATCH}/lld-mmri-hcc/8_results_lld-mmri-hcc/01_predictions/atlas_liver_hcc_model/t1w/auglab/${RID}/fold${k}/srcmatch/${it}" 157; done
  for it in ce-pre_T1w ce-art_T1w ce-pv_T1w ce-del_T1w; do
    chk "${SCRATCH}/liverhccseg/8_results_liverhccseg/01_predictions/atlas_liver_hcc_model/t1w/auglab/${RID}/fold${k}/srcmatch/${it}" 14; done
done
echo "[pred] DONE bad=${bad}"; exit ${bad}
