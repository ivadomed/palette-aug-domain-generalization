#!/bin/bash
#SBATCH --job-name=sm_liver_eval
#SBATCH --account=aip-jcohen
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=150G
#SBATCH --output=/scratch/p/paulh/srcsm_match/_status/liver_tamia_eval_%j.out
# Step 3/3 (CPU): evaluate the source-matched SRCSM predictions with each dataset's own evaluator and CKPT_TAG=srcmatch
# (reads fold{k}/srcmatch/<item>, writes <cat>_<RUN_ID>_srcmatch next to the plain SRCSM metrics), run inline
# (RUN_JOB_INLINE=1), then audit the scored case sets against the plain SRCSM metrics copied from Vulcan
# ($SCRATCH/srcsm_match/liver_t1w_vulcan_plain_metrics/<ds>). lld-mmri-hcc / liverhccseg score the tumour only.
# Atlas-own test set: its own evaluator unchanged (liver + tumour rows; tumour-only is applied at aggregation). ATLAS_OWN=0 skips it.
#   sbatch --dependency=afterok:<predict job> scripts/cluster/srcsm_srcmatch/liver_tamia/03_eval.sh
#   ATLAS_OWN=1 COMPANIONS=0 sbatch --export=ALL scripts/cluster/srcsm_srcmatch/liver_tamia/03_eval.sh
set -uo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project; cd "${ROOT}"
export SCRATCH=/scratch/p/paulh
AR="${ROOT}/benchmark/03_archive"
RID=atlas-liver-hcc_t1w_srcsm_20260804_062605
PLAIN="${SCRATCH}/srcsm_match/liver_t1w_vulcan_plain_metrics"
AUDIT="${ROOT}/scripts/cluster/rung5_val000/audit.py"
export RUN_JOB_INLINE=1 CKPT_TAG=srcmatch
bad=0
echo "[eval] host=$(hostname) job=${SLURM_JOB_ID:-?} $(date)"
if [ "${COMPANIONS:-1}" = 1 ]; then
  for d in lld-mmri-hcc liverhccseg; do
    (
      source "${AR}/${d}/5_scripts_${d}/00_utils/env.sh"
      export nnUNet_raw="${SCRATCH}/${d}/2_nnUNet_${d}/raw" PREDICTIONS_ROOT="${SCRATCH}/${d}/8_results_${d}/01_predictions" CATEGORY=auglab
      grep -q CKPT_TAG "${AR}/${d}/5_scripts_${d}/06_evaluate/06_01_evaluate_run.sh" || { echo "[eval] FAILED: ${d} evaluator has no CKPT_TAG"; exit 1; }
      bash "${AR}/${d}/5_scripts_${d}/06_evaluate/06_01_evaluate_run.sh" "${RID}" all
      M="${METRICS_ROOT}/atlas_liver_hcc_model/t1w/auglab_${RID}_srcmatch"
      echo "[eval] ${d} -> ${M}"
      "${ROOT}/.venv/bin/python" "${AUDIT}" "${PLAIN}/${d}" "${M}" "liver_${d}"
    ) || bad=1
  done
fi
if [ "${ATLAS_OWN:-1}" = 1 ]; then
  (
    source "${AR}/atlas-liver-hcc/5_scripts_atlas-liver-hcc/00_utils/env.sh"; source "${ROOT}/scripts/cluster/tamia_env_atlas-liver-hcc.sh"
    bash "${AR}/atlas-liver-hcc/5_scripts_atlas-liver-hcc/06_evaluate/06_01_evaluate_run.sh" "${RID}" auglab
    # the evaluator re-sources env.sh, which exports METRICS_ROOT unconditionally (project archive path, not tamia_env's
    # scratch one); PREDICTIONS_ROOT/nnUNet_raw keep the scratch overrides through common_env.sh's guard
    M="${AR}/atlas-liver-hcc/8_results_atlas-liver-hcc/02_metrics/atlas_liver_hcc_model/t1w/auglab_${RID}_srcmatch"
    echo "[eval] atlas-liver-hcc -> ${M}"
    "${ROOT}/.venv/bin/python" "${AUDIT}" "${PLAIN}/atlas-liver-hcc" "${M}" "liver_atlas-liver-hcc"
  ) || bad=1
fi
echo "[eval] DONE bad=${bad} $(date)"; exit ${bad}
