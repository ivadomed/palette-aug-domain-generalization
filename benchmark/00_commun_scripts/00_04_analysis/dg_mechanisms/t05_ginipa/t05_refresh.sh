#!/bin/bash
# T05 refresh, once more GIN-IPA settings are DONE + AUDIT OK (Glioma x4, Brain T1w were still predicting at the first
# scoring). Re-extracts the per-fold values (Slurm, ~9 min; extras sem50 + ginipa, a setting's GIN-IPA run is included
# only when DONE + AUDIT OK + complete in every source metrics dir) and, in a dependent job, rescores T01 and T05.
# Writes to T01/full_refresh/ (extraction) and T01/refresh/ (T01 rescoring, for comparison with T01/: the headline
# arms are unchanged by adding GIN-IPA), overwrites T05/ and appends the job ids to T05/PROVENANCE.txt.
# Usage (Vulcan login node): bash t05_refresh.sh            -> submits the two jobs
#                            bash t05_refresh.sh score      -> the scoring step only (what the dependent job runs)
set -eo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
D=benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms
R=benchmark/01_commun_results/dg_mechanisms
if [ "${1:-}" = "score" ]; then
  command -v module >/dev/null 2>&1 || source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
  module load python/3.11
  S=paper/cvpr_format_latex/sec
  mkdir -p $R/T01/refresh
  .venv/bin/python $D/t01_step_noise/t01_score.py --indood $R/T01/full_refresh/indood_folds.json \
    --ladder $R/T01/full/ladder_folds.json --tex-indood $S/_suppl_ind_ood.tex --tex-steps $S/_suppl_texture_steps.tex \
    --sem50-md paper/generated_results/sem50/sem50_ind_ood.md --out $R/T01/refresh > $R/T01/refresh/score_console.txt 2>&1
  .venv/bin/python $D/t05_ginipa/t05_score.py --indood $R/T01/full_refresh/indood_folds.json \
    --t01-gate $R/T01/refresh/gate.tsv --out $R/T05 > $R/T05/score_console.txt 2>&1
  echo "refresh $(date -Is): rescored from T01/full_refresh/indood_folds.json (scoring job ${SLURM_JOB_ID:-login}); T01 rescoring in T01/refresh/" >> $R/T05/PROVENANCE.txt
  tail -6 $R/T05/score_console.txt
  exit 0
fi
mkdir -p $R/T05/logs
J=$(sbatch --parsable $D/t01_step_noise/t01_job.sh indood full_refresh)
echo "refresh $(date -Is): extraction job $J -> T01/full_refresh/" >> $R/T05/PROVENANCE.txt
sbatch --dependency=afterok:$J --account=aip-jcohen --time=00:10:00 --cpus-per-task=1 --mem=4G \
  --output=$R/T05/logs/score_%j.log --wrap "bash $D/t05_ginipa/t05_refresh.sh score"
echo "extraction job $J; scoring queued behind it"
