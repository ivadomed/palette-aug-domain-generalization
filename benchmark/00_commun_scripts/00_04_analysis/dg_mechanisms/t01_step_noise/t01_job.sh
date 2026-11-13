#!/bin/bash
# T01/T05 step 1: per-fold arm values. Usage: sbatch t01_job.sh PART(indood|ladder) OUTSUB [SETTINGS] [EXTRAS]
#SBATCH --job-name=t01_folds
#SBATCH --account=aip-jcohen
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/01_commun_results/dg_mechanisms/T01/logs/folds_%j.log
echo "start $(date -Is) $(hostname)"
set -o pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project || exit 1
command -v module >/dev/null 2>&1 || source /cvmfs/soft.computecanada.ca/config/profile/bash.sh
module load python/3.11 || echo "WARN module load failed"
set -eu
PART="$1"; SUB="$2"; SETTINGS="${3:-}"; EXTRAS="${4:-sem50,ginipa}"
OUT=benchmark/01_commun_results/dg_mechanisms/T01/${SUB}
SCR=/scratch/paulh/dg_mechanisms/T01/${SUB}_${PART}
mkdir -p "$OUT" "$SCR"
export MPLBACKEND=Agg OMP_NUM_THREADS=2
echo "job $SLURM_JOB_ID part $PART sub $SUB host $(hostname) HEAD $(git rev-parse HEAD) $(date -Is)"
.venv/bin/python benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t01_step_noise/t01_fold_values.py \
  --part "$PART" --out "$OUT" --scratch "$SCR" --extras "$EXTRAS" ${SETTINGS:+--settings "$SETTINGS"}
