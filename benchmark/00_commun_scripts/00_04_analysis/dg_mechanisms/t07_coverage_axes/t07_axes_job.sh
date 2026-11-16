#!/bin/bash -l
# T07 step 1 for one task: appearance axes (A_res, A_noise, A_merge) of T02's OOD cells + gates.
# Usage: sbatch t07_axes_job.sh <Task> [modes]   (modes default: "images gate train merge aggregate")
#SBATCH --job-name=t07_axes
#SBATCH --account=aip-jcohen
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/01_commun_results/dg_mechanisms/T07/logs/axes_%j.log
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
module load python/3.11 >/dev/null 2>&1 || true
TASK="$1"; MODES="${2:-images gate train merge aggregate}"
OUT=benchmark/01_commun_results/dg_mechanisms/T07/axes_${TASK}
SCR=/scratch/paulh/dg_mechanisms/T07/axes_${TASK}
PY=benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t07_coverage_axes/t07_axes.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLOSC2_NTHREADS=1
FLAGS=""; for m in $MODES; do FLAGS="$FLAGS --$m"; done
echo "task $TASK modes $MODES host $(hostname) job ${SLURM_JOB_ID:-none}"
.venv/bin/python "$PY" --task "$TASK" --scratch "$SCR" --out "$OUT" --workers "${SLURM_CPUS_PER_TASK:-8}" $FLAGS
