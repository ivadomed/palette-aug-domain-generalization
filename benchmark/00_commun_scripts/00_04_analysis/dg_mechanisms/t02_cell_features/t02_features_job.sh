#!/bin/bash -l
# T02 step 2 for one task: cell -> image mapping (+ optional features and cell aggregation).
# Usage: sbatch [--dependency=afterok:<step1 job>] t02_features_job.sh <Task> [map|all]
#SBATCH --job-name=t02_feat
#SBATCH --account=aip-jcohen
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/01_commun_results/dg_mechanisms/T02/logs/features_%j.log
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
module load python/3.11 >/dev/null 2>&1 || true
TASK="$1"; MODE="${2:-all}"
OUT=benchmark/01_commun_results/dg_mechanisms/T02/features_${TASK}
SCR=/scratch/paulh/dg_mechanisms/T02/features_${TASK}
CELLS=/scratch/paulh/dg_mechanisms/T02/step1_${TASK}/cell_cases.json
PY=benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t02_cell_features/t02_features.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python "$PY" --cells "$CELLS" --scratch "$SCR" --out "$OUT" --map
if [ "$MODE" = all ]; then
  .venv/bin/python "$PY" --cells "$CELLS" --scratch "$SCR" --out "$OUT" --features --aggregate --workers "${SLURM_CPUS_PER_TASK:-8}"
fi
