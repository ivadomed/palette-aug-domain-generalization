#!/bin/bash -l
# T02 step 1: OOD cells + per-cell arm Dice + gates. Usage: sbatch t02_cells_job.sh [OUT_SUBDIR] [TASKS]
#SBATCH --job-name=t02_cells
#SBATCH --account=aip-jcohen
#SBATCH --time=00:50:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --output=/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/01_commun_results/dg_mechanisms/T02/logs/cells_%j.log
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
module load python/3.11 >/dev/null 2>&1 || true
SUB="${1:-full}"; TASKS="${2:-}"
OUT=benchmark/01_commun_results/dg_mechanisms/T02/step1_${SUB}
SCR=/scratch/paulh/dg_mechanisms/T02/step1_${SUB}
.venv/bin/python benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t02_cell_features/t02_cells.py \
  --out "$OUT" --scratch "$SCR" --paper-ref /scratch/paulh/dg_mechanisms/T02/paper_ref ${TASKS:+--tasks "$TASKS"}
