#!/bin/bash -l
# T02 step 3: assemble cells.tsv and score the registered predictions; writes PROVENANCE.txt.
#SBATCH --job-name=t02_score
#SBATCH --account=aip-jcohen
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/01_commun_results/dg_mechanisms/T02/logs/score_%j.log
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
module load python/3.11 >/dev/null 2>&1 || true
D=benchmark/01_commun_results/dg_mechanisms/T02
C=benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t02_cell_features
.venv/bin/python $C/t02_score.py --dir $D
{
  echo "T02 provenance ($(date -u +%FT%TZ))"
  echo "main repo HEAD: $(git rev-parse HEAD)"
  echo "AugLab HEAD: $(git -C sub-workspaces/auglab_workspace/AugLab rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "registration: plans/2026-11-12-PREDICTION_T01-T05.md sha256 0b9f8710ec9e84eca7814db8f404b17a3e589fc1cf1a6ca8b18a2c70153b9fa2"
  echo "scripts: $C/{t02_cells.py,t02_features.py,t02_score.py} + *_job.sh"
  sha256sum $C/*.py $C/*.sh
  echo "paper reference tables (copied from Killarney paper/cvpr_format_latex/sec, identical to the Vulcan copy):"
  sha256sum /scratch/paulh/dg_mechanisms/T02/paper_ref/*.tex
  echo "settings snapshot: $(sha256sum benchmark/01_commun_results/gain_decomposition/v2/settings_snapshot.yaml)"
  echo "case-level files (case ids, scratch only): /scratch/paulh/dg_mechanisms/T02/"
  echo "job ids:"; ls $D/logs
} > $D/PROVENANCE.txt
