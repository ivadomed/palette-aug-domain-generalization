#!/bin/bash -l
# T07 step 2: gates + registered rules; writes scores.txt, rule1.tsv, cells_t07.tsv, gate.tsv, PROVENANCE.txt.
#SBATCH --job-name=t07_score
#SBATCH --account=aip-jcohen
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/01_commun_results/dg_mechanisms/T07/logs/score_%j.log
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
module load python/3.11 >/dev/null 2>&1 || true
D=benchmark/01_commun_results/dg_mechanisms/T07
C=benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t07_coverage_axes
REF=/scratch/paulh/dg_mechanisms/T07/paper_ref
.venv/bin/python $C/t07_score.py --dir $D --paper $REF/_suppl_ind_ood.tex
{
  echo "T07 provenance ($(date -u +%FT%TZ))"
  echo "main repo HEAD: $(git rev-parse HEAD)"
  echo "AugLab HEAD: $(git -C sub-workspaces/auglab_workspace/AugLab rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "registration: research/dg_mechanisms/plans/2026-11-16-PREDICTION_T07.md (Killarney) sha256 771c20db47aed1b9e896e4a86254aecdf8e19da5e8391424899e59d23d5ae632, tag reg-T07-20261116"
  echo "scripts: $C/{t07_axes.py,t07_score.py,t07_axes_job.sh,t07_score_job.sh}"
  sha256sum $C/*.py $C/*.sh
  echo "inputs:"
  sha256sum benchmark/01_commun_results/dg_mechanisms/T02/cells.tsv benchmark/01_commun_results/dg_mechanisms/T02/cells_per_class.tsv
  sha256sum /scratch/paulh/dg_mechanisms/T02/features_*/manifest.json
  echo "AugLab fromSeg.py: $(sha256sum sub-workspaces/auglab_workspace/AugLab/auglab/transforms/gpu/fromSeg.py)"
  echo "settings snapshot: $(sha256sum benchmark/01_commun_results/gain_decomposition/v2/settings_snapshot.yaml)"
  echo "paper reference tables (copied from Killarney paper/cvpr_format_latex/sec):"
  sha256sum $REF/*.tex
  echo "case-level files (case ids, scratch only): /scratch/paulh/dg_mechanisms/T07/"
  echo "job logs:"; ls $D/logs
} > $D/PROVENANCE.txt
