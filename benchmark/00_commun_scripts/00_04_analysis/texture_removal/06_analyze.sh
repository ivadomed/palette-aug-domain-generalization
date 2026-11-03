#!/bin/bash
# Step 6 (Vulcan): DiD table + group permutation (analyze.py) as a small run_job CPU job. Output (aggregates) in the
# log and /scratch/paulh/texture_removal_test/analysis/texture_removal_did.tsv (scratch, never git).
#   bash 06_analyze.sh [analyze.py args]
set -euo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
HERE="${ROOT}/benchmark/00_commun_scripts/00_04_analysis/texture_removal"
OUT=/scratch/paulh/texture_removal_test; mkdir -p "${OUT}/logs" "${OUT}/analysis"
cd "${ROOT}"; source scripts/job_runner/run_job.sh; export RUN_JOB_ACCOUNT="${RUN_JOB_ACCOUNT:-aip-jcohen}"
run_job --name texrm_analyze --gpus 0 --cpus 2 --mem 16G --time 00:30:00 --log "${OUT}/logs/analyze.log" --wait -- \
  .venv/bin/python "${HERE}/analyze.py" --out "${OUT}/analysis/texture_removal_did.tsv" "$@"
cat "${OUT}/logs/analyze.log"
