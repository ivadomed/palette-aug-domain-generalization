#!/bin/bash
# Step 3: QC table of the generated inputs + PNG of 3 public cases (scratch only), as a small run_job CPU job.
set -euo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
HERE="${ROOT}/benchmark/00_commun_scripts/00_04_analysis/texture_removal"
OUT=/scratch/paulh/texture_removal_test
cd "${ROOT}"; source scripts/job_runner/run_job.sh; export RUN_JOB_ACCOUNT="${RUN_JOB_ACCOUNT:-aip-jcohen}"
run_job --name texrm_qc --gpus 0 --cpus 2 --mem 16G --time 00:30:00 --log "${OUT}/logs/qc.log" --wait -- \
  .venv/bin/python "${HERE}/qc_texture_removal.py" --stats-dir "${OUT}/gen_stats" \
    --png-dir "${OUT}/qc" --gen-jobs "${OUT}/gen_jobs.tsv"
cat "${OUT}/logs/qc.log"
