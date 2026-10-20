#!/bin/bash
# Submit 00_prep_job.sh as one CPU job (Vulcan). Log: $SCRATCH/review_20261020/brats_ssa/logs/prep.log
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
source scripts/job_runner/run_job.sh
mkdir -p "${SCRATCH}/review_20261020/brats_ssa/logs"
run_job --name brats_ssa_prep --gpus 0 --cpus 8 --mem 32G --time 03:00:00 \
  --log "${SCRATCH}/review_20261020/brats_ssa/logs/prep.log" -- bash scripts/cluster/brats_ssa_pa_alone/00_prep_job.sh
