#!/bin/bash
# Step 1 of the "texture removed at test time" experiment (tasks.yaml): check the ladders / runs / original metrics and
# prepare the new test item (symlinks <input>_texrm, gen_jobs.tsv, runs.tsv) through ONE run_job CPU job on Vulcan.
#   bash 01_prepare.sh [--check-only]
set -euo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
HERE="${ROOT}/benchmark/00_commun_scripts/00_04_analysis/texture_removal"
OUT=/scratch/paulh/texture_removal_test; mkdir -p "${OUT}/logs"
cd "${ROOT}"
source scripts/job_runner/run_job.sh
export RUN_JOB_ACCOUNT="${RUN_JOB_ACCOUNT:-aip-jcohen}"
# the venv trainer shims import each dataset's trainers package (prepare.py resolves every trainer)
PP="$(ls -d "${ROOT}"/benchmark/02_tasks/*/*/5_scripts_* | tr '\n' ':')"
run_job --name texrm_prepare --gpus 0 --cpus 4 --mem 24G --time 01:00:00 --log "${OUT}/logs/prepare.log" --wait -- \
  bash -c "export NNUNET_PROJECT_ROOT='${ROOT}' PYTHONPATH='${PP}'; cd '${ROOT}'; .venv/bin/python '${HERE}/prepare.py' $*"
tail -40 "${OUT}/logs/prepare.log"
