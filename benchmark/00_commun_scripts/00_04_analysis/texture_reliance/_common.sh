#!/bin/bash
# Shared settings of the texture-reliance scripts (sourced). Vulcan only: data, checkpoints and the runs live there.
set -euo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
HERE="${ROOT}/benchmark/00_commun_scripts/00_04_analysis/texture_reliance"
WORK="${TR_WORK:-/scratch/paulh/texture_reliance}"
PY="${ROOT}/.venv/bin/python"
mkdir -p "${WORK}/logs"
cd "${ROOT}"
source scripts/job_runner/run_job.sh
export RUN_JOB_ACCOUNT="${RUN_JOB_ACCOUNT:-aip-jcohen}"
# submit <run_job args...>: run_job, echo the Slurm job id only (spaced out: no burst of sbatch calls)
submit() { local out; out="$(run_job "$@" </dev/null)"; echo "${out}" >&2; sleep 2; echo "${out}" | grep -o '[0-9]\+$' | tail -1; }
