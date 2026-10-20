#!/bin/bash
# Submit analysis.py as a CPU job (Vulcan). Usage: bash run_analysis.sh [tag-substring ...]
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
source scripts/job_runner/run_job.sh
L="${SCRATCH}/review_20261020/brats_ssa/logs"; mkdir -p "${L}"
run_job --name bssa_analysis --gpus 0 --cpus 4 --mem 32G --time 03:00:00 \
  --log "${L}/analysis${1:+_$1}.log" -- bash -c "cd /project/aip-jcohen/paulh/mri_synthesis_project && module load python/3.11 >/dev/null 2>&1 || true; export PROJECT_ROOT=\$PWD; .venv/bin/python scripts/cluster/brats_ssa_pa_alone/analysis.py $*"
