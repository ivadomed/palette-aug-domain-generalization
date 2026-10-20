#!/bin/bash
# Queue one run_family.sh controller (CPU job) per training family. Usage: bash launch.sh [t1n t2w t2f t1c]
# Waits for nothing itself: each controller submits its own 1-GPU fold jobs and evaluates inline.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source scripts/job_runner/run_job.sh
L="${SCRATCH}/review_20261020/brats_ssa/logs"; mkdir -p "${L}"
for f in "${@:-t1n t2w t2f t1c}"; do
  for ff in ${f}; do
    run_job --name "bssa_ctl_${ff}" --gpus 0 --cpus 8 --mem 48G --time "${BSSA_CTL_TIME:-24:00:00}" \
      --log "${L}/${ff}.controller.log" -- bash scripts/cluster/brats_ssa_pa_alone/run_family.sh "${ff}"
    sleep 3
  done
done
