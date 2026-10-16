#!/bin/bash
# Submit the read-only ladder audit (benchmark/00_commun_scripts/00_03_evaluate/audit_ladders.py) as a small CPU job.
# Usage: bash scripts/evaluate/run_audit_ladders.sh [LOG]   (default log: $SCRATCH/ladder_audit/audit_<date>.log)
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/job_runner/run_job.sh
LOG="${1:-${SCRATCH}/ladder_audit/audit_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "${LOG}")"
run_job --name audit_ladders --gpus 0 --cpus 1 --mem 8G --time 00:30:00 --log "${LOG}" -- \
    bash -c "cd '${PWD}' && .venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/audit_ladders.py"
echo "audit log: ${LOG}"
