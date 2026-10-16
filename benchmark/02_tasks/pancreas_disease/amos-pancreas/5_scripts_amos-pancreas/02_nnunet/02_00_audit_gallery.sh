#!/usr/bin/env bash
# CPU job: audit + PNG gallery of the AMOS-MRI cases (pancreas label 10). Never python on the login node.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; DS="$(cd "$HERE/../.." && pwd)"; PROJECT_ROOT="$(cd "$DS/../../../.." && pwd)"
LEAF="$PROJECT_ROOT/benchmark/02_tasks/abdomen_healthy/amos/1_BIDS_amos/abdomen-amos"
source "$PROJECT_ROOT/scripts/job_runner/run_job.sh"
run_job --name amospan_audit --cpus 2 --mem 8G --time 00:40:00 --log "$DS/0_raw_amos-pancreas/audit_job.log" --wait -- \
  "$PROJECT_ROOT/.venv/bin/python" -I "$HERE/02_00_audit_gallery.py" "$LEAF" "$DS/0_raw_amos-pancreas/audit"
