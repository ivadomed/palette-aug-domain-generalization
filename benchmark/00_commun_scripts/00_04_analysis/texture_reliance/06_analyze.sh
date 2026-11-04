#!/bin/bash
# Step 6: statistics + report -> ${WORK}/analysis/REPORT.md, domains.tsv (scratch only). One CPU job.
#   [RUN_JOB_DEPENDENCY=afterany:...] bash 06_analyze.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
submit --name trel_analyze --gpus 0 --cpus 2 --mem 16G --time 00:30:00 --log "${WORK}/logs/analyze.log" -- \
  "${PY}" "${HERE}/analyze.py" --work "${WORK}"
