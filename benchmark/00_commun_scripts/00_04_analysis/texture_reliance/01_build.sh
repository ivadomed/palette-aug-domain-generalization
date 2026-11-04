#!/bin/bash
# Step 1: manifest (domains, runs, OOD items, inputs, GT, labels; checkpoint trainer names; identical-copy check of
# the Abdomen t2spir crops) -> ${WORK}/manifest.json, gen_jobs.tsv, pred_jobs.tsv. One CPU job, blocking.
#   bash 01_build.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
run_job --name trel_build --gpus 0 --cpus 4 --mem 32G --time 01:00:00 --wait --log "${WORK}/logs/build.log" -- \
  "${PY}" "${HERE}/build_manifest.py" --work "${WORK}"
tail -3 "${WORK}/logs/build.log"
