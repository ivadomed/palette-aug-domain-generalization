#!/usr/bin/env bash
# Submit one Vulcan L40S job per NGF registry key (19 keys = the 9 tasks of the paper's NGF table) for the PA round-2 variants.
# Output: $SCRATCH-side dir (default /scratch/paulh/review_20261017/ngf_variants), never the paper's results dir.
# usage: bash scripts/cluster/ngf_variants/run_all.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
OUT="${NGF_VAR_OUT:-/scratch/paulh/review_20261017/ngf_variants}"
source "${REPO}/benchmark/02_tasks/brain_ms/open-ms/5_scripts_open-ms/00_utils/env.sh"
PY="${REPO}/.venv/bin/python"
export RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-rack02-06}"
mkdir -p "${OUT}/data" "${OUT}/logs"
KEYS=(brats2024-glioma/t1n brats2024-glioma/t2w brats2024-glioma/t2f brats2024-glioma/t1c open-ms/flair open-ms/t1w ispy2/t1wce ispy2/t2w
      chaos/t1in chaos/t2spir on-harmony/t1w on-harmony/t2w on-harmony/dwi toothfairy2/cbct totalseg-pelvic/ct totalseg-pelvic/mri
      atlas-liver-hcc/t1w pansegdata/t1wce pansegdata/t2w)
for k in "${KEYS[@]}"; do
  tag="${k//\//__}"; MAXV=12000000; [ "$k" = "totalseg-pelvic/ct" ] && MAXV=8000000   # as run_all.sh of the paper's NGF
  run_job --name "ngfv_${tag}" --gpus 1 --time 01:00:00 --log "${OUT}/logs/${tag}.log" -- \
    bash -c "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ${PY} ${HERE}/ngf_variants.py --key ${k} --n-scans 20 --n-draws 5 --variants noblur --max-vox ${MAXV} --out ${OUT}/data/ngfv_${tag}.csv"
  sleep 3
done
