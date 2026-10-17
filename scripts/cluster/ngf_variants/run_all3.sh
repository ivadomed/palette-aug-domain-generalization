#!/usr/bin/env bash
# Round 3: per key one Vulcan L40S job running the 5 arms (ctrl + 4 weakening arms) one process each, identical per-draw seeds.
# usage: bash scripts/cluster/ngf_variants/run_all3.sh   (NGF_VAR_KEYS="k1 k2" for a subset, RUN_JOB_EXCLUDE_NODES to skip nodes)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../../.." && pwd)"
OUT="${NGF_VAR_OUT:-/scratch/paulh/review_20261017/ngf_variants/round3}"
source "${REPO}/benchmark/02_tasks/brain_ms/open-ms/5_scripts_open-ms/00_utils/env.sh"
PY="${REPO}/.venv/bin/python"
export RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-rack02-06,rack13-08}"
mkdir -p "${OUT}/data" "${OUT}/logs"
KEYS=(brats2024-glioma/t1n brats2024-glioma/t2w brats2024-glioma/t2f brats2024-glioma/t1c open-ms/flair open-ms/t1w ispy2/t1wce ispy2/t2w
      chaos/t1in chaos/t2spir on-harmony/t1w on-harmony/t2w on-harmony/dwi toothfairy2/cbct totalseg-pelvic/ct totalseg-pelvic/mri
      atlas-liver-hcc/t1w pansegdata/t1wce pansegdata/t2w)
[ -n "${NGF_VAR_KEYS:-}" ] && read -ra KEYS <<< "${NGF_VAR_KEYS}"
for k in "${KEYS[@]}"; do
  tag="${k//\//__}"; MAXV=12000000; [ "$k" = "totalseg-pelvic/ct" ] && MAXV=8000000
  run_job --name "ngf3_${tag}" --gpus 1 --time 02:00:00 --log "${OUT}/logs/${tag}.log" -- \
    bash -c "export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; for a in ${NGF_ARMS:-ctrl unsharp_off redist_off edgehalf texhalf}; do ${PY} ${HERE}/ngf_variants3.py \$a --key ${k} --n-scans 20 --n-draws 5 --variants noblur --max-vox ${MAXV} --out ${OUT}/data/ngf3_${tag}__\$a.csv || exit 1; done"
  sleep 2
done
