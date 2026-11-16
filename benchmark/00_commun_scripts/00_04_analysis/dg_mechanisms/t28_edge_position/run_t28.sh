#!/bin/bash
# T28 (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T10_T28.md, section T28, sha256 e761e43e...,
# tag reg-T10-T28-20261116): edge position E (CF19) and ramp width W vs the T27 per-item offset O. CPU only, through
# run_job (no GPU, no partition). Reads T27/items_scored.tsv, the T04 row parquets (case lists), test images and GT labels,
# all read-only.
#   bash run_t28.sh smoke   -> one case per gate item, quick check (scratch output)
#   bash run_t28.sh gate    -> response gate (dilate / erode by one voxel on 3 items per task); exit 3 if it fails
#   bash run_t28.sh full    -> E and W on every T27 item, then the registered rules (t28_score.py)
# Outputs: benchmark/01_commun_results/dg_mechanisms/T28/ (untracked); job ids appended to submitted_jobs.txt
set -euo pipefail
MODE=${1:?usage: run_t28.sh smoke|gate|full}
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
HERE=${ROOT}/benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t28_edge_position
OUT=${ROOT}/benchmark/01_commun_results/dg_mechanisms/T28
SCR=/scratch/paulh/dg_mechanisms/T28
export RUN_JOB_ACCOUNT=aip-jcohen
cd "${ROOT}"; source scripts/job_runner/run_job.sh
mkdir -p "${OUT}/logs" "${SCR}"
PY=${ROOT}/.venv/bin/python
ENVS="export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; echo HEAD \$(git -C ${ROOT} rev-parse HEAD) host \$(hostname) \$(date -Is)"
case "${MODE}" in
  smoke) CMD="${ENVS}; ${PY} ${HERE}/t28_edge.py --mode smoke --limit 1 --workers 4 --out ${SCR}/smoke"
         j=$(run_job --name t28_smoke --gpus 0 --cpus 4 --mem 24G --time 00:30:00 --log "${OUT}/logs/t28_smoke.log" -- bash -c "${CMD}" | awk '{print $NF}');;
  gate)  CMD="${ENVS}; ${PY} ${HERE}/t28_edge.py --mode gate --workers 16 --out ${OUT}"
         j=$(run_job --name t28_gate --gpus 0 --cpus 16 --mem 64G --time 03:00:00 --log "${OUT}/logs/t28_gate.log" -- bash -c "${CMD}" | awk '{print $NF}');;
  full)  CMD="${ENVS}; ${PY} ${HERE}/t28_edge.py --mode full --workers 16 --out ${OUT} && ${PY} ${HERE}/t28_score.py --out ${OUT}"
         j=$(run_job --name t28_full --gpus 0 --cpus 16 --mem 64G --time 04:00:00 --log "${OUT}/logs/t28_full.log" -- bash -c "${CMD}" | awk '{print $NF}');;
  *) echo "unknown mode ${MODE}"; exit 2;;
esac
echo "T28_${MODE^^}_JOB ${j} $(date -Is)" | tee -a "${OUT}/submitted_jobs.txt"
