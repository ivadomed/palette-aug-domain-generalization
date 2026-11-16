#!/bin/bash
# T27 (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T11_T13_T27.md, sha256 e435ef58...): shared inward
# offset vs voxel size. CPU only, through run_job (no GPU, no partition). Reads the T04 row parquets (read-only), image
# headers and GT label files; one job: extraction then scoring.
#   bash run_t27.sh     -> benchmark/01_commun_results/dg_mechanisms/T27/ (untracked); job id appended to submitted_jobs.txt
set -euo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
HERE=${ROOT}/benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t27_offset_resolution
OUT=${ROOT}/benchmark/01_commun_results/dg_mechanisms/T27
export RUN_JOB_ACCOUNT=aip-jcohen
cd "${ROOT}"; source scripts/job_runner/run_job.sh
mkdir -p "${OUT}/logs"
PY=${ROOT}/.venv/bin/python
CMD="export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; echo HEAD \$(git -C ${ROOT} rev-parse HEAD) host \$(hostname) \$(date -Is); \
${PY} ${HERE}/t27_offsets.py --out ${OUT} --workers 8 && ${PY} ${HERE}/t27_score.py --items ${OUT}/items.tsv --out ${OUT}"
j=$(run_job --name t27_offset --gpus 0 --cpus 8 --mem 32G --time 01:30:00 --log "${OUT}/logs/t27.log" -- bash -c "${CMD}" | awk '{print $NF}')
echo "T27_JOB ${j} $(date -Is)" | tee -a "${OUT}/submitted_jobs.txt"
