#!/bin/bash
# T08 (registration 2026-11-14-PREDICTION_T08.md, sha256 023b9d1c...): mask agreement across contrasts. CPU only via run_job.
#   bash t08_job.sh [smoke|run]      outputs: benchmark/01_commun_results/dg_mechanisms/T08/ (untracked), scratch /scratch/paulh/dg_mechanisms/T08/
set -euo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
HERE=${ROOT}/benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t08_mask_agreement
OUT=${ROOT}/benchmark/01_commun_results/dg_mechanisms/T08
SCR=/scratch/paulh/dg_mechanisms/T08
export RUN_JOB_ACCOUNT=aip-jcohen
cd "${ROOT}"; source scripts/job_runner/run_job.sh
mkdir -p "${OUT}/logs" "${SCR}"
WHAT="${1:-run}"
if [ "${WHAT}" = smoke ]; then EXTRA="--max-subjects 3"; O="${SCR}/smoke"; else EXTRA=""; O="${OUT}"; fi
THR="export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
j=$(run_job --name "t08_${WHAT}" --gpus 0 --cpus 2 --mem 16G --time 00:40:00 --log "${OUT}/logs/t08_${WHAT}.log" -- \
      bash -c "${THR}; ${ROOT}/.venv/bin/python ${HERE}/t08_agreement.py --root ${ROOT} --out ${O} ${EXTRA}" | awk '{print $NF}')
echo "${WHAT} ${j}" | tee -a "${OUT}/submitted_jobs.txt"
