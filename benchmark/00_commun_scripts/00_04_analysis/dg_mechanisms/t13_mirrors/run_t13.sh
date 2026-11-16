#!/bin/bash
# T13 (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T11_T13_T27.md, sha256 e435ef58...): checkpoint mirrors
# of Palette+AugLab. CPU only, through run_job (no GPU, no partition). One job: per-fold extraction, then scoring.
#   bash run_t13.sh            -> benchmark/01_commun_results/dg_mechanisms/T13/ (untracked); job id appended to submitted_jobs.txt
set -euo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
HERE=${ROOT}/benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t13_mirrors
OUT=${ROOT}/benchmark/01_commun_results/dg_mechanisms/T13
SCR=/scratch/paulh/dg_mechanisms/T13/indood
T01J=${ROOT}/benchmark/01_commun_results/dg_mechanisms/T01/full_refresh/indood_folds.json
TEX=${ROOT}/paper/cvpr_format_latex/sec/_suppl_ind_ood.tex
export RUN_JOB_ACCOUNT=aip-jcohen
cd "${ROOT}"; source scripts/job_runner/run_job.sh
mkdir -p "${OUT}/logs" "${SCR}"
PY=${ROOT}/.venv/bin/python
CMD="export OMP_NUM_THREADS=2 MPLBACKEND=Agg SCRATCH=/scratch/paulh PROJECT_ROOT=${ROOT}; echo HEAD \$(git -C ${ROOT} rev-parse HEAD) host \$(hostname) \$(date -Is); \
${PY} ${HERE}/t13_fold_values.py --out ${OUT} --scratch ${SCR} && \
${PY} ${HERE}/t13_score.py --folds ${OUT}/t13_folds.json --tex ${TEX} --t01 ${T01J} --out ${OUT}"
j=$(run_job --name t13_mirrors --gpus 0 --cpus 2 --mem 16G --time 01:30:00 --log "${OUT}/logs/t13.log" -- bash -c "${CMD}" | awk '{print $NF}')
echo "T13_JOB ${j} $(date -Is)" | tee -a "${OUT}/submitted_jobs.txt"
