#!/bin/bash
# T11 (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T11_T13_T27.md, sha256 e435ef58...): SynthSeg-noEM
# in-domain errors (empty predictions vs far false positives) and CF14. CPU only, through run_job (no GPU, no partition).
#   bash run_t11.sh smoke   -> MS_flair + Pancreas_t1wce into T11/smoke/ (code-path check)
#   bash run_t11.sh run     -> all settings into T11/ (untracked), then scoring; job id appended to submitted_jobs.txt
set -euo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
HERE=${ROOT}/benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t11_noem_errors
OUT=${ROOT}/benchmark/01_commun_results/dg_mechanisms/T11
TEX=${ROOT}/paper/cvpr_format_latex/sec/_suppl_ind_ood.tex
export RUN_JOB_ACCOUNT=aip-jcohen
cd "${ROOT}"; source scripts/job_runner/run_job.sh
mkdir -p "${OUT}/logs"
PY=${ROOT}/.venv/bin/python
THR="export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; echo HEAD \$(git -C ${ROOT} rev-parse HEAD) host \$(hostname) \$(date -Is)"
case "${1:?smoke|run}" in
  smoke)
    D=${OUT}/smoke
    j=$(run_job --name t11_smoke --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --log "${OUT}/logs/smoke.log" -- \
          bash -c "${THR}; ${PY} ${HERE}/t11_cases.py --out ${D} --workers 4 --settings MS_flair,Pancreas_t1wce; ${PY} ${HERE}/t11_score.py --dir ${D} --tex ${TEX}" | awk '{print $NF}')
    echo "T11_SMOKE ${j} $(date -Is)" | tee -a "${OUT}/submitted_jobs.txt" ;;
  run)
    j=$(run_job --name t11_noem --gpus 0 --cpus 8 --mem 64G --time 04:00:00 --log "${OUT}/logs/t11.log" -- \
          bash -c "${THR}; ${PY} ${HERE}/t11_cases.py --out ${OUT} --workers 8; ${PY} ${HERE}/t11_score.py --dir ${OUT} --tex ${TEX}" | awk '{print $NF}')
    echo "T11_JOB ${j} $(date -Is)" | tee -a "${OUT}/submitted_jobs.txt" ;;
  *) echo "unknown"; exit 2 ;;
esac
