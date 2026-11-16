#!/bin/bash
# T10 (registration research/dg_mechanisms/plans/2026-11-16-PREDICTION_T10_T28.md, section T10, sha256 e761e43e...,
# tag reg-T10-T28-20261116): what the specialist gap (P15) is made of. CPU only, through run_job (no GPU, no partition).
#   bash run_t10.sh smoke   -> 2 cases per test item into T10/smoke/ (code-path check, no scoring decision)
#   bash run_t10.sh run     -> all cases into T10/ (untracked), then gates + scoring; job id appended to submitted_jobs.txt
#   bash run_t10.sh specqc  -> repredict-indomain acceptance rule (qc_indomain.py, unchanged) on the FIRST-PASS specialists
#                              (Glioma T1w / T2w plain) into T10/specialist_qc/ (gate G1)
#   bash run_t10.sh score [smoke] -> scoring only (t10_score.py) on T10/ (or T10/smoke/)
set -euo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
HERE=${ROOT}/benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t10_specialist_gap
OUT=${ROOT}/benchmark/01_commun_results/dg_mechanisms/T10
SCR=/scratch/paulh/dg_mechanisms/T10
export RUN_JOB_ACCOUNT=aip-jcohen
cd "${ROOT}"; source scripts/job_runner/run_job.sh
mkdir -p "${OUT}/logs" "${SCR}"
PY=${ROOT}/.venv/bin/python
THR="export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; echo HEAD \$(git -C ${ROOT} rev-parse HEAD) host \$(hostname) \$(date -Is)"
case "${1:?smoke|run}" in
  smoke)
    D=${OUT}/smoke
    j=$(run_job --name t10_smoke --gpus 0 --cpus 8 --mem 48G --time 01:00:00 --log "${OUT}/logs/smoke.log" -- \
          bash -c "${THR}; ${PY} ${HERE}/t10_cases.py --out ${D} --scratch ${SCR}/smoke --workers 8 --max-cases 2; ${PY} ${HERE}/t10_score.py --dir ${D} --scratch ${SCR}/smoke" | awk '{print $NF}')
    echo "T10_SMOKE ${j} $(date -Is)" | tee -a "${OUT}/submitted_jobs.txt" ;;
  run)
    j=$(run_job --name t10_run --gpus 0 --cpus 8 --mem 40G --time 01:30:00 --log "${OUT}/logs/t10.log" -- \
          bash -c "${THR}; ${PY} ${HERE}/t10_cases.py --out ${OUT} --scratch ${SCR}/run --workers 8; ${PY} ${HERE}/t10_score.py --dir ${OUT} --scratch ${SCR}/run --spec-qc ${OUT}/specialist_qc/qc_indomain.tsv" | awk '{print $NF}')
    echo "T10_JOB ${j} $(date -Is)" | tee -a "${OUT}/submitted_jobs.txt" ;;
  specqc)
    j=$(run_job --name t10_specqc --gpus 0 --cpus 8 --mem 32G --time 01:00:00 --log "${OUT}/logs/specqc.log" -- \
          bash -c "${THR}; ${PY} ${HERE}/../repredict_indomain/qc_indomain.py --stage /scratch/paulh/dg_mechanisms/indomain/staged_predictions --out ${OUT}/specialist_qc --pairs Glioma_t1n:baseline,Glioma_t2w:baseline --workers 8" | awk '{print $NF}')
    echo "T10_SPECQC ${j} $(date -Is)" | tee -a "${OUT}/submitted_jobs.txt" ;;
  score)
    D=${OUT}; SD=${SCR}/run; L=score; [ "${2:-}" = smoke ] && { D=${OUT}/smoke; SD=${SCR}/smoke; L=score_smoke; }
    j=$(run_job --name t10_score --gpus 0 --cpus 2 --mem 16G --time 00:30:00 --log "${OUT}/logs/${L}.log" -- \
          bash -c "${THR}; ${PY} ${HERE}/t10_score.py --dir ${D} --scratch ${SD} --spec-qc ${OUT}/specialist_qc/qc_indomain.tsv" | awk '{print $NF}')
    echo "T10_SCORE ${j} $(date -Is) dir ${D}" | tee -a "${OUT}/submitted_jobs.txt" ;;
  *) echo "unknown"; exit 2 ;;
esac
