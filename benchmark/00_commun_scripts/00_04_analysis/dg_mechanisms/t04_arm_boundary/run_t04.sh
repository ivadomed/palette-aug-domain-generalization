#!/bin/bash
# T04 (registration 2026-11-12-PREDICTION_T01-T05.md, sha256 0b9f8710...): signed boundary shift of every arm vs GT.
# CPU only, through run_job (no GPU, no partition).
#   bash run_t04.sh smoke                 MS_flair, 2 cases per item (gate check of the code path)
#   bash run_t04.sh run [S1,S2,...]       one job per setting, then the analysis job (afterany: it scores what exists)
#   bash run_t04.sh analyze               analysis only
# Outputs: benchmark/01_commun_results/dg_mechanisms/T04/ (untracked). Job ids appended to T04/submitted_jobs.txt.
set -euo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
HERE=${ROOT}/benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t04_arm_boundary
OUT=${ROOT}/benchmark/01_commun_results/dg_mechanisms/T04
export RUN_JOB_ACCOUNT=aip-jcohen
cd "${ROOT}"; source scripts/job_runner/run_job.sh
mkdir -p "${OUT}/logs"
THR="export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
PY=${ROOT}/.venv/bin/python
WHAT="${1:?smoke|run|analyze}"
# Settings whose headline predictions are on Vulcan (the others: see REPORT.md, DEVIATIONS)
DEFAULT="MS_flair MS_t1w Glioma_t1n Glioma_t2w Abdomen_t1in Abdomen_t2spir Brain_T1w Brain_T2w Brain_dwi_ap Pancreas_t1wce Pancreas_t2w"
case "${WHAT}" in
  smoke)
    j=$(run_job --name t04_smoke --gpus 0 --cpus 4 --mem 16G --time 00:30:00 --log "${OUT}/logs/smoke_MS_flair.log" -- \
          bash -c "${THR}; ${PY} ${HERE}/t04_signed.py --setting MS_flair --out ${OUT} --workers 4 --max-cases 2" | awk '{print $NF}')
    echo "SMOKE ${j}" | tee -a "${OUT}/submitted_jobs.txt" ;;
  run)
    SETTINGS="${DEFAULT}"; if [ -n "${2:-}" ]; then SETTINGS="${2//,/ }"; fi
    ids=""
    for S in ${SETTINGS}; do
      case "${S}" in
        Pancreas_*) T=04:00:00; M=64G; W=8 ;;
        Abdomen_t2spir) T=03:00:00; M=48G; W=8 ;;
        *) T=03:00:00; M=32G; W=8 ;;
      esac
      j=$(run_job --name "t04_${S}" --gpus 0 --cpus "${W}" --mem "${M}" --time "${T}" --log "${OUT}/logs/t04_${S}.log" -- \
            bash -c "${THR}; ${PY} ${HERE}/t04_signed.py --setting ${S} --out ${OUT} --workers ${W}" | awk '{print $NF}')
      ids="${ids}${ids:+,}${j}"
      sleep 2
    done
    a=$(RUN_JOB_DEPENDENCY="afterany:${ids//,/:}" run_job --name t04_analyze --gpus 0 --cpus 2 --mem 32G --time 01:00:00 \
          --log "${OUT}/logs/analyze.log" -- bash -c "${THR}; ${PY} ${HERE}/t04_analyze.py --dir ${OUT}" | awk '{print $NF}')
    echo "SETTING_JOBS ${ids}" | tee -a "${OUT}/submitted_jobs.txt"
    echo "ANALYZE_JOB ${a}" | tee -a "${OUT}/submitted_jobs.txt" ;;
  analyze)
    a=$(run_job --name t04_analyze --gpus 0 --cpus 2 --mem 32G --time 01:00:00 --log "${OUT}/logs/analyze.log" -- \
          bash -c "${THR}; ${PY} ${HERE}/t04_analyze.py --dir ${OUT}" | awk '{print $NF}')
    echo "ANALYZE_JOB ${a}" | tee -a "${OUT}/submitted_jobs.txt" ;;
  *) echo "unknown: ${WHAT}"; exit 2 ;;
esac
