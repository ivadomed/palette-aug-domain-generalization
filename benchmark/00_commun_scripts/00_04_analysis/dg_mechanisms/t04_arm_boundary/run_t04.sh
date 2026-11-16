#!/bin/bash
# T04 (registration 2026-11-12-PREDICTION_T01-T05.md, sha256 0b9f8710...): signed boundary shift of every arm vs GT.
# CPU only, through run_job (no GPU, no partition).
#   bash run_t04.sh smoke                 MS_flair, 2 cases per item (gate check of the code path)
#   bash run_t04.sh run [S1,S2,...]       one job per setting, then the analysis job (afterany: it scores what exists)
#   bash run_t04.sh analyze               analysis only
#   bash run_t04.sh dev12                 deviation 12 (plans/2026-11-14-T04-repredict-deviation.md): the 4 re-predicted
#                                         settings into T04/dev12/ (+ copies, never moves, of the scored rows/qc/prov), then
#                                         t04_qc_dev12.py and t04_analyze.py --qc-dev12 on T04/dev12 (scored 12/16 untouched)
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
WHAT="${1:?smoke|run|analyze|dev12}"
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
  dev12)
    D12="${OUT}/dev12"; mkdir -p "${D12}/rows" "${D12}/qc" "${D12}/prov" "${D12}/logs"
    for sub in rows qc; do for f in "${OUT}/${sub}"/*.parquet; do
      case "${f}" in *.smoke.parquet) continue ;; esac; [ -e "${D12}/${sub}/${f##*/}" ] || cp -p "${f}" "${D12}/${sub}/"; done; done
    for f in "${OUT}"/prov/*.tsv; do case "${f}" in *.smoke.tsv) continue ;; esac; [ -e "${D12}/prov/${f##*/}" ] || cp -p "${f}" "${D12}/prov/"; done
    DEV_NEW="Glioma_t1c Glioma_t2f Pelvis_ct Pelvis_mri"; DEV_ALL="Glioma_t1c,Glioma_t2f,Pelvis_ct,Pelvis_mri,Glioma_t1n,Glioma_t2w"
    ids=""
    for S in ${DEV_NEW}; do
      j=$(run_job --name "t04d12_${S}" --gpus 0 --cpus 8 --mem 32G --time 03:00:00 --log "${D12}/logs/t04_${S}.log" -- \
            bash -c "${THR}; ${PY} ${HERE}/t04_signed.py --setting ${S} --out ${D12} --workers 8" | awk '{print $NF}')
      ids="${ids}${ids:+,}${j}"; sleep 2
    done
    q=$(RUN_JOB_DEPENDENCY="afterany:${ids//,/:}" run_job --name t04d12_qc --gpus 0 --cpus 8 --mem 32G --time 01:00:00 \
          --log "${D12}/logs/qc_dev12.log" -- bash -c "${THR}; ${PY} ${HERE}/t04_qc_dev12.py --dir ${D12} --settings ${DEV_ALL} --workers 8" | awk '{print $NF}')
    a=$(RUN_JOB_DEPENDENCY="afterok:${q}" run_job --name t04d12_analyze --gpus 0 --cpus 2 --mem 32G --time 01:00:00 \
          --log "${D12}/logs/analyze.log" -- bash -c "${THR}; ${PY} ${HERE}/t04_analyze.py --dir ${D12} --qc-dev12 ${D12}/qc_dev12.tsv" | awk '{print $NF}')
    echo "DEV12_SETTING_JOBS ${ids} QC_JOB ${q} ANALYZE_JOB ${a}" | tee -a "${OUT}/submitted_jobs.txt" ;;
  *) echo "unknown: ${WHAT}"; exit 2 ;;
esac
