#!/bin/bash
# In-domain re-prediction chain (plans/2026-11-16-repredict-indomain.md). CPU steps run through run_job --gpus 0.
#   bash run_indomain.sh smoke          CPU job: QC of the pairs needing no GPU (Mandible cbct noEM TamIA copy, Pelvis mri palette)
#   bash run_indomain.sh post <deps>    CPU job after <deps> (afterany:id:id...): audit -> QC (all pairs) -> T11 completion
#   bash run_indomain.sh post_body      (the body of the post job; not called by hand)
set -euo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
HERE=${ROOT}/benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/repredict_indomain
OUT=${ROOT}/benchmark/01_commun_results/dg_mechanisms/repredict_indomain
T11OUT=${ROOT}/benchmark/01_commun_results/dg_mechanisms/T11
STAGE=/scratch/paulh/dg_mechanisms/indomain/staged_predictions
TEX=${ROOT}/paper/cvpr_format_latex/sec/_suppl_ind_ood.tex
PY=${ROOT}/.venv/bin/python
PAIRS_ALL=Glioma_t1c:synthseg_noEM,Glioma_t1c:baseline,Glioma_t2f:synthseg_noEM,Glioma_t2f:baseline,Pelvis_ct:synthseg_noEM,Pelvis_mri:synthseg_noEM,Pelvis_mri:texture,Mandible_cbct:synthseg_noEM
PAIRS_SMOKE=Mandible_cbct:synthseg_noEM,Pelvis_mri:texture
export RUN_JOB_ACCOUNT=aip-jcohen
cd "${ROOT}"
mkdir -p "${OUT}/logs"
THR="export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1; echo HEAD \$(git -C ${ROOT} rev-parse HEAD) host \$(hostname) \$(date -Is)"
case "${1:?smoke|post|post_body}" in
  smoke)
    source scripts/job_runner/run_job.sh
    j=$(run_job --name ind_qc_smoke --gpus 0 --cpus 8 --mem 48G --time 01:00:00 --log "${OUT}/logs/smoke.log" -- \
          bash -c "${THR}; ${PY} ${HERE}/qc_indomain.py --stage ${STAGE} --out ${OUT}/smoke --pairs ${PAIRS_SMOKE} --workers 8" | awk '{print $NF}')
    echo "IND_QC_SMOKE ${j} $(date -Is)" | tee -a "${OUT}/submitted_jobs.txt" ;;
  post)
    source scripts/job_runner/run_job.sh
    j=$(RUN_JOB_DEPENDENCY="${2:?afterany:...}" run_job --name ind_post --gpus 0 --cpus 8 --mem 64G --time 03:00:00 \
          --log "${OUT}/logs/post.log" -- bash "${HERE}/run_indomain.sh" post_body | awk '{print $NF}')
    echo "IND_POST ${j} dep ${2} $(date -Is)" | tee -a "${OUT}/submitted_jobs.txt" ;;
  post_body)
    eval "${THR}"
    bash "${HERE}/repredict_indomain.sh" audit | tee "${OUT}/audit.tsv"
    rc=0
    ${PY} "${HERE}/qc_indomain.py" --stage "${STAGE}" --out "${OUT}" --pairs "${PAIRS_ALL}" --workers 8 || rc=$?
    echo "[post] qc_indomain exit ${rc}"
    [ -s "${OUT}/qc_indomain.tsv" ] || { echo "[post] no QC table; stopping"; exit 3; }
    ${PY} "${HERE}/t11_completion.py" --qc "${OUT}/qc_indomain.tsv" --stage "${STAGE}" --first "${T11OUT}" \
        --out "${T11OUT}/completion" --tex "${TEX}" --workers 8
    echo "[post] done $(date -Is)" ;;
  *) echo "unknown"; exit 2 ;;
esac
