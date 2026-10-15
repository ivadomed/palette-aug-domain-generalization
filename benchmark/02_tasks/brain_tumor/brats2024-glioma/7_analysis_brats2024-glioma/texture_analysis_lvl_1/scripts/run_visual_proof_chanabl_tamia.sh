# TamIA: v18 texture-energy-selective channel ablation (FINDINGS 45), BraTS cells.
# MODE=smoke | fit | run [NCH] [TIME] (DEP=<fit job id> optional) | report | figure ; one whole H100 node per GPU job, one process (one cell) per GPU.
# Code and outputs live in the scratch mirror (TamIA project space is at its file quota): sync the script + PREREG there from Vulcan first.
MODE=${1:?smoke|fit|run|report|figure}
cd /project/aip-jcohen/paulh/mri_synthesis_project
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
source scripts/cluster/tamia_env.sh
export RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G
A=benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1; T=benchmark/02_tasks
M=/scratch/p/paulh/vp_mirror; LOGS=/scratch/p/paulh/vp_logs; mkdir -p $LOGS
XPY=$(ls -d ${PROJECT_ROOT}/${T}/*/*/5_scripts_* | tr '\n' ':')
O="${M}/${A}/outputs/plots/chanabl"; OS="/scratch/p/paulh/toDelete/chanabl_smoke"
ENVS="export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'; export nnUNet_compile=false; export PYTHONPATH='${XPY}${PROJECT_ROOT}/${T}/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma'; export VP8_OUT='${M}/${A}/outputs/plots/visual_proof_erf_tasks'; export VP_SWITCH_OUT='${M}/${A}/outputs/plots/switch'; export VP_TEXTHINK_OUT='${M}/${A}/outputs/plots/texthink'; cd '${PROJECT_ROOT}'"
PYF="${M}/${A}/scripts/visual_proof_chanabl.py"; PY=".venv/bin/python ${PYF}"
CELLS="glioma_t2w_t1n glioma_t2w_t2f glioma_t1n_t2f glioma_t1n_t2w"
case $MODE in
  smoke)  # compile; 3 fit + 2 eval cases of glioma_t1n_t2f, fold 0, 2 draws, into the smoke dir; report; figure
    run_job --name ca_smoke --gpus 4 --time 00:50:00 --wait --log $LOGS/ca_smoke.log -- bash -c "${ENVS}; .venv/bin/python -m py_compile ${PYF} && echo COMPILE_OK && export VP_CHANABL_OUT=${OS} CA_FOLDS=0 CA_NFIT=3 CA_DRAWS=2 && ${PY} --fit --cells glioma_t1n_t2f && ${PY} --cells glioma_t1n_t2f --case-slice 0:5 && ${PY} --report && ${PY} --figure" ;;
  fit)    # one node: 4 cells on 4 GPUs, selection on the first 20 cases
    body="export VP_CHANABL_OUT=${O}; .venv/bin/python -m py_compile ${PYF} || exit 3;"; g=0
    for c in $CELLS; do body="${body} (CUDA_VISIBLE_DEVICES=${g} ${PY} --fit --cells ${c} > ${LOGS}/ca_fit_${c}.log 2>&1) &"; g=$((g+1)); done
    run_job --name ca_fit --gpus 4 --time ${2:-01:00:00} --log $LOGS/ca_fit.log -- bash -c "${ENVS}; ${body} wait; grep -c CHANABL_FIT_DONE ${LOGS}/ca_fit_*.log; echo FIT_ROUND_DONE" ;;
  run)    # evaluation cases 20:70 in NCH chunks, one node per chunk, 4 cells = 4 GPUs per node; optional DEP = afterok job id of the fit
    NCH=${2:-3}; TL=${3:-02:00:00}; [ -n "${DEP:-}" ] && export RUN_JOB_DEPENDENCY=afterok:${DEP}; step=$(( (50 + NCH - 1) / NCH ))
    for ((i=20; i<70; i+=step)); do j=$(( i + step < 70 ? i + step : 70 )); sl="${i}:${j}"; tag="${i}-${j}"
      body="export VP_CHANABL_OUT=${O}; .venv/bin/python -m py_compile ${PYF} || exit 3;"; g=0
      for c in $CELLS; do body="${body} (CUDA_VISIBLE_DEVICES=${g} ${PY} --cells ${c} --case-slice ${sl} > ${LOGS}/ca_run_${c}_${tag}.log 2>&1) &"; g=$((g+1)); done
      run_job --name ca_run_${tag} --gpus 4 --time ${TL} --log $LOGS/ca_run_${tag}.log -- bash -c "${ENVS}; ${body} wait; echo RUN_ROUND_DONE"; sleep 2; done ;;
  report) run_job --name ca_report --gpus 0 --cpus 4 --mem 32G --time 00:30:00 --wait --log $LOGS/ca_report.log -- bash -c "${ENVS}; export VP_CHANABL_OUT=${O}; ${PY} --report" ;;
  figure) run_job --name ca_figure --gpus 0 --cpus 4 --mem 32G --time 00:30:00 --wait --log $LOGS/ca_figure.log -- bash -c "${ENVS}; export VP_CHANABL_OUT=${O}; ${PY} --figure" ;;
esac
sleep 3; squeue -u paulh -h -o "%i %j %T %M %r"
