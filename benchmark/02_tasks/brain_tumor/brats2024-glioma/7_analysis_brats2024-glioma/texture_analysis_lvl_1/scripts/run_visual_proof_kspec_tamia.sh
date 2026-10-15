# TamIA: v19 Part B channel-selective first-stage kernel smoothing (FINDINGS 46), BraTS cells.
# MODE=spectra (Part A fallback, CPU; needs the 135 roster checkpoints rsynced into the mirror) | smoke | run [NCH] [TIME] | report | figure ; one whole H100 node per GPU job, one process (one cell) per GPU.
# Code and outputs live in the scratch mirror (TamIA project space is at its file quota): sync the script + PREREG there from Vulcan first.
MODE=${1:?spectra|smoke|run|report|figure}
cd /project/aip-jcohen/paulh/mri_synthesis_project
source benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh
source scripts/cluster/tamia_env.sh
export RUN_JOB_CPUS_PER_GPU=12 RUN_JOB_MEM_PER_GPU=115G
A=benchmark/02_tasks/brain_tumor/brats2024-glioma/7_analysis_brats2024-glioma/texture_analysis_lvl_1; T=benchmark/02_tasks
M=/scratch/p/paulh/vp_mirror; LOGS=/scratch/p/paulh/vp_logs; mkdir -p $LOGS
XPY=$(ls -d ${PROJECT_ROOT}/${T}/*/*/5_scripts_* | tr '\n' ':')
O="${M}/${A}/outputs/plots/kspec"; OS="/scratch/p/paulh/toDelete/kspec_smoke"
ENVS="export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'; export nnUNet_compile=false; export PYTHONPATH='${XPY}${PROJECT_ROOT}/${T}/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma'; export VP8_OUT='${M}/${A}/outputs/plots/visual_proof_erf_tasks'; export VP_ERFO_OUT='${M}/${A}/outputs/plots/erf_outcome'; export VP_SWITCH_OUT='${M}/${A}/outputs/plots/switch'; cd '${PROJECT_ROOT}'"
PYF="${M}/${A}/scripts/visual_proof_kspec.py"; PY=".venv/bin/python ${PYF}"
CELLS="glioma_t2w_t1n glioma_t2w_t2f glioma_t1n_t2f glioma_t1n_t2w"
case $MODE in
  spectra) run_job --name ks_spectra --gpus 0 --cpus 8 --mem 32G --time 01:30:00 --wait --log $LOGS/ks_spectra.log -- bash -c "${ENVS}; export VP_KSPEC_OUT=${O} OMP_NUM_THREADS=8; ${PY} --spectra" ;;
  smoke)  # compile; glioma_t1n_t2f, cases 0-1 (blur check included), fold 0, 2 draws, into the smoke dir; report; figure
    run_job --name ks_smoke --gpus 4 --time 01:00:00 --wait --log $LOGS/ks_smoke.log -- bash -c "${ENVS}; .venv/bin/python -m py_compile ${PYF} && echo COMPILE_OK && export VP_KSPEC_OUT=${OS} KS_FOLDS=0 KS_DRAWS=2 && mkdir -p ${OS}/_A; cp ${O}/kspec_A_layers.csv ${OS}/ 2>/dev/null; cp ${O}/_A/* ${OS}/_A/ 2>/dev/null; ${PY} --smoke --cells glioma_t1n_t2f --case-slice 0:2 && ${PY} --report && ${PY} --figure" ;;
  run)    # all 70 cases in NCH chunks, one node per chunk, 4 cells = 4 GPUs per node
    NCH=${2:-2}; TL=${3:-08:00:00}; step=$(( (70 + NCH - 1) / NCH ))
    for ((i=0; i<70; i+=step)); do j=$(( i + step < 70 ? i + step : 70 )); sl="${i}:${j}"; tag="${i}-${j}"
      body="export VP_KSPEC_OUT=${O}; .venv/bin/python -m py_compile ${PYF} || exit 3;"; g=0
      for c in $CELLS; do body="${body} (CUDA_VISIBLE_DEVICES=${g} ${PY} --cells ${c} --case-slice ${sl} > ${LOGS}/ks_run_${c}_${tag}.log 2>&1) &"; g=$((g+1)); done
      run_job --name ks_run_${tag} --gpus 4 --time ${TL} --log $LOGS/ks_run_${tag}.log -- bash -c "${ENVS}; ${body} wait; echo RUN_ROUND_DONE"; sleep 2; done ;;
  report) run_job --name ks_report --gpus 0 --cpus 4 --mem 32G --time 00:30:00 --wait --log $LOGS/ks_report.log -- bash -c "${ENVS}; export VP_KSPEC_OUT=${O}; ${PY} --report" ;;
  figure) run_job --name ks_figure --gpus 0 --cpus 4 --mem 32G --time 00:30:00 --wait --log $LOGS/ks_figure.log -- bash -c "${ENVS}; export VP_KSPEC_OUT=${O}; ${PY} --figure" ;;
esac
sleep 3; squeue -u paulh -h -o "%i %j %T %M %r"
