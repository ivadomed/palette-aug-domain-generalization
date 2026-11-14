#!/bin/bash
# VULCAN post-training controller for ONE setting of the SLAug arm (slaug, METHOD auglab_slaug; clone of scripts/cluster/ginipa/post_run.sh): own-test predict + evaluate AND
# every external cohort the task's combined config reads, then the audit vs the headline PA+Auglab run (row count + distinct cases per
# item, per fold, in every source). Clone of scripts/cluster/flatauglab/vulcan_post_run.sh (same recipes, same evaluators, metrics FLAT
# next to the headline runs), with: the arm's own predict wrappers (pin on PYTHONPATH via pin_env.sh: nnU-Net imports the SLAug
# trainer class to build the network), the breast (duke / ispy1 / acrin6698) and pancreas (totalsegmri / msd-pancreas) companions run
# INLINE here instead of a separate poller (evaluated with flatauglab's vulcan_companion_eval_job.sh body + companion_audit.py, both
# method-agnostic), and the audit = scripts/cluster/flatauglab/audit_vs_headline.py --run-table <this arm's run table>.
# Model = the Vulcan run dir that post_poller.sh verified (<post dir>/<setting>/SYNCED: all 3 folds best + final, md5).
# Submitted (CPU job; predicts are L40S run_job --wait jobs) by post_poller.sh, or by hand:
#   sbatch --account=aip-jcohen --cpus-per-task=16 --mem=96G --time=3-00:00:00 --job-name=slaugpost_<setting> \
#     --output=<root>/post/<setting>/post_job_%j.log --wrap "bash scripts/cluster/slaug/post_run.sh <setting>"
# DRY=1: print the steps without running them (works before training ends). Status: <post dir>/<setting>/POST.status.
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/paulh}"
S_="${1:?usage: post_run.sh <setting>}"
R="${SLAUG_ROOT:-$(cat /scratch/paulh/_slaug_root.txt)}"; RUNS="${R}/run_table.tsv"; PD="${R}/post"; SD="${PD}/${S_}"; mkdir -p "${SD}"
if [ -z "${SLPOST_SNAPSHOT:-}" ] && [ -z "${DRY:-}" ]; then   # run a private snapshot: later edits never hit a RUNNING controller
  snap="${SD}/post_run.snapshot.sh"; cp "$0" "${snap}" && export SLPOST_SNAPSHOT=1 && exec bash "${snap}" "$@"
fi
ROOT="$PWD"; T=benchmark/02_tasks; METHOD=auglab_slaug; CAT=auglab
ST="${SD}/POST.status"; OUTD="${SD}/post_logs"; mkdir -p "${OUTD}"
export RUN_JOB_ACCOUNT=aip-jcohen RUN_JOB_GPU_TYPE=l40s
export RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-rack02-06}"
unset RUN_JOB_DEPENDENCY RUN_JOB_INLINE
log() { echo "[$(date '+%F %T')] $*" | tee -a "${ST}"; }
FAIL=0; fail() { log "ERROR: $*"; FAIL=1; }
log "START ${S_} host=$(hostname) job=${SLURM_JOB_ID:-none} HEAD=$(git rev-parse --short HEAD)${DRY:+ (DRY)}"
source scripts/cluster/slaug/pin_env.sh   # the Vulcan pin (refuses on mismatch); exported for the fov_crop driver below
RID="$(awk -F'\t' -v s="${S_}" 'NR>1 && $1==s {print $16}' "${RUNS}" | sort -u)"
[ "$(echo "${RID}" | grep -c .)" = 1 ] && [[ "${RID}" == *_${METHOD}_[0-9]* ]] || { log "FAILED: run id of ${S_} not unique / not the arm: '${RID}'"; exit 2; }
if [ -z "${DRY:-}" ]; then
  [ -f "${SD}/SYNCED" ] || { log "FAILED: ${SD}/SYNCED missing"; exit 2; }
  RDIR="$(head -1 "${SD}/SYNCED" | cut -d' ' -f1)"; [ "$(basename "${RDIR}")" = "${RID}" ] || { log "FAILED: SYNCED dir ${RDIR} is not ${RID}"; exit 2; }
  for k in 0 1 2; do for ck in checkpoint_best checkpoint_final; do
    ls "${RDIR}"/Dataset*/*/fold_${k}/${ck}.pth >/dev/null 2>&1 || { log "FAILED: missing ${ck} fold ${k} in ${RDIR}"; exit 2; }; done; done
  TR="$(basename "$(ls -d "${RDIR}"/Dataset*/*__nnUNetPlans__3d_fullres | head -1)" | sed 's/__nnUNetPlans__3d_fullres//')"
else TR="$(awk -F'\t' -v s="${S_}" 'NR>1 && $1==s {print $6; exit}' "${RUNS}")"; fi
log "run ${RID} trainer ${TR}"
run() { if [ -n "${DRY:-}" ]; then echo "[dry] $*" | tee -a "${ST}"; return 0; fi; "$@"; }
inline_eval() { ( export RUN_JOB_INLINE=1; run "$@" ) > "${OUTD}/eval_$(date +%H%M%S)_$RANDOM.log" 2>&1 || fail "eval: ${*: -1}"; }
ev() { inline_eval bash -c "$1"; }
predict() { local w; w="$(printf '%s\n' "$@" | grep -m1 '\.sh$')"; local tag; tag="$(basename "$(dirname "$(dirname "$w")")")_$(basename "$w" .sh)"
  log "predict ${tag}"; run "$@" > "${OUTD}/predict_${tag}.log" 2>&1 || fail "predict: $*"; }
panseg_comp() {   # $1 companion dir name, $2 items: the companion's roster shim with the explicit slaug RUN_ID (= companions_vulcan.sh recipe)
  local CN="$1" S="${T}/pancreas_disease/$1/5_scripts_$1" tc="$3"
  log "predict ${CN} (${2})"
  if [ -n "${DRY:-}" ]; then echo "[dry] ${CN}: roster shim 05_01_predict_pansegdata_common.sh ${RID} all (TRAINER ${TR})" | tee -a "${ST}"; return 0; fi
  ( set -euo pipefail; unset PREDICTIONS_ROOT METRICS_ROOT nnUNet_raw nnUNet_preprocessed RUN_JOB_INLINE
    source "${S}/00_utils/env.sh"; source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_00_utils/roster_lib.sh"
    rd="$(roster_src_run_dir PANSEG "${tc}" auglab "${RID}")"; roster_run_trainer_id "${rd}" || { echo "no trained model under ${rd}"; exit 1; }
    export PANSEG_TRAINING_CONTRAST="${tc}" PANSEG_DATASET_ID="${RUN_DATASET_ID}" PYTHONPATH="${SLAUG_PIN}:${PYTHONPATH:-}"
    METHOD="${METHOD}"; TRAINER="${RUN_TRAINER}"; CATEGORY=auglab
    source "${S}/05_predict/05_01_predict_pansegdata_common.sh" "${RID}" all ) > "${OUTD}/predict_${CN}.log" 2>&1 || { fail "predict ${CN}"; return; }
  ev "bash scripts/cluster/flatauglab/vulcan_companion_eval_job.sh ${CN} ${RID} ${tc} $2"
}
breast_comp() {   # $1 companion, $2 wrapper, $3 items
  predict bash "${T}/breast_cancer/$1/5_scripts_$1/05_predict/$2" "${RID}" all $3
  ev "bash scripts/cluster/flatauglab/vulcan_companion_eval_job.sh $1 ${RID} ${tc} $3"
}
ONLY=""

case "${S_}" in
pelvic_ct|pelvic_mri)
  P="${T}/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic"; tc="${S_#pelvic_}"
  case "$tc" in ct) W=05_34_predict_ct_${METHOD}.sh ;; mri) W=05_35_predict_mri_${METHOD}.sh ;; esac
  predict bash "${P}/05_predict/${W}" "${RID}"
  ev "export TRAINING_CONTRAST=${tc} METRICS_SUBDIR=; bash ${P}/06_evaluate/06_01_evaluate_run.sh ${RID} ${CAT}"
  ONLY=totalseg-pelvic ;;
brats_t1n|brats_t2w|brats_t2f|brats_t1c)
  B="${T}/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma"; tc="${S_#brats_}"
  case "$tc" in t1n) envf=env.sh; W=05_32; DS=051 ;; t2w) envf=env_t2w.sh; W=05_33; DS=052 ;;
                t2f) envf=env_t2f.sh; W=05_34; DS=053 ;; t1c) envf=env_t1c.sh; W=05_35; DS=054 ;; esac
  predict env TRAINING_CONTRAST="${tc}" bash "${B}/05_predict/${W}_predict_${tc}_${METHOD}.sh" "${RID}"
  ev "export TRAINING_CONTRAST=${tc}; source ${B}/00_utils/${envf}; export TRAINING_CONTRAST=${tc} DATASET_ID=${DS} EVAL_INLINE=1 CATEGORY=${CAT} METRICS_SUBDIR=
      for F in 0 1 2; do bash ${B}/06_evaluate/06_01_evaluate_run.sh ${RID} \${F} || exit 1; done"
  ONLY=brats2024-glioma ;;
onh_T1w|onh_T2w|onh_dwi_ap)   # checkpoint_best; RAS->native resample inside each predict job; evaluator refuses off-grid
  O="${T}/brain_healthy/on-harmony/5_scripts_on-harmony"; tc="${S_#onh_}"
  case "$tc" in T1w) envf=env.sh; W=05_30_predict_t1w ;; T2w) envf=env_t2w.sh; W=05_31_predict_t2w ;; dwi_ap) envf=env_dwi.sh; W=05_32_predict_dwi_ap ;; esac
  predict bash "${O}/05_predict/${W}_${METHOD}.sh" "${RID}"
  ev "source ${O}/00_utils/${envf}; export METRICS_SUBDIR=; for F in 0 1 2; do bash ${O}/06_evaluate/06_01_evaluate_testset.sh ${RID} \${F} || exit 1; done"
  ONLY=on-harmony ;;
openms_flair|openms_t1w)
  S="${T}/brain_ms/open-ms/5_scripts_open-ms"; tc="${S_#openms_}"
  case "$tc" in flair) W=05_64_predict_flair; E=06_01_evaluate_run.sh ;; t1w) W=05_65_predict_t1w; E=06_13_evaluate_t1w.sh ;; esac
  predict bash "${S}/05_predict/${W}_${METHOD}.sh" "${RID}"
  ev "export METRICS_SUBDIR=; bash ${S}/06_evaluate/${E} ${RID} ${CAT}"
  ONLY=open-ms ;;
ispy2_t1wce|ispy2_t2w)   # own + duke (A-P cropped *_uniap) + ispy1 + acrin6698 (dwi_uniap)
  I="${T}/breast_cancer/ispy2/5_scripts_ispy2"; tc="${S_#ispy2_}"
  case "$tc" in t1wce) envf=env.sh; W=05_60; DS=100; CW=05_33_predict_ispy2_t1wce_${METHOD}.sh ;;
                t2w) envf=env_t2w.sh; W=05_61; DS=101; CW=05_34_predict_ispy2_t2w_${METHOD}.sh ;; esac
  predict bash "${I}/05_predict/${W}_predict_${tc}_${METHOD}.sh" "${RID}"
  ev "source ${I}/00_utils/${envf}; export METRICS_SUBDIR=; bash ${I}/06_evaluate/06_01_evaluate_own_run.sh ${RID} ${CAT} ${DS} all"
  breast_comp duke-breast-mri "${CW}" "t1wce_uniap precontrast_uniap"
  breast_comp ispy1 "${CW}" "t1wce precontrast"
  breast_comp acrin6698 "${CW}" "dwi_uniap"
  ONLY=ispy2,duke-breast-mri,ispy1,acrin6698 ;;
panseg_t1wce|panseg_t2w)   # own + totalsegmri-pancreas (t1gre t2like) + msd-pancreas (ct), per-case crop-before-predict test sets
  S="${T}/pancreas_disease/pansegdata/5_scripts_pansegdata"; tc="${S_#panseg_}"
  case "$tc" in t1wce) envf=env.sh; W=05_43 ;; t2w) envf=env_t2w.sh; W=05_44 ;; esac
  predict bash "${S}/05_predict/${W}_predict_${tc}_${METHOD}.sh" "${RID}"
  ev "source ${S}/00_utils/${envf}; export TRAINING_CONTRAST=${tc} METRICS_SUBDIR=; bash ${S}/06_evaluate/06_01_evaluate_run.sh ${RID} ${CAT}"
  panseg_comp totalsegmri-pancreas "t1gre t2like" "${tc}"
  panseg_comp msd-pancreas "ct" "${tc}"
  ONLY=pansegdata,totalsegmri-pancreas,msd-pancreas ;;
chaos_t1in|chaos_t2spir)   # own + amos / sliver07 crop-before-predict (shared driver, Vulcan production crops, rung-5 fov mirror)
  C="${T}/abdomen_healthy/chaos/5_scripts_chaos"; tc="${S_#chaos_}"
  case "$tc" in t1in) envf=env.sh; DS=60; W=05_80 ;; t2spir) envf=env_t2spir.sh; DS=61; W=05_81 ;; esac
  predict bash "${C}/05_predict/${W}_predict_${tc}_${METHOD}.sh" "${RID}"
  ev "source ${C}/00_utils/${envf}; export DATASET_ID=${DS} CATEGORY=${CAT} METRICS_SUBDIR=; bash ${C}/06_evaluate/06_01_evaluate_run.sh ${RID}"
  MIR="${SCRATCH}/rung5_val000/fovmirror"; CROPS="${SCRATCH}/srcsm_match/fovmirror"; RF="${SD}/fovruns.txt"
  echo "${tc}|${CAT}|${RID}|${TR}|" > "${RF}"
  DRV="${ROOT}/benchmark/00_commun_scripts/00_02_predict/fov_crop_predict_evaluate.sh"
  for ds in amos sliver07; do
    if [ "$ds" = amos ]; then X="DATASET=amos ITEMS='ct mri' ANCHOR=kidney ANCHOR_IDS=2,3 EVAL_MODE=amos"
    else X="DATASET=sliver07 ITEMS=ct ANCHOR=liver ANCHOR_IDS=1 EVAL_MODE=generic_labels EVAL_LABELS=liver"; fi
    X="${X} CROP_REUSE_DIR=${CROPS}/${ds}/_fovcrop/vulcanprep"
    log "fov_crop ${ds}: predict + eval"
    # the pin goes on PYTHONPATH INSIDE the job command (the driver's nnUNetv2_predict must find nnUNetTrainerCHAOSSLAug)
    ( source scripts/job_runner/run_job.sh
      run run_job --name "slfov_${ds}_${tc}" --gpus 1 --cpus 16 --mem 64G --time 03:00:00 --log "${OUTD}/fov_${ds}_predict.log" --wait -- \
        bash -c "export PYTHONPATH=${SLAUG_PIN}:\${PYTHONPATH:-} SCRATCH=${MIR} RUNS_FILE=${RF} CONTRASTS=${tc} SKIP_REPORT=1 PHASES=2 FOV_NGPU=1 ${X}; bash ${DRV}" ) || fail "fov predict ${ds}"
    ( source scripts/job_runner/run_job.sh
      run run_job --name "slfovev_${ds}_${tc}" --gpus 0 --cpus 16 --mem 120G --time 02:00:00 --log "${OUTD}/fov_${ds}_eval.log" --wait -- \
        bash -c "export SCRATCH=${MIR} RUNS_FILE=${RF} CONTRASTS=${tc} SKIP_REPORT=1 PHASES='3 4' EVAL_PARALLEL=6 ${X}; bash ${DRV}" ) || fail "fov eval ${ds}"
  done
  ONLY=chaos,amos,sliver07 ;;
tf2_cbct)   # toothfairy2 own cbct + hanseg CT/MR + pddca CT on the S-I flipped _sif inputs; mandible-only
  MD="${ROOT}/${T}/mandible_healthy"
  predict bash "${MD}/toothfairy2/5_scripts_toothfairy2/05_predict/05_37_predict_cbct_${METHOD}.sh" "${RID}"
  predict env PREDICT_INPUT_SUFFIX=_sif PREDICT_OUTPUT_SUBDIR=sif bash "${MD}/hanseg/5_scripts_hanseg/05_predict/05_36_predict_${METHOD}.sh" "${RID}" all ct mrt1
  predict env PREDICT_INPUT_SUFFIX=_sif PREDICT_OUTPUT_SUBDIR=sif bash "${MD}/pddca/5_scripts_pddca/05_predict/05_36_predict_${METHOD}.sh" "${RID}" all ct
  TM="${MD}/toothfairy2/8_results_toothfairy2/02_metrics/toothfairy2_model/cbct"
  ev "export TF2_PRED=${MD}/toothfairy2/8_results_toothfairy2/01_predictions/toothfairy2_model/cbct TF2_METRICS=${TM} TF2_METRICS_SRC=${TM} \
        TF2_RAW=${MD}/toothfairy2/2_nnUNet_toothfairy2/raw/Dataset110_ToothFairy2CBCT FORCE_SUB= ONLY_RUN=${RID}
      bash ${MD}/toothfairy2/5_scripts_toothfairy2/06_evaluate/06_08_evaluate_cbct.sh"
  ev "export HS_PRED=${MD}/hanseg/8_results_hanseg/01_predictions/toothfairy2_model/cbct HS_METRICS=${MD}/hanseg/8_results_hanseg/02_metrics/toothfairy2_model/cbct \
        MO_DIR= FORCE_SUB= GT_OVERRIDE=${MD}/hanseg/2_nnUNet_hanseg/raw/labelsTs_ct_sif PRED_SUBDIR=sif ONLY_RUN=${RID}
      bash ${MD}/hanseg/5_scripts_hanseg/06_evaluate/06_03_eval_mandible_only.sh"
  ev "export PD_PRED=${MD}/pddca/8_results_pddca/01_predictions/toothfairy2_model/cbct PD_METRICS=${MD}/pddca/8_results_pddca/02_metrics/toothfairy2_model/cbct \
        MO_DIR= FORCE_SUB= GT_OVERRIDE=${MD}/pddca/2_nnUNet_pddca/raw/labelsTs_ct_sif PRED_SUBDIR=sif ONLY_RUN=${RID}
      bash ${MD}/pddca/5_scripts_pddca/06_evaluate/06_01_eval_mandible_only.sh"
  ONLY=toothfairy2,hanseg,pddca ;;
atlas_t1w)   # HCC: atlas own t1w + lld-mmri-hcc (8 items) + liverhccseg (4 items)
  L="${T}/liver_hcc"
  predict bash "${L}/atlas-liver-hcc/5_scripts_atlas-liver-hcc/05_predict/05_41_predict_t1w_${METHOD}.sh" "${RID}"
  predict bash "${L}/lld-mmri-hcc/5_scripts_lld-mmri-hcc/05_predict/05_29_predict_atlas_${METHOD}.sh" "${RID}" all t2wi dwi ce-pre ce-art ce-pv ce-del inphase outphase
  predict bash "${L}/liverhccseg/5_scripts_liverhccseg/05_predict/05_29_predict_atlas_${METHOD}.sh" "${RID}" all ce-pre_T1w ce-art_T1w ce-pv_T1w ce-del_T1w
  ev "export METRICS_SUBDIR=; bash ${L}/atlas-liver-hcc/5_scripts_atlas-liver-hcc/06_evaluate/06_01_evaluate_run.sh ${RID} ${CAT}"
  ev "export CATEGORY=${CAT} METRICS_SUBDIR=; bash ${L}/lld-mmri-hcc/5_scripts_lld-mmri-hcc/06_evaluate/06_01_evaluate_run.sh ${RID}"
  ev "export CATEGORY=${CAT} METRICS_SUBDIR=; bash ${L}/liverhccseg/5_scripts_liverhccseg/06_evaluate/06_01_evaluate_run.sh ${RID}"
  ONLY=atlas-liver-hcc,lld-mmri-hcc,liverhccseg ;;
*) log "FAILED: no recipe for ${S_}"; exit 3 ;;
esac

log "audit vs the headline PA+Auglab run (sources: ${ONLY})"
if [ -z "${DRY:-}" ]; then
  .venv/bin/python scripts/cluster/flatauglab/audit_vs_headline.py "${S_}" --only "${ONLY}" --run-table "${RUNS}" > "${SD}/POST_AUDIT.txt" 2>&1 || FAIL=1
  grep -v '^ok ' "${SD}/POST_AUDIT.txt" | tee -a "${ST}"
else echo "[dry] audit_vs_headline.py ${S_} --only ${ONLY} --run-table ${RUNS}" | tee -a "${ST}"; fi
if [ "${FAIL}" = 0 ]; then log "DONE ${S_}"; else log "FAILED ${S_} (see above; logs ${OUTD})"; fi
exit ${FAIL}
