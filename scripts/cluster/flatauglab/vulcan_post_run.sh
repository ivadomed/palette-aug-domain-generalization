#!/bin/bash
# VULCAN post-training controller for ONE setting of the flatauglab arm ("flat fill + Auglab", METHOD
# auglabAugFlat_v26_6_2_train050_val000), 2026-10-29: own-test predict + evaluate AND the companions that were planned on TamIA
# (mandible hanseg/pddca on the S-I flipped _sif inputs, chaos -> amos/sliver07 crop-before-predict, HCC lld-mmri-hcc/liverhccseg),
# moved off TamIA so its priority goes to the trainings (TamIA post jobs 651756-651798 cancelled). Recipes = a clone of
# scripts/cluster/rung45_flat/post_run.sh (which ran the same datasets on Vulcan) with: the arm's own predict wrappers, metrics FLAT next
# to the headline PA+Auglab runs (METRICS_SUBDIR= / FORCE_SUB= set-but-empty), no breast / pancreas companions (companions_poller.sh +
# companions_vulcan.sh run them), no CE_EXTRA_PYTHONPATH pin, an HCC recipe from the liver_hcc standard 06_01 evaluators, and the audit
# = audit_vs_headline.py (row count + distinct cases per item, per fold, vs the headline PA+Auglab run in every source it produced).
# Model = the checkpoints companions_vulcan.sh rsynced from TamIA and md5-verified (<companions dir>/<setting>/SYNCED).
# Submitted (CPU job; the predicts are L40S run_job --wait jobs) by vulcan_post_poller.sh, or by hand:
#   sbatch --account=aip-jcohen --cpus-per-task=16 --mem=96G --time=16:00:00 --job-name=fapost_<setting> \
#     --output=/scratch/paulh/flatauglab_20261028/companions/<setting>/post_job_%j.log \
#     --wrap "bash scripts/cluster/flatauglab/vulcan_post_run.sh <setting>"
# Status: <companions dir>/<setting>/POST.status (last line "DONE <setting>" or "FAILED <setting>").
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/paulh}"
S_="${1:?usage: vulcan_post_run.sh <setting>}"
CD="${FLATAUG_COMPANIONS_DIR:-/scratch/paulh/flatauglab_20261028/companions}"
RUNS="${FLATAUG_RUN_TABLE:-/scratch/paulh/flatauglab_20261028/run_ids.tsv}"
SD="${CD}/${S_}"; mkdir -p "${SD}"
# bash reads a script lazily: run a private snapshot so later edits never hit a RUNNING controller
if [ -z "${FAPOST_SNAPSHOT:-}" ]; then
  snap="${SD}/post_run.snapshot.sh"; cp "$0" "${snap}" && export FAPOST_SNAPSHOT=1 && exec bash "${snap}" "$@"
fi
ROOT="$PWD"; T=benchmark/02_tasks; METHOD=auglabAugFlat_v26_6_2_train050_val000; CAT=auglab
ST="${SD}/POST.status"; OUTD="${SD}/post_logs"; mkdir -p "${OUTD}"
export RUN_JOB_ACCOUNT=aip-jcohen RUN_JOB_GPU_TYPE=l40s
export RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-rack02-06}"   # CUDA-init failures there end COMPLETED (2026-10-06)
unset RUN_JOB_DEPENDENCY RUN_JOB_INLINE
log() { echo "[$(date '+%F %T')] $*" | tee -a "${ST}"; }
FAIL=0; fail() { log "ERROR: $*"; FAIL=1; }
log "START ${S_} host=$(hostname) job=${SLURM_JOB_ID:-none} HEAD=$(git rev-parse --short HEAD)"

RID="$(awk -F'\t' -v s="${S_}" '$1==s {print $3}' "${RUNS}" | sort -u)"
[ "$(echo "${RID}" | grep -c .)" = 1 ] && [[ "${RID}" == *_${METHOD}_[0-9]* ]] || { log "FAILED: run id of ${S_} not unique / not the arm: '${RID}'"; exit 2; }
[ -f "${SD}/SYNCED" ] || { log "FAILED: ${SD}/SYNCED missing (checkpoints not synced from TamIA yet)"; exit 2; }
RDIR="$(head -1 "${SD}/SYNCED" | cut -d' ' -f1)"
[ "$(basename "${RDIR}")" = "${RID}" ] || { log "FAILED: SYNCED dir ${RDIR} is not ${RID}"; exit 2; }
for k in 0 1 2; do for ck in checkpoint_best checkpoint_final; do
  ls "${RDIR}"/Dataset*/*/fold_${k}/${ck}.pth >/dev/null 2>&1 || { log "FAILED: missing ${ck} fold ${k} in ${RDIR}"; exit 2; }
done; done
TR="$(basename "$(ls -d "${RDIR}"/Dataset*/*__nnUNetPlans__3d_fullres | head -1)" | sed 's/__nnUNetPlans__3d_fullres//')"
log "run ${RID} trainer ${TR} dir ${RDIR}: checkpoints OK (best + final, folds 0-2)"

inline_eval() { ( export RUN_JOB_INLINE=1; "$@" ) > "${OUTD}/eval_$(date +%H%M%S)_$RANDOM.log" 2>&1 || fail "eval: ${*: -1}"; }
ev() { inline_eval bash -c "$1"; }
predict() {   # predict [env K=V ...] bash <wrapper> args  -- blocking (predict_common: run_job --gpus 1 --wait per fold)
  local w; w="$(printf '%s\n' "$@" | grep -m1 '\.sh$')"; local tag; tag="$(basename "$(dirname "$(dirname "$w")")")_$(basename "$w" .sh)"
  log "predict ${tag}"; "$@" > "${OUTD}/predict_${tag}.log" 2>&1 || fail "predict: $*"; }
ONLY=""   # datasets this controller produces metrics for (audit scope)

case "${S_}" in
pelvic_ct|pelvic_mri)
  P="${T}/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic"; tc="${S_#pelvic_}"
  case "$tc" in ct) W=05_30_predict_ct_${METHOD}.sh ;; mri) W=05_31_predict_mri_${METHOD}.sh ;; esac
  predict bash "${P}/05_predict/${W}" "${RID}"
  ev "export TRAINING_CONTRAST=${tc} METRICS_SUBDIR=; bash ${P}/06_evaluate/06_01_evaluate_run.sh ${RID} ${CAT}"
  ONLY=totalseg-pelvic ;;
brats_t1n|brats_t2w|brats_t2f|brats_t1c)
  B="${T}/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma"; tc="${S_#brats_}"
  case "$tc" in t1n) envf=env.sh; W=05_95; DS=051 ;; t2w) envf=env_t2w.sh; W=05_96; DS=052 ;;
                t2f) envf=env_t2f.sh; W=05_97; DS=053 ;; t1c) envf=env_t1c.sh; W=05_98; DS=054 ;; esac
  predict env TRAINING_CONTRAST="${tc}" bash "${B}/05_predict/${W}_predict_${tc}_${METHOD}.sh" "${RID}"
  ev "export TRAINING_CONTRAST=${tc}; source ${B}/00_utils/${envf}; export TRAINING_CONTRAST=${tc} DATASET_ID=${DS} EVAL_INLINE=1 CATEGORY=${CAT} METRICS_SUBDIR=
      for F in 0 1 2; do bash ${B}/06_evaluate/06_01_evaluate_run.sh ${RID} \${F} || exit 1; done"
  ONLY=brats2024-glioma ;;
onh_T1w|onh_T2w|onh_dwi_ap)   # checkpoint_best; RAS->native resample inside each predict job (PREDICT_POST_ITEM_CMD); evaluator refuses off-grid
  O="${T}/brain_healthy/on-harmony/5_scripts_on-harmony"; tc="${S_#onh_}"
  case "$tc" in T1w) envf=env.sh; W=05_24_predict_t1w ;; T2w) envf=env_t2w.sh; W=05_25_predict_t2w ;; dwi_ap) envf=env_dwi.sh; W=05_26_predict_dwi_ap ;; esac
  predict bash "${O}/05_predict/${W}_${METHOD}.sh" "${RID}"
  ev "source ${O}/00_utils/${envf}; export METRICS_SUBDIR=; for F in 0 1 2; do bash ${O}/06_evaluate/06_01_evaluate_testset.sh ${RID} \${F} || exit 1; done"
  ONLY=on-harmony ;;
openms_flair|openms_t1w)
  S="${T}/brain_ms/open-ms/5_scripts_open-ms"; tc="${S_#openms_}"
  case "$tc" in flair) W=05_60_predict_flair; E=06_01_evaluate_run.sh ;; t1w) W=05_61_predict_t1w; E=06_13_evaluate_t1w.sh ;; esac
  predict bash "${S}/05_predict/${W}_${METHOD}.sh" "${RID}"
  ev "export METRICS_SUBDIR=; bash ${S}/06_evaluate/${E} ${RID} ${CAT}"
  ONLY=open-ms ;;
ispy2_t1wce|ispy2_t2w)   # own test set only (duke / ispy1 / acrin6698: companions_vulcan.sh)
  I="${T}/breast_cancer/ispy2/5_scripts_ispy2"; tc="${S_#ispy2_}"
  case "$tc" in t1wce) envf=env.sh; W=05_56; DS=100 ;; t2w) envf=env_t2w.sh; W=05_57; DS=101 ;; esac
  predict bash "${I}/05_predict/${W}_predict_${tc}_${METHOD}.sh" "${RID}"
  ev "source ${I}/00_utils/${envf}; export METRICS_SUBDIR=; bash ${I}/06_evaluate/06_01_evaluate_own_run.sh ${RID} ${CAT} ${DS} all"
  ONLY=ispy2 ;;
panseg_t1wce|panseg_t2w)   # own test set only (totalsegmri / msd-pancreas: companions_vulcan.sh)
  S="${T}/pancreas_disease/pansegdata/5_scripts_pansegdata"; tc="${S_#panseg_}"
  case "$tc" in t1wce) envf=env.sh; W=05_38 ;; t2w) envf=env_t2w.sh; W=05_39 ;; esac
  predict bash "${S}/05_predict/${W}_predict_${tc}_${METHOD}.sh" "${RID}"
  ev "source ${S}/00_utils/${envf}; export TRAINING_CONTRAST=${tc} METRICS_SUBDIR=; bash ${S}/06_evaluate/06_01_evaluate_run.sh ${RID} ${CAT}"
  ONLY=pansegdata ;;
chaos_t1in|chaos_t2spir)   # own + amos / sliver07 crop-before-predict (shared driver, Vulcan production crops, rung-5 fov mirror)
  C="${T}/abdomen_healthy/chaos/5_scripts_chaos"; A="${T}/abdomen_healthy"; tc="${S_#chaos_}"
  case "$tc" in t1in) envf=env.sh; DS=60; W=05_75 ;; t2spir) envf=env_t2spir.sh; DS=61; W=05_76 ;; esac
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
    ( source scripts/job_runner/run_job.sh
      run_job --name "fafov_${ds}_${tc}" --gpus 1 --cpus 16 --mem 64G --time 03:00:00 --log "${OUTD}/fov_${ds}_predict.log" --wait -- \
        bash -c "export SCRATCH=${MIR} RUNS_FILE=${RF} CONTRASTS=${tc} SKIP_REPORT=1 PHASES=2 FOV_NGPU=1 ${X}; bash ${DRV}" ) || fail "fov predict ${ds}"
    ( source scripts/job_runner/run_job.sh
      run_job --name "fafovev_${ds}_${tc}" --gpus 0 --cpus 16 --mem 120G --time 02:00:00 --log "${OUTD}/fov_${ds}_eval.log" --wait -- \
        bash -c "export SCRATCH=${MIR} RUNS_FILE=${RF} CONTRASTS=${tc} SKIP_REPORT=1 PHASES='3 4' EVAL_PARALLEL=6 ${X}; bash ${DRV}" ) || fail "fov eval ${ds}"
  done
  ONLY=chaos,amos,sliver07 ;;
tf2_cbct)   # toothfairy2 own cbct + hanseg CT/MR + pddca CT on the S-I flipped _sif inputs; mandible-only
  MD="${ROOT}/${T}/mandible_healthy"
  predict bash "${MD}/toothfairy2/5_scripts_toothfairy2/05_predict/05_35_predict_cbct_${METHOD}.sh" "${RID}"
  predict env PREDICT_INPUT_SUFFIX=_sif PREDICT_OUTPUT_SUBDIR=sif bash "${MD}/hanseg/5_scripts_hanseg/05_predict/05_34_predict_${METHOD}.sh" "${RID}" all ct mrt1
  predict env PREDICT_INPUT_SUFFIX=_sif PREDICT_OUTPUT_SUBDIR=sif bash "${MD}/pddca/5_scripts_pddca/05_predict/05_34_predict_${METHOD}.sh" "${RID}" all ct
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
atlas_t1w)   # HCC: atlas own t1w + lld-mmri-hcc (8 items) + liverhccseg (4 items), standard liver_hcc 06_01 evaluators, flat
  L="${T}/liver_hcc"
  predict bash "${L}/atlas-liver-hcc/5_scripts_atlas-liver-hcc/05_predict/05_39_predict_t1w_${METHOD}.sh" "${RID}"
  predict bash "${L}/lld-mmri-hcc/5_scripts_lld-mmri-hcc/05_predict/05_27_predict_atlas_${METHOD}.sh" "${RID}" all t2wi dwi ce-pre ce-art ce-pv ce-del inphase outphase
  predict bash "${L}/liverhccseg/5_scripts_liverhccseg/05_predict/05_27_predict_atlas_${METHOD}.sh" "${RID}" all ce-pre_T1w ce-art_T1w ce-pv_T1w ce-del_T1w
  ev "export METRICS_SUBDIR=; bash ${L}/atlas-liver-hcc/5_scripts_atlas-liver-hcc/06_evaluate/06_01_evaluate_run.sh ${RID} ${CAT}"
  ev "export CATEGORY=${CAT} METRICS_SUBDIR=; bash ${L}/lld-mmri-hcc/5_scripts_lld-mmri-hcc/06_evaluate/06_01_evaluate_run.sh ${RID}"
  ev "export CATEGORY=${CAT} METRICS_SUBDIR=; bash ${L}/liverhccseg/5_scripts_liverhccseg/06_evaluate/06_01_evaluate_run.sh ${RID}"
  ONLY=atlas-liver-hcc,lld-mmri-hcc,liverhccseg ;;
*) log "FAILED: no recipe for ${S_}"; exit 3 ;;
esac

log "audit vs the headline PA+Auglab run (sources: ${ONLY})"
.venv/bin/python scripts/cluster/flatauglab/audit_vs_headline.py "${S_}" --only "${ONLY}" --run-table "${RUNS}" > "${SD}/POST_AUDIT.txt" 2>&1 || FAIL=1
grep -v '^ok ' "${SD}/POST_AUDIT.txt" | tee -a "${ST}"
if [ "${FAIL}" = 0 ]; then log "DONE ${S_}"; else log "FAILED ${S_} (see above; logs ${OUTD})"; fi
exit ${FAIL}
