#!/bin/bash
# VULCAN post-training controller of ONE low-data setting (both arms of the last launch.sh row): checks every fold has
# checkpoint_final + checkpoint_best, predicts (L40S run_job --wait jobs) and evaluates (inline in this CPU job) on the task's
# own test items and every external cohort the task's ladder pools, under the `lowdata` metrics subdir, then writes the summary
# (chaos 06_38 / pansegdata 06_21 -> <metrics>/<model>/<contrast>/lowdata/lowdata_texture_step_summary.md).
#   chaos : own t1in/t1out/t2spir/ct (06_01, METRICS_SUBDIR=lowdata) + amos (ct, mri) / sliver07 (ct) crop-before-predict with the
#           shared fov_crop_predict_evaluate.sh (RUNS_FILE subdir lowdata; production crops + fov mirror, the GIN-IPA recipe)
#   panseg: own t1wce/t2w (06_01) + totalsegmri-pancreas (t1gre, t2like) + msd-pancreas (ct) via their roster predict shim and
#           06_01 with METRICS_SUBDIR=lowdata (evaluate_companion_run_common.sh)
#   onh   : own cross-contrast test set (T1w T2w bold dwi_ap epi_ap gre_echo1_mag; checkpoint_best, no TTA, RAS->native resample in
#           each predict job) + 06_01_evaluate_testset.sh per fold with METRICS_SUBDIR=lowdata (refuses off-grid predictions)
# Queued by launch.sh (afterany on the 6 fold jobs) or by hand:
#   sbatch --account=aip-jcohen --cpus-per-task=16 --mem=96G --time=40:00:00 --wrap "bash scripts/cluster/lowdata_vulcan/post_run.sh <setting>"
# DRY=1 prints the steps. Status: /scratch/paulh/lowdata_vulcan/post/<setting>/POST.status. Re-running is safe for predict
# (overwrites); move an existing lowdata metrics dir aside before re-evaluating (stale-skip).
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH=/scratch/paulh
S_="${1:?usage: post_run.sh <setting>}"
source scripts/cluster/lowdata_vulcan/common.sh
ld_setting "${S_}" || exit 3
SD="${LD_POST}/${S_}"; mkdir -p "${SD}"
if [ -z "${LDPOST_SNAPSHOT:-}" ] && [ -z "${DRY:-}" ]; then   # private snapshot: later edits never hit a running controller
  snap="${SD}/post_run.snapshot.sh"; cp "$0" "${snap}" && export LDPOST_SNAPSHOT=1 && exec bash "${snap}" "$@"
fi
ST="${SD}/POST.status"; OUTD="${SD}/post_logs"; mkdir -p "${OUTD}"
export EVAL_MEM="${EVAL_MEM:-64G}"
unset RUN_JOB_DEPENDENCY RUN_JOB_INLINE
log() { echo "[$(date '+%F %T')] $*" | tee -a "${ST}"; }
FAIL=0; fail() { log "ERROR: $*"; FAIL=1; }
run() { if [ -n "${DRY:-}" ]; then echo "[dry] $*" | tee -a "${ST}"; return 0; fi; "$@"; }
T=benchmark/02_tasks
log "START ${S_} host=$(hostname) job=${SLURM_JOB_ID:-none} HEAD=$(git rev-parse --short HEAD)${DRY:+ (DRY)}"
RIDS=(); TRS=()
for ARM in flat texture; do
  rid="$(ld_rid "${S_}" "${ARM}")"
  [[ "${rid}" == *"_${LD_TC}_lowdata_n0${LD_N}_${ARM}_"* ]] || { log "FAILED: no ${ARM} run in ${LD_TABLE} ('${rid}')"; exit 2; }
  rd="${LD_PRED_ROOT}/${rid}"
  if [ -z "${DRY:-}" ]; then
    for k in 0 1 2; do for ck in checkpoint_best checkpoint_final; do
      ls "${rd}"/Dataset*/*/fold_${k}/${ck}.pth >/dev/null 2>&1 || { log "FAILED: ${ARM} ${rid} fold ${k} has no ${ck}.pth (training incomplete: resume with the same RUN_ID, then rerun this)"; exit 2; }
    done; done
  fi
  tr="$(basename "$(ls -d "${rd}"/Dataset*/*__nnUNetPlans__3d_fullres 2>/dev/null | head -1)" | sed 's/__nnUNetPlans__3d_fullres//')"
  RIDS+=("${rid}"); TRS+=("${tr}")
  log "${ARM}: ${rid} trainer ${tr:-?}"
done
inline() { ( export RUN_JOB_INLINE=1; run bash -c "$1" ) > "${OUTD}/eval_$2.log" 2>&1 || fail "eval $2 (${OUTD}/eval_$2.log)"; }

for i in 0 1; do
  RID="${RIDS[$i]}"; TR="${TRS[$i]}"; tag="${RID##*_lowdata_}"
  log "predict own ${RID}"
  run env LOWDATA_CONTRAST="${LD_TC}" bash "${LD_SCRIPTS}/${LD_PREDICT}" "${RID}" > "${OUTD}/predict_own_${tag}.log" 2>&1 || fail "predict own ${RID}"
  case "${S_}" in
  chaos_*)
    case "${LD_TC}" in t1in) envf=env.sh; DS=60 ;; t2spir) envf=env_t2spir.sh; DS=61 ;; esac
    inline "source ${LD_SCRIPTS}/00_utils/${envf}; export DATASET_ID=${DS} CATEGORY=${LD_CAT} METRICS_SUBDIR=lowdata; bash ${LD_SCRIPTS}/06_evaluate/06_01_evaluate_run.sh ${RID}" "own_${tag}"
    MIR="${SCRATCH}/rung5_val000/fovmirror"; CROPS="${SCRATCH}/srcsm_match/fovmirror"; RF="${SD}/fovruns_${tag}.txt"
    echo "${LD_TC}|${LD_CAT}|${RID}|${TR}|lowdata" > "${RF}"
    DRV="${PWD}/benchmark/00_commun_scripts/00_02_predict/fov_crop_predict_evaluate.sh"
    for ds in amos sliver07; do
      if [ "${ds}" = amos ]; then X="DATASET=amos ITEMS='ct mri' ANCHOR=kidney ANCHOR_IDS=2,3 EVAL_MODE=amos"
      else X="DATASET=sliver07 ITEMS=ct ANCHOR=liver ANCHOR_IDS=1 EVAL_MODE=generic_labels EVAL_LABELS=liver"; fi
      X="${X} CROP_REUSE_DIR=${CROPS}/${ds}/_fovcrop/vulcanprep"
      log "fov_crop ${ds} ${tag}: predict + eval"
      ( source scripts/job_runner/run_job.sh
        run run_job --name "ldfov_${ds}_${S_}_${i}" --gpus 1 --cpus 16 --mem 64G --time 03:00:00 --log "${OUTD}/fov_${ds}_${tag}_predict.log" --wait -- \
          bash -c "export SCRATCH=${MIR} RUNS_FILE=${RF} CONTRASTS=${LD_TC} SKIP_REPORT=1 PHASES=2 FOV_NGPU=1 ${X}; bash ${DRV}" ) || fail "fov predict ${ds} ${tag}"
      ( source scripts/job_runner/run_job.sh
        run run_job --name "ldfovev_${ds}_${S_}_${i}" --gpus 0 --cpus 16 --mem 120G --time 02:00:00 --log "${OUTD}/fov_${ds}_${tag}_eval.log" --wait -- \
          bash -c "export SCRATCH=${MIR} RUNS_FILE=${RF} CONTRASTS=${LD_TC} SKIP_REPORT=1 PHASES='3 4' EVAL_PARALLEL=6 ${X}; bash ${DRV}" ) || fail "fov eval ${ds} ${tag}"
    done ;;
  panseg_*)
    case "${LD_TC}" in t1wce) envf=env.sh ;; t2w) envf=env_t2w.sh ;; esac
    inline "source ${LD_SCRIPTS}/00_utils/${envf}; export TRAINING_CONTRAST=${LD_TC} METRICS_SUBDIR=lowdata; bash ${LD_SCRIPTS}/06_evaluate/06_01_evaluate_run.sh ${RID} ${LD_CAT}" "own_${tag}"
    for CN in totalsegmri-pancreas msd-pancreas; do
      C="${T}/pancreas_disease/${CN}/5_scripts_${CN}"
      log "predict ${CN} ${tag}"
      if [ -n "${DRY:-}" ]; then echo "[dry] ${CN}: roster shim 05_01_predict_pansegdata_common.sh ${RID} all (TRAINER ${TR})" | tee -a "${ST}"
      else
        ( set -euo pipefail; unset PREDICTIONS_ROOT METRICS_ROOT nnUNet_raw nnUNet_preprocessed RUN_JOB_INLINE
          source "${C}/00_utils/env.sh"; source "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_00_utils/roster_lib.sh"
          rd="$(roster_src_run_dir PANSEG "${LD_TC}" "${LD_CAT}" "${RID}")"; roster_run_trainer_id "${rd}" || { echo "no trained model under ${rd}"; exit 1; }
          export PANSEG_TRAINING_CONTRAST="${LD_TC}" PANSEG_DATASET_ID="${RUN_DATASET_ID}"
          METHOD=lowdata; TRAINER="${RUN_TRAINER}"; CATEGORY="${LD_CAT}"
          source "${C}/05_predict/05_01_predict_pansegdata_common.sh" "${RID}" all ) > "${OUTD}/predict_${CN}_${tag}.log" 2>&1 || { fail "predict ${CN} ${tag}"; continue; }
      fi
      inline "export EVAL_INLINE=1 METRICS_SUBDIR=lowdata LADDER=0; bash ${C}/06_evaluate/06_01_evaluate_run.sh ${RID} ${LD_CAT} ${LD_TC} all" "${CN}_${tag}"
    done ;;
  onh_*)
    case "${LD_TC}" in T1w) envf=env.sh ;; T2w) envf=env_t2w.sh ;; dwi_ap) envf=env_dwi.sh ;; esac
    inline "source ${LD_SCRIPTS}/00_utils/${envf}; export METRICS_SUBDIR=lowdata; for F in 0 1 2; do bash ${LD_SCRIPTS}/06_evaluate/06_01_evaluate_testset.sh ${RID} \${F} || exit 1; done" "own_${tag}" ;;
  esac
done
log "summary ${LD_SUMMARY} ${LD_TC}"
run bash "${LD_SCRIPTS}/${LD_SUMMARY}" "${LD_TC}" > "${OUTD}/summary.log" 2>&1 || fail "summary"
if [ -z "${DRY:-}" ]; then
  grep -q '^| N=' "${OUTD}/summary.log" || fail "summary has no low-data row (see ${OUTD}/summary.log: Pending section)"
  grep -A3 '^Pending' "${OUTD}/summary.log" | tee -a "${ST}"
  grep -m1 -- '^-> ' "${OUTD}/summary.log" | tee -a "${ST}"
fi
if [ "${FAIL}" = 0 ]; then log "DONE ${S_}"; else log "FAILED ${S_} (see above; logs ${OUTD})"; fi
exit ${FAIL}
