#!/bin/bash
# Run ON TASSAN (repo checkout, after `source scripts/cluster/tassan_env.sh`): one data level N of the low-data texture
# step (totalseg-pelvic MRI-trained) end to end on ONE set_slot slot S (= GPU S, tassan_env.sh pairing):
#   1. train both arms (04_35 flat / texture) concurrently: 3 folds each, 6 folds packed on GPU S
#      (FOLD_SLOT_GPU "F,S,S"), DA_WORKERS per fold (default 3: 6 x 3 workers on the slot's 20 cores; dataset default
#      is 8, so ALLOW_METHOD_LOADER_OVERRIDE=1 -- loader speed only, same for both arms)
#   2. check every fold has checkpoint_final.pth and the used AugLab config equals the full-data run's
#   3. predict ct + mri for each fold on slot S only (05_24 with SLOT=S GPU=S, never FOLD=all), check 30 ct + 27 mri files
#   4. push run dirs (checkpoints, logs, predictions) + the run table to Vulcan, verify, write <VSTAGE>/pelvis_nNN.tassan_done
# Any failure writes pelvis_nNN.tassan_failed (reason + step) and a TASSAN_LOG line on Vulcan. Re-running resumes: the run
# table gives the RUN_IDs, train_common.sh resumes from checkpoint_latest, finished folds/predictions are skipped.
#   nohup bash scripts/cluster/lowdata_tassan/run_level.sh <N> <S> > <log> 2>&1 &
# Env: DA_WORKERS (3), MIN_FREE_GB (60: refuse to start training if GPU S has less free memory), PROBE=1 (3 epochs,
#      throwaway RUN_IDs ZZPROBE_*, no predict / push / marker), STEPS ("train predict push").
set -euo pipefail
N="${1:?N}"; SLOT="${2:?slot (0|1)}"
NN="$(printf '%02d' "${N}")"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO}"
D=benchmark/02_tasks/pelvis_healthy/totalseg-pelvic
SC="${D}/5_scripts_totalseg-pelvic"
RUNS="${D}/8_results_totalseg-pelvic/01_predictions/totalseg_pelvic_model/mri/auglab"
LOGS="${D}/8_results_totalseg-pelvic/_logs"
TABLE="${D}/8_results_totalseg-pelvic/_launch/RUN_IDS_lowdata_tassan.tsv"
VULCAN="${VULCAN:-vulcan.alliancecan.ca}"; VREPO="${VREPO:-/project/aip-jcohen/paulh/mri_synthesis_project}"
VSTAGE="${VSTAGE:-/scratch/paulh/lowdata_tassan}"
NAME="pelvis_n${NN}"; [ "${PROBE:-0}" = 1 ] && NAME="pelvis_probe_n${NN}"
TRAINER_DIR="Dataset131_TotalsegPelvic_MRI/nnUNetTrainerTotalsegPelvicAugLabDefaultLowData__nnUNetPlans__3d_fullres"
STEPS="${STEPS:-train predict push}"
mkdir -p "$(dirname "${TABLE}")" "${LOGS}"

vlog() { echo "[$(date -u +%FT%TZ)] ${NAME} $*"; [ "${PROBE:-0}" = 1 ] && return 0
         ssh "${VULCAN}" "echo '$(date -u +%FT%TZ) ${NAME} $(echo "$*" | tr -d "'")' >> ${VSTAGE}/TASSAN_LOG.txt" || true; }
fail() { vlog "FAIL step=$1: $2"; [ "${PROBE:-0}" = 1 ] && exit 1
         ssh "${VULCAN}" "printf '%s\n' '${NAME} tassan_failed at step $1 $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD)' '$(echo "$2" | tr -d "'")' > ${VSTAGE}/${NAME}.tassan_failed" || true
         exit 1; }
clean_env() { env -u nnUNet_raw -u nnUNet_preprocessed -u nnUNet_results -u PREDICTIONS_ROOT -u METRICS_ROOT \
                  -u RESULTS_DIR -u CHECKPOINTS_DIR -u SPLITS_DIR -u RUN_ID -u TRAINING_CONTRAST "$@"; }

# ---- preflight
[ "${RUN_JOB_BACKEND:-}" = set_slot ] && [ "${RUN_JOB_SET_SLOT_CUDA_FROM_SLOT:-}" = 1 ] || fail preflight "source scripts/cluster/tassan_env.sh first"
ls .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/TotalsegPelvicLowDataTrainers.py >/dev/null 2>&1 \
  || fail preflight "TotalsegPelvicLowDataTrainers.py not in the venv nnunetv2 trainer dir"
cmp -s "${SC}/02_nnunet/TotalsegPelvicLowDataTrainers.py" .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/TotalsegPelvicLowDataTrainers.py \
  || fail preflight "venv copy of TotalsegPelvicLowDataTrainers.py differs from the repo"
[ -d "${D}/2_nnUNet_totalseg-pelvic/preprocessed/Dataset131_TotalsegPelvic_MRI/nnUNetPlans_3d_fullres" ] || fail preflight "data not pulled"
HEAD_SHA="$(git rev-parse --short HEAD)"
AL=sub-workspaces/auglab_workspace/AugLab
AL_SHA="$(git -C "${AL}" rev-parse --short HEAD)"; AL_DIRTY="$(git -C "${AL}" status --porcelain | wc -l)"
DIRTY="$(git status --porcelain --untracked-files=no | wc -l)"
[ "${DIRTY}" = 0 ] || fail preflight "tracked files modified in the checkout (launch only from committed code)"

run_id() {   # RUN_ID of (N, arm) from the table, else a new one (recorded at launch)
  local arm="$1" r=""
  [ -f "${TABLE}" ] && r="$(awk -F'\t' -v n="${NN}" -v a="${arm}" -v p="${PROBE:-0}" '$1==n && $2==a && $9==p {print $3}' "${TABLE}" | tail -1)"
  if [ -z "${r}" ]; then
    r="totalseg-pelvic_mri_lowdata_n${NN}_${arm}_$(date +%Y%m%d_%H%M%S)"; [ "${PROBE:-0}" = 1 ] && r="ZZPROBE_${r}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${NN}" "${arm}" "${r}" "${SLOT}" "$(date -u +%FT%TZ)" "${HEAD_SHA}" "${AL_SHA}" "${AL_DIRTY}" "${PROBE:-0}" >> "${TABLE}"
    sleep 1
  fi
  echo "${r}"
}
RID_flat="$(run_id flat)"; RID_texture="$(run_id texture)"
fold_done() { [ -f "${RUNS}/$1/${TRAINER_DIR}/fold_$2/checkpoint_final.pth" ]; }

# ---- 1. train
if [[ " ${STEPS} " == *" train "* ]]; then
  todo=()
  for arm in flat texture; do rid="RID_${arm}"; for f in 0 1 2; do fold_done "${!rid}" "${f}" || { todo+=("${arm}"); break; }; done; done
  if [ ${#todo[@]} -gt 0 ]; then
    free_gb=$(( $(nvidia-smi -i "${SLOT}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ') / 1024 ))
    [ "${free_gb}" -ge "${MIN_FREE_GB:-60}" ] || fail train "GPU ${SLOT} has ${free_gb} GB free < ${MIN_FREE_GB:-60} GB"
    vlog "train start: arms=${todo[*]} slot=${SLOT} DA_WORKERS=${DA_WORKERS:-3} GPU free ${free_gb} GB main=${HEAD_SHA} AugLab=${AL_SHA} (dirty ${AL_DIRTY}) flat=${RID_flat} texture=${RID_texture}"
    EPOCHS_ENV=(); [ "${PROBE:-0}" = 1 ] && EPOCHS_ENV=(NNUNET_NUM_EPOCHS=3)
    pids=()
    for arm in "${todo[@]}"; do
      rid="RID_${arm}"
      clean_env FOLD_SLOT_GPU="0,${SLOT},${SLOT} 1,${SLOT},${SLOT} 2,${SLOT},${SLOT}" LAUNCH_WAIT=1 \
        DA_WORKERS="${DA_WORKERS:-3}" ALLOW_METHOD_LOADER_OVERRIDE=1 \
        "${EPOCHS_ENV[@]}" \
        bash "${SC}/04_train/04_35_train_mri_lowdata.sh" "${arm}" "${N}" "${!rid}" \
        > "${LOGS}/lowdata_launch_${!rid}.log" 2>&1 &
      pids+=($!); sleep 20
    done
    for p in "${pids[@]}"; do wait "${p}" || true; done
  fi
  for arm in flat texture; do rid="RID_${arm}"; for f in 0 1 2; do
    fold_done "${!rid}" "${f}" || fail train "${!rid} fold ${f}: no checkpoint_final.pth (see ${LOGS}/nnunet_totalseg_pelvic_mri_lowdata_n${NN}_${arm}/fold${f}.log)"
  done; done
  for arm in flat texture; do
    rid="RID_${arm}"; ref="$(ls ${RUNS}/*/Dataset131_TotalsegPelvic_MRI/nnUNetTrainerTotalsegPelvicAugLabDefault__nnUNetPlans__3d_fullres/fold_0/transform_params_gpu_used_for_training.json | grep -E "$([ "${arm}" = flat ] && echo flatfill_20261008_191257 || echo v26_6_2_train050_val000_20261005_223058)")"
    for f in 0 1 2; do
      .venv/bin/python -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])) != json.load(open(sys.argv[2])))' \
        "${RUNS}/${!rid}/${TRAINER_DIR}/fold_${f}/transform_params_gpu_used_for_training.json" "${ref}" \
        || fail train "${!rid} fold ${f}: used AugLab config differs from the full-data ${arm} run"
      grep -q "\[LowData\] fold ${f}: N=${N} " "${RUNS}/${!rid}/${TRAINER_DIR}/fold_${f}"/training_log_*.txt \
        || fail train "${!rid} fold ${f}: no [LowData] N=${N} line in the training log"
    done
  done
  vlog "train done: 6 folds have checkpoint_final; configs equal full-data runs; LowData N=${N} confirmed"
fi
[ "${PROBE:-0}" = 1 ] && { echo "PROBE done (no predict/push)"; exit 0; }

# ---- 2. predict (slot S only, one fold at a time, both items)
n_pred() { ls "${RUNS}/$1/fold$2/$3" 2>/dev/null | grep -c '\.nii\.gz$' || true; }
if [[ " ${STEPS} " == *" predict "* ]]; then
  for arm in flat texture; do rid="RID_${arm}"; for f in 0 1 2; do
    if [ "$(n_pred "${!rid}" "${f}" ct)" = 30 ] && [ "$(n_pred "${!rid}" "${f}" mri)" = 27 ]; then continue; fi
    clean_env SLOT="${SLOT}" GPU="${SLOT}" bash "${SC}/05_predict/05_24_predict_mri_lowdata.sh" "${!rid}" "${f}" ct mri \
      > "${LOGS}/lowdata_predict_${!rid}_fold${f}.log" 2>&1 || fail predict "${!rid} fold ${f}: predict exited non-zero"
    [ "$(n_pred "${!rid}" "${f}" ct)" = 30 ] && [ "$(n_pred "${!rid}" "${f}" mri)" = 27 ] \
      || fail predict "${!rid} fold ${f}: $(n_pred "${!rid}" "${f}" ct)/30 ct, $(n_pred "${!rid}" "${f}" mri)/27 mri predictions"
  done; done
  vlog "predict done: 2 runs x 3 folds x (30 ct + 27 mri)"
fi

# ---- 3. push to Vulcan + verify + marker
if [[ " ${STEPS} " == *" push "* ]]; then
  paths=("${TABLE}")
  for arm in flat texture; do
    rid="RID_${arm}"; paths+=("${RUNS}/${!rid}" "${LOGS}/nnunet_totalseg_pelvic_mri_lowdata_n${NN}_${arm}")
    paths+=($(ls -d "${LOGS}"/lowdata_*"${!rid}"* "${RUNS}/_logs/"*"${!rid}"* 2>/dev/null || true))
  done
  ssh "${VULCAN}" "test -e ${VSTAGE}/${NAME}.vulcan_eval_done" && fail push "${NAME} already evaluated on Vulcan (re-push would change evaluated inputs; ask first)"
  printf '%s\n' "${paths[@]}" | xargs -P 6 -I{} rsync -rlptR --partial ./{} "${VULCAN}:${VREPO}/" || fail push "rsync failed"
  left=0; for p in "${paths[@]}"; do left=$(( left + $(rsync -rlptR --dry-run --itemize-changes "./${p}" "${VULCAN}:${VREPO}/" | grep -c '^<f' || true) )); done
  [ "${left}" = 0 ] || fail push "${left} file(s) differ on Vulcan after push"
  ssh "${VULCAN}" "printf '%s\n' 'flat ${RID_flat}' 'texture ${RID_texture}' 'main ${HEAD_SHA} AugLab ${AL_SHA} (dirty ${AL_DIRTY}) $(date -u +%FT%TZ)' > ${VSTAGE}/${NAME}.tassan_done"
  vlog "push done: verified 0 differences; wrote ${NAME}.tassan_done"
fi
