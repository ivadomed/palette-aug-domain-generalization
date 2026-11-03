#!/bin/bash
# Run ON TASSAN (repo checkout, after `source scripts/cluster/tassan_env.sh`): train some of the pansegdata low-data (N=5)
# texture-step folds that Vulcan has queued, into the SAME RUN_IDs (Vulcan run table
# 8_results_pansegdata/_launch/RUN_IDS_lowdata_vulcan.tsv, staged by `LOWDATA_TASK=panseg stage_list.sh`), then push the
# fold dirs back. Prediction + evaluation stay on Vulcan (post_run.sh, started by panseg_handoff.sh there).
#   bash run_panseg.sh probe       PROBE (20 epochs, throwaway RUN_IDs ZZPROBE_*, nothing written on Vulcan): GPU 0 = t1wce
#                                  texture folds 0-2 (3 folds/GPU), GPU 1 = t2w flat + t2w texture folds 0-2 (6 folds/GPU),
#                                  concurrently; then `bash run_panseg.sh report probe` prints steady s/epoch per fold
#   bash run_panseg.sh train a     all 12 folds: GPU 0 = t1wce flat+texture (6), GPU 1 = t2w flat+texture (6)
#   bash run_panseg.sh train b     the 6 texture folds: GPU 0 = t1wce texture (3), GPU 1 = t2w texture (3)
#   bash run_panseg.sh push        push-only retry (train does it at the end)
#   bash run_panseg.sh status|report [probe]
# train: refuses unless every taken fold's Vulcan job is still PENDING and its Vulcan fold dir does not exist; launches the
# 04_45 wrapper per (contrast, arm) with FOLD_SLOT_GPU; once EVERY taken fold has logged >= 2 epochs it writes
# <VSTAGE>/Pancreas.tassan_taking (one line per fold: setting arm fold RUN_ID gpu) -- the Vulcan poller then cancels exactly
# those Vulcan fold jobs and the two post controllers; waits for checkpoint_final, checks configs == full-data runs and the
# [LowData] N=5 line, pushes fold dirs (no wandb/, no validation volumes) + plans/dataset json, md5-checks checkpoint_final/best
# on Vulcan (-> <fold dir>/.lowdata_synced_from_tassan) and writes Pancreas_<setting>.tassan_done per setting.
# Failures write Pancreas.tassan_failed (step + reason). Re-running `train <same layout>` resumes (RUN_IDs from the table,
# train_common.sh resumes from checkpoint_latest, finished folds are not relaunched, the taking marker is not rewritten).
# Env: DA_WORKERS_6 (2: 6 folds/GPU), DA_WORKERS_3 (3: 3 folds/GPU), MIN_FREE_GB_PER_FOLD (10), PROBE_EPOCHS (20).
#      Recommended for every run: nnUNet_def_n_proc=1 nnUNet_wandb_enabled=0 (Pelvis lessons on tassan).
set -euo pipefail
MODE="${1:?probe|train <a|b>|push|status|report}"; LAYOUT="${2:-}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO}"
D=benchmark/02_tasks/pancreas_disease/pansegdata
SC="${D}/5_scripts_pansegdata"
RES="${D}/8_results_pansegdata"
VTABLE="${RES}/_launch/RUN_IDS_lowdata_vulcan.tsv"
UNITS="${RES}/_launch/lowdata_tassan_units.tsv"; [ "${MODE}" = probe ] || [ "${LAYOUT}" = probe ] && UNITS="${RES}/_launch/lowdata_tassan_probe_units.tsv"
LOGS="${RES}/_logs"
VULCAN="${VULCAN:-vulcan.alliancecan.ca}"; VREPO="${VREPO:-/project/aip-jcohen/paulh/mri_synthesis_project}"
VSTAGE="${VSTAGE:-/scratch/paulh/lowdata_tassan}"
TRAINER_DIR_OF() { case "$1" in t1wce) echo Dataset150_PanSegData_T1WCE ;; t2w) echo Dataset151_PanSegData_T2W ;; esac; }
TRAINER=nnUNetTrainerPANSEGDATAAugLabDefaultLowData__nnUNetPlans__3d_fullres
RUNS_OF() { echo "${RES}/01_predictions/pansegdata_model/$1/auglab"; }
declare -A REF=(
  [t1wce_flat]=pansegdata_t1wce_baseline_kmeans_label_remap_voronoi_flatfill_20261009_080433
  [t1wce_texture]=pansegdata_t1wce_v26_6_2_train050_val000_20261005_223237
  [t2w_flat]=pansegdata_t2w_baseline_kmeans_label_remap_voronoi_flatfill_20261009_080507
  [t2w_texture]=pansegdata_t2w_v26_6_2_train050_val000_20261005_223311 )
NAME=pancreas; [ "${MODE}" = probe ] && NAME=pancreas_probe
vlog() { echo "[$(date -u +%FT%TZ)] ${NAME} $*"; [ "${MODE}" = probe ] && return 0
         ssh "${VULCAN}" "echo '$(date -u +%FT%TZ) ${NAME} $(echo "$*" | tr -d "'")' >> ${VSTAGE}/TASSAN_LOG.txt" || true; }
fail() { vlog "FAIL step=$1: $2"; [ "${MODE}" = probe ] && exit 1
         ssh "${VULCAN}" "printf '%s\n' 'pancreas tassan_failed at step $1 $(date -u +%FT%TZ) HEAD=$(git rev-parse --short HEAD)' '$(echo "$2" | tr -d "'")' > ${VSTAGE}/Pancreas.tassan_failed" || true
         exit 1; }
clean_env() { env -u nnUNet_raw -u nnUNet_preprocessed -u nnUNet_results -u PREDICTIONS_ROOT -u METRICS_ROOT \
                  -u RESULTS_DIR -u CHECKPOINTS_DIR -u SPLITS_DIR -u RUN_ID -u TRAINING_CONTRAST -u DATASET_ID "$@"; }
fold_dir() { echo "$(RUNS_OF "$1")/$2/$(TRAINER_DIR_OF "$1")/${TRAINER}/fold_$3"; }   # tc rid fold
n_epochs() { local l; l="$(ls -t "$1"/training_log_*.txt 2>/dev/null | head -1 || true)"; if [ -n "${l}" ]; then grep -c 'Epoch time' "${l}" || true; else echo 0; fi; }

# ---- status / report (read the units file: mode setting arm fold rid gpu)
if [ "${MODE}" = status ] || [ "${MODE}" = report ]; then
  [ -f "${UNITS}" ] || { echo "no units file ${UNITS}"; exit 0; }
  while IFS=$'\t' read -r m s arm f rid g; do
    tc="${s#panseg_}"; fd="$(fold_dir "${tc}" "${rid}" "${f}")"; l="$(ls -t "${fd}"/training_log_*.txt 2>/dev/null | head -1 || true)"
    ts="$(grep -oE 'Epoch time: [0-9.]+' "${l}" 2>/dev/null | awk '{print $3}' || true)"
    steady="$(echo "${ts}" | awk 'NF && NR>5' | sort -n | awk '{a[NR]=$1} END {if (NR) print a[int((NR+1)/2)]; else print "-"}')"
    printf '%-6s %-13s %-7s f%s gpu%s epochs %-5s last %-7s median(ep>5) %-7s %s%s\n' "${m}" "${s}" "${arm}" "${f}" "${g}" \
      "$(n_epochs "${fd}")" "$(echo "${ts}" | tail -1)" "${steady}" "${rid}" "$([ -f "${fd}/checkpoint_final.pth" ] && echo ' FINAL')"
  done < "${UNITS}"
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  exit 0
fi

# ---- preflight
[ "${RUN_JOB_BACKEND:-}" = set_slot ] && [ "${RUN_JOB_SET_SLOT_CUDA_FROM_SLOT:-}" = 1 ] || fail preflight "source scripts/cluster/tassan_env.sh first"
VT=(.venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer)
for shim in PANSEGDATATrainers.py PANSEGDATALowDataTrainers.py; do
  cmp -s "${SC}/02_nnunet/${shim}" "${VT[0]}/${shim}" || fail preflight "venv copy of ${shim} missing or differs from the repo (cp it into ${VT[0]}/)"
done
for tc in t1wce t2w; do
  [ -d "${D}/2_nnUNet_pansegdata/preprocessed/$(TRAINER_DIR_OF "${tc}")/nnUNetPlans_3d_fullres" ] || fail preflight "data not pulled (${tc})"
done
[ -f "${VTABLE}" ] || fail preflight "Vulcan run table not pulled (${VTABLE})"
HEAD_SHA="$(git rev-parse --short HEAD)"
AL=sub-workspaces/auglab_workspace/AugLab
AL_SHA="$(git -C "${AL}" rev-parse --short HEAD)"; AL_DIRTY="$(git -C "${AL}" status --porcelain | wc -l)"
[ "$(git status --porcelain --untracked-files=no | wc -l)" = 0 ] || fail preflight "tracked files modified in the checkout (launch only from committed code)"

# ---- units: setting arm fold gpu
case "${MODE}:${LAYOUT}" in
  probe:)  U="panseg_t1wce texture 0 0|panseg_t1wce texture 1 0|panseg_t1wce texture 2 0|panseg_t2w flat 0 1|panseg_t2w flat 1 1|panseg_t2w flat 2 1|panseg_t2w texture 0 1|panseg_t2w texture 1 1|panseg_t2w texture 2 1" ;;
  train:a|push:a) U="panseg_t1wce flat 0 0|panseg_t1wce flat 1 0|panseg_t1wce flat 2 0|panseg_t1wce texture 0 0|panseg_t1wce texture 1 0|panseg_t1wce texture 2 0|panseg_t2w flat 0 1|panseg_t2w flat 1 1|panseg_t2w flat 2 1|panseg_t2w texture 0 1|panseg_t2w texture 1 1|panseg_t2w texture 2 1" ;;
  train:b|push:b) U="panseg_t1wce texture 0 0|panseg_t1wce texture 1 0|panseg_t1wce texture 2 0|panseg_t2w texture 0 1|panseg_t2w texture 1 1|panseg_t2w texture 2 1" ;;
  push:)   [ -f "${UNITS}" ] || fail push "no units file"; U="$(awk -F'\t' '{printf "%s%s %s %s %s", (NR>1?"|":""), $2, $3, $4, $6}' "${UNITS}")" ;;
  *) echo "usage: run_panseg.sh probe | train <a|b> | push [a|b] | status | report [probe]" >&2; exit 2 ;;
esac
IFS='|' read -r -a UL <<< "${U}"
vrid() { awk -F'\t' -v s="$1" -v a="$2" '$1==s && $2==a {r=$4} END {print r}' "${VTABLE}"; }
vjob() { awk -F'\t' -v s="$1" -v a="$2" -v f="$3" '$1==s && $2==a {j=$(5+f)} END {print j}' "${VTABLE}"; }
declare -A RID
if [ "${MODE}" = probe ]; then
  ts="$(date +%Y%m%d_%H%M%S)"
  for u in "${UL[@]}"; do read -r s arm f g <<< "${u}"; RID[${s}_${arm}]="ZZPROBE_pansegdata_${s#panseg_}_lowdata_n05_${arm}_${ts}"; done
else
  for u in "${UL[@]}"; do read -r s arm f g <<< "${u}"; r="$(vrid "${s}" "${arm}")"
    [[ "${r}" == "pansegdata_${s#panseg_}_lowdata_n05_${arm}_"* ]] || fail preflight "no ${s} ${arm} RUN_ID in ${VTABLE} ('${r}')"; RID[${s}_${arm}]="${r}"; done
fi
fold_done() { [ -f "$(fold_dir "${1#panseg_}" "${RID[$1_$2]}" "$3")/checkpoint_final.pth" ]; }

# ---- train
if [ "${MODE}" = probe ] || [ "${MODE}" = train ]; then
  TAKEN_EXISTS=0
  if [ "${MODE}" = train ]; then
    if ssh "${VULCAN}" "test -e ${VSTAGE}/Pancreas.tassan_taking"; then
      TAKEN_EXISTS=1   # resume: Vulcan already cancelled these; the marker must name exactly this layout
      want="$(for u in "${UL[@]}"; do read -r s arm f g <<< "${u}"; echo "${s} ${arm} ${f} ${RID[${s}_${arm}]} ${g}"; done | sort)"
      have="$(ssh "${VULCAN}" "grep -v '^#' ${VSTAGE}/Pancreas.tassan_taking" | sort)"
      [ "${want}" = "${have}" ] || fail preflight "Pancreas.tassan_taking on Vulcan names other folds than layout ${LAYOUT}"
    else
      ssh "${VULCAN}" "test -e ${VSTAGE}/Pancreas.tassan_failed" && fail preflight "Pancreas.tassan_failed exists on Vulcan: read it, move it to ${VSTAGE}/toDelete/ when fixed, rerun"
      for u in "${UL[@]}"; do read -r s arm f g <<< "${u}"
        j="$(vjob "${s}" "${arm}" "${f}")"; st="$(ssh "${VULCAN}" "squeue -h -j ${j} -o %T 2>/dev/null" || true)"
        [ "${st}" = PENDING ] || fail preflight "Vulcan job ${j} (${s} ${arm} fold ${f}) is '${st:-gone}', not PENDING: do not take it"
        vfd="${VREPO}/$(fold_dir "${s#panseg_}" "${RID[${s}_${arm}]}" "${f}")"
        ssh "${VULCAN}" "test ! -e ${vfd}" || fail preflight "${vfd} already exists on Vulcan"
      done
    fi
  fi
  [ "${MODE}" = probe ] && : > "${UNITS}"
  if [ "${MODE}" = train ]; then : > "${UNITS}.new"; fi
  for u in "${UL[@]}"; do read -r s arm f g <<< "${u}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${MODE}" "${s}" "${arm}" "${f}" "${RID[${s}_${arm}]}" "${g}" >> "${UNITS}$([ "${MODE}" = train ] && echo .new)"; done
  [ "${MODE}" = train ] && mv "${UNITS}.new" "${UNITS}"
  # per GPU: folds placed there -> loader workers
  declare -A NG; for u in "${UL[@]}"; do read -r s arm f g <<< "${u}"; NG[${g}]=$(( ${NG[${g}]:-0} + 1 )); done
  for g in "${!NG[@]}"; do
    free_gb=$(( $(nvidia-smi -i "${g}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ') / 1024 ))
    [ "${free_gb}" -ge $(( ${NG[${g}]} * ${MIN_FREE_GB_PER_FOLD:-10} )) ] || fail train "GPU ${g} has ${free_gb} GB free for ${NG[${g}]} folds"
  done
  pgrep -f "${TRAINER%%__*}" >/dev/null && fail train "a ${TRAINER%%__*} process is still running (a rerun must not train a fold twice: wait for it / kill it first)"
  EP_ENV=(); [ "${MODE}" = probe ] && EP_ENV=(NNUNET_NUM_EPOCHS="${PROBE_EPOCHS:-20}")
  vlog "train start: layout=${LAYOUT:-probe} folds/GPU $(for g in "${!NG[@]}"; do printf 'gpu%s=%s ' "${g}" "${NG[${g}]}"; done)main=${HEAD_SHA} AugLab=${AL_SHA} (dirty ${AL_DIRTY}) $(for k in "${!RID[@]}"; do printf '%s ' "${RID[${k}]}"; done)"
  pids=()
  for key in $(for u in "${UL[@]}"; do read -r s arm f g <<< "${u}"; echo "${s}_${arm}"; done | sort -u); do
    s="${key%_*}"; arm="${key##*_}"; tc="${s#panseg_}"; place=""; n=0
    for u in "${UL[@]}"; do read -r s2 a2 f g <<< "${u}"; [ "${s2}" = "${s}" ] && [ "${a2}" = "${arm}" ] || continue
      fold_done "${s}" "${arm}" "${f}" || { place+="${f},${g},${g} "; n=${NG[${g}]}; }; done
    [ -n "${place}" ] || continue
    W="${DA_WORKERS_3:-3}"; [ "${n}" -gt 3 ] && W="${DA_WORKERS_6:-2}"
    clean_env LOWDATA_CONTRAST="${tc}" FOLD_SLOT_GPU="${place% }" LAUNCH_WAIT=1 DA_WORKERS="${W}" ALLOW_METHOD_LOADER_OVERRIDE=1 \
      "${EP_ENV[@]}" bash "${SC}/04_train/04_45_train_lowdata_texture_step.sh" "${arm}" 5 "${RID[${key}]}" \
      > "${LOGS}/lowdata_tassan_launch_${RID[${key}]}.log" 2>&1 &
    pids+=($!); echo "launched ${key} placement '${place% }' DA_WORKERS=${W} pid $!"; sleep 20
  done
  # confirm every taken fold is training (>= 2 epochs logged), then tell Vulcan (train mode, first time only)
  if [ "${MODE}" = train ] && [ "${TAKEN_EXISTS}" = 0 ]; then
    t0=$(date +%s)
    while :; do
      ok=1; for u in "${UL[@]}"; do read -r s arm f g <<< "${u}"
        [ "$(n_epochs "$(fold_dir "${s#panseg_}" "${RID[${s}_${arm}]}" "${f}")")" -ge 2 ] || ok=0; done
      [ "${ok}" = 1 ] && break
      alive=0; for p in "${pids[@]}"; do kill -0 "${p}" 2>/dev/null && alive=1; done
      [ "${alive}" = 1 ] || fail train "launchers exited before every fold logged 2 epochs (see ${LOGS}/lowdata_tassan_launch_*.log); Vulcan untouched"
      [ $(( $(date +%s) - t0 )) -lt 7200 ] || fail train "not every fold logged 2 epochs within 2 h; Vulcan untouched (kill the trainers before rerunning)"
      sleep 60
    done
    { echo "# pancreas tassan layout ${LAYOUT}, confirmed training (>= 2 epochs per fold) $(date -u +%FT%TZ), main ${HEAD_SHA} AugLab ${AL_SHA}${ETA:+, ETA ${ETA}}"
      echo "# columns: setting arm fold RUN_ID tassan_gpu"
      for u in "${UL[@]}"; do read -r s arm f g <<< "${u}"; echo "${s} ${arm} ${f} ${RID[${s}_${arm}]} ${g}"; done; } > "${UNITS}.marker"
    rsync -q "${UNITS}.marker" "${VULCAN}:${VSTAGE}/Pancreas.tassan_taking" || fail train "could not write Pancreas.tassan_taking on Vulcan (trainers keep running; write it by hand from ${UNITS}.marker)"
    vlog "training confirmed on every taken fold; wrote Pancreas.tassan_taking (layout ${LAYOUT}${ETA:+, ETA ${ETA}})"
  fi
  for p in "${pids[@]}"; do wait "${p}" || true; done
  for u in "${UL[@]}"; do read -r s arm f g <<< "${u}"; fd="$(fold_dir "${s#panseg_}" "${RID[${s}_${arm}]}" "${f}")"; tc="${s#panseg_}"
    [ -f "${fd}/checkpoint_final.pth" ] && [ -f "${fd}/checkpoint_best.pth" ] || fail train "${RID[${s}_${arm}]} fold ${f}: no checkpoint_final/best (see ${LOGS}/nnunet_pansegdata_${tc}_lowdata_n05_${arm}/fold${f}.log)"
    ref="$(ls "$(RUNS_OF "${tc}")/${REF[${tc}_${arm}]}"/*/nnUNetTrainerPANSEGDATAAugLabDefault__nnUNetPlans__3d_fullres/fold_0/transform_params_gpu_used_for_training.json)"
    .venv/bin/python -c 'import json,sys; sys.exit(json.load(open(sys.argv[1])) != json.load(open(sys.argv[2])))' \
      "${fd}/transform_params_gpu_used_for_training.json" "${ref}" || fail train "${RID[${s}_${arm}]} fold ${f}: used AugLab config differs from the full-data ${arm} run"
    grep -q "\[LowData\] fold ${f}: N=5 " "${fd}"/training_log_*.txt || fail train "${RID[${s}_${arm}]} fold ${f}: no [LowData] N=5 line"
  done
  vlog "train done: ${#UL[@]} folds have checkpoint_final/best; configs equal full-data runs; LowData N=5 confirmed"
  [ "${MODE}" = probe ] && { echo "PROBE done -> bash $0 report probe"; exit 0; }
fi

# ---- push (fold dirs + trainer-level json; md5 of checkpoints; per-setting done marker)
MD5Q='for f in checkpoint_final.pth checkpoint_best.pth; do echo $f $(stat -c %s $f) $(md5sum < $f | cut -c1-32); done'
declare -A SET_DONE
for u in "${UL[@]}"; do read -r s arm f g <<< "${u}"; tc="${s#panseg_}"; rid="${RID[${s}_${arm}]}"
  fd="$(fold_dir "${tc}" "${rid}" "${f}")"; vfd="${VREPO}/${fd}"
  [ -f "${fd}/checkpoint_final.pth" ] || fail push "${rid} fold ${f}: no checkpoint_final.pth"
  if ssh "${VULCAN}" "test -e ${vfd}/.lowdata_synced_from_tassan"; then SET_DONE[${s}]+="${arm}${f} "; continue; fi
  ssh "${VULCAN}" "ls ${vfd}/training_log_* >/dev/null 2>&1 && [ ! -e ${vfd}/.lowdata_pushing_from_tassan ]" \
    && fail push "${vfd} on Vulcan already has a training log not pushed from tassan (Vulcan trained this fold too?): resolve by hand"
  ssh "${VULCAN}" "mkdir -p ${vfd} && touch ${vfd}/.lowdata_pushing_from_tassan" || fail push "mkdir ${vfd}"
  rsync -rlpt --include='*.json' --exclude='*' "$(dirname "${fd}")/" "${VULCAN}:$(dirname "${vfd}")/" || fail push "rsync json ${rid}"
  rsync -rlpt --partial --exclude=wandb/ --exclude='validation/*.nii.gz' --exclude='validation/*.npz' --exclude=checkpoint_latest.pth \
    "${fd}/" "${VULCAN}:${vfd}/" || fail push "rsync ${rid} fold ${f}"
  left="$(rsync -rlpt --dry-run --itemize-changes --exclude=wandb/ --exclude='validation/*.nii.gz' --exclude='validation/*.npz' --exclude=checkpoint_latest.pth \
    "${fd}/" "${VULCAN}:${vfd}/" | grep -c '^<f' || true)"
  [ "${left}" = 0 ] || fail push "${rid} fold ${f}: ${left} file(s) differ after push"
  L="$(cd "${fd}" && bash -c "${MD5Q}")"; R="$(ssh "${VULCAN}" "cd ${vfd} && ${MD5Q}")"
  [ "${L}" = "${R}" ] || fail push "${rid} fold ${f}: checkpoint md5 mismatch on Vulcan"
  printf '%s\n%s\n' "$(date -u +%FT%TZ) tassan main ${HEAD_SHA} AugLab ${AL_SHA}" "${L}" | ssh "${VULCAN}" "cat > ${vfd}/.lowdata_synced_from_tassan"
  ssh "${VULCAN}" "mkdir -p ${VSTAGE}/panseg_logs/${rid}"
  rsync -q "${LOGS}/nnunet_pansegdata_${tc}_lowdata_n05_${arm}/fold${f}.log" "${VULCAN}:${VSTAGE}/panseg_logs/${rid}/" || true
  SET_DONE[${s}]+="${arm}${f} "
done
for s in "${!SET_DONE[@]}"; do
  echo "${s} folds ${SET_DONE[${s}]}pushed + md5-verified $(date -u +%FT%TZ) main ${HEAD_SHA} AugLab ${AL_SHA}" | ssh "${VULCAN}" "cat > ${VSTAGE}/Pancreas_${s}.tassan_done"
done
vlog "push done: ${#UL[@]} fold dirs on Vulcan, md5 OK; wrote $(for s in "${!SET_DONE[@]}"; do printf 'Pancreas_%s.tassan_done ' "${s}"; done)"
