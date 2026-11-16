#!/bin/bash
# In-domain re-prediction (research/dg_mechanisms/plans/2026-11-16-repredict-indomain.md, tag reg-T10-T28-20261116):
# the in-domain test item (= the training contrast) of the (setting, arm) pairs whose in-domain test predictions are absent on
# Vulcan and TamIA, folds 0-2, GPU inference only (checkpoint_best, no TTA = the shared driver's defaults).
# Same machinery as T04 deviation 12 (../t04_arm_boundary/t04_repredict.sh, left unchanged): each dataset's own 05_predict
# wrapper -> predict_common.sh with
#   PREDICT_OUTPUT_BASE = STAGE mirror of the wrapper's model-category dir (never 8_results_*/01_predictions or 02_metrics)
#   RUN_JOB_PACK_DIR    = one pack dir per (setting, arm, fold): the driver RECORDS its nnUNetv2_predict command;
# the recorded commands are then submitted as per-GPU L40S jobs through run_job (no partition, explicit time/mem).
# Differences from t04_repredict.sh: ITEMS = the training contrast only; run ids from t04_arms.run_ids (the paper's headline
# configs; T04's roster lists OOD items only); its own STAGE / WORK under /scratch/paulh/dg_mechanisms/indomain/.
# Pelvis models are one run dir per fold (<m>_f<k>_<ts>); the collapsed id (<m>_<ts>/fold<k>) gets a relative symlink, as T04.
#
#   bash repredict_indomain.sh record <setting> <arm ...>       record (no compute)
#   bash repredict_indomain.sh submit <name> <time> <cmdfile...> one GPU job running the recorded commands in turn (prints job id)
#   bash repredict_indomain.sh cmds <setting> <arm ...>          list the recorded cmdfiles
#   bash repredict_indomain.sh audit                             staged .nii.gz count per setting x arm x fold (in-domain item)
# Settings / arms in scope (the plan): Glioma_t1c, Glioma_t2f: synthseg_noEM baseline; Pelvis_ct, Pelvis_mri: synthseg_noEM.
# (Mandible_cbct synthseg_noEM: md5-verified copy from TamIA into the same STAGE, no prediction; Pelvis_mri palette: first-pass
# files already on Vulcan.)
set -euo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
STAGE=/scratch/paulh/dg_mechanisms/indomain/staged_predictions
WORK=/scratch/paulh/dg_mechanisms/indomain/repredict
T04=${ROOT}/benchmark/00_commun_scripts/00_04_analysis/dg_mechanisms/t04_arm_boundary
export RUN_JOB_ACCOUNT=aip-jcohen
cd "${ROOT}"

setting_info() {
  case "$1" in
    Glioma_t1c) DS=benchmark/02_tasks/brain_tumor/brats2024-glioma; DSN=brats2024-glioma; MT=brats2024_glioma_model; TC=t1c; PERFOLD=0 ;;
    Glioma_t2f) DS=benchmark/02_tasks/brain_tumor/brats2024-glioma; DSN=brats2024-glioma; MT=brats2024_glioma_model; TC=t2f; PERFOLD=0 ;;
    Pelvis_ct)  DS=benchmark/02_tasks/pelvis_healthy/totalseg-pelvic; DSN=totalseg-pelvic; MT=totalseg_pelvic_model; TC=ct; PERFOLD=1 ;;
    Pelvis_mri) DS=benchmark/02_tasks/pelvis_healthy/totalseg-pelvic; DSN=totalseg-pelvic; MT=totalseg_pelvic_model; TC=mri; PERFOLD=1 ;;
    *) echo "unknown setting $1" >&2; exit 2 ;;
  esac
  PRED=${DS}/5_scripts_${DSN}/05_predict
  ITEMS=${TC}
}
# (setting, arm) -> the dataset's headline predict wrapper for that method (same map as t04_repredict.sh)
wrapper() {
  case "$1:$2" in
    Glioma_t1c:baseline) echo 05_70_predict_t1c_baseline.sh ;;
    Glioma_t1c:synthseg_noEM) echo 05_73_predict_t1c_synthseg_noEM.sh ;;
    Glioma_t2f:baseline) echo 05_50_predict_t2f_baseline.sh ;;
    Glioma_t2f:synthseg_noEM) echo 05_53_predict_t2f_synthseg_noEM.sh ;;
    Pelvis_ct:synthseg_noEM) echo 05_04_predict_ct_synthseg_noEM.sh ;;
    Pelvis_mri:synthseg_noEM) echo 05_15_predict_mri_synthseg_noEM.sh ;;
    *) echo "no wrapper for $1:$2 (not in the plan's scope)" >&2; exit 2 ;;
  esac
}
# run id of (setting, arm) from the paper's headline config (t04_arms.run_ids)
run_id() {
  "${ROOT}/.venv/bin/python" - "$1" "$2" <<PY
import sys
sys.path.insert(0, "${T04}")
import t04_arms as TA
print(TA.run_ids(TA.settings()[sys.argv[1]])[sys.argv[2]])
PY
}
wrapper_category() { grep -m1 -E '^CATEGORY="' "$1" | sed -E 's/^CATEGORY="([^"]*)".*/\1/'; }

record() {
  local S="$1"; shift; setting_info "$S"
  local A W CAT RID OB k MRID PD ts
  for A in "$@"; do
    W=${PRED}/$(wrapper "$S" "$A"); [ -f "$W" ] || { echo "missing $W" >&2; exit 2; }
    CAT=$(wrapper_category "$W"); RID=$(run_id "$S" "$A")
    OB=${STAGE}/${DS}/8_results_${DSN}/01_predictions/${MT}/${TC}/${CAT}
    case "${OB}" in "${STAGE}"/*) ;; *) echo "refusing output base outside STAGE: ${OB}" >&2; exit 4 ;; esac
    for k in 0 1 2; do
      if [ "${PERFOLD}" = "1" ]; then ts=${RID: -15}; MRID="${RID%_${ts}}_f${k}_${ts}"; else MRID=${RID}; fi
      ls "${ROOT}/${DS}/8_results_${DSN}/01_predictions/${MT}/${TC}/${CAT}/${MRID}"/Dataset*/*/fold_${k}/checkpoint_best.pth >/dev/null \
        || { echo "MISSING checkpoint_best ${S} ${A} ${MRID} fold ${k}" >&2; exit 5; }
      PD=${WORK}/packs/${S}_${A}_f${k}
      [ -e "${PD}/index.tsv" ] && { echo "already recorded: ${PD}" >&2; continue; }
      echo "[record] ${S} ${A} fold ${k}: ${W##*/} ${MRID} item ${ITEMS} -> ${OB}"
      PREDICT_OUTPUT_BASE="${OB}" RUN_JOB_PACK_DIR="${PD}" bash "${W}" "${MRID}" "${k}" ${ITEMS}
      if [ "${PERFOLD}" = "1" ] && [ "${MRID}" != "${RID}" ]; then
        mkdir -p "${OB}/${RID}"
        [ -e "${OB}/${RID}/fold${k}" ] || ln -s "../${MRID}/fold${k}" "${OB}/${RID}/fold${k}"
      fi
    done
  done
}

cmds() {
  local S="$1"; shift; local A k
  for A in "$@"; do for k in 0 1 2; do cut -f1 "${WORK}/packs/${S}_${A}_f${k}/index.tsv"; done; done
}

submit() {
  local name="$1" time="$2"; shift 2
  mkdir -p "${WORK}/groups" "${WORK}/logs"
  local g="${WORK}/groups/${name}.sh" c
  { echo "#!/bin/bash"; echo "rc=0"; echo "echo HEAD \$(git -C ${ROOT} rev-parse HEAD) host \$(hostname)"
    echo "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true"
    for c in "$@"; do [ -f "$c" ] || { echo "no cmdfile $c" >&2; exit 2; }
      echo "echo \"[\$(date '+%F %T')] start ${c##*/}\"; bash '${c}' || rc=1; echo \"[\$(date '+%F %T')] end ${c##*/} rc=\$rc\""; done
    echo "exit \$rc"; } > "${g}"
  source "${ROOT}/scripts/job_runner/run_job.sh"
  local out; out=$(run_job --name "ind_${name}" --gpus 1 --cpus 8 --mem 64G --time "${time}" --log "${WORK}/logs/${name}.log" -- bash "${g}")
  echo "${out}"; echo "$(date '+%F %T') ${name} ${out##* } ${time} $*" >> "${WORK}/submitted.txt"
}

audit() {
  local S A k CAT RID OB n
  printf 'setting\tarm\tfold\titem\tn_nii\tdir\n'
  for SA in Glioma_t1c:synthseg_noEM Glioma_t1c:baseline Glioma_t2f:synthseg_noEM Glioma_t2f:baseline Pelvis_ct:synthseg_noEM Pelvis_mri:synthseg_noEM; do
    S=${SA%%:*}; A=${SA#*:}; setting_info "$S"
    CAT=$(wrapper_category "${PRED}/$(wrapper "$S" "$A")"); RID=$(run_id "$S" "$A")
    OB=${STAGE}/${DS}/8_results_${DSN}/01_predictions/${MT}/${TC}/${CAT}
    for k in 0 1 2; do
      n=$(ls "${OB}/${RID}/fold${k}/${TC}"/*.nii.gz 2>/dev/null | wc -l)
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$S" "$A" "$k" "$TC" "$n" "${OB}/${RID}/fold${k}/${TC}"
    done
  done
  local M=${STAGE}/benchmark/02_tasks/mandible_healthy/toothfairy2/8_results_toothfairy2/01_predictions/toothfairy2_model/cbct/auglab/toothfairy2_cbct_synthseg_noEM_20260908_013025
  for k in 0 1 2; do
    printf 'Mandible_cbct\tsynthseg_noEM\t%s\tcbct\t%s\t%s\n' "$k" "$(ls "${M}/fold${k}/cbct"/*.nii.gz 2>/dev/null | wc -l)" "${M}/fold${k}/cbct"
  done
}

case "${1:-}" in
  record) shift; record "$@" ;;
  cmds) shift; cmds "$@" ;;
  submit) shift; submit "$@" ;;
  audit) audit ;;
  *) echo "usage: record|cmds|submit|audit" >&2; exit 2 ;;
esac
