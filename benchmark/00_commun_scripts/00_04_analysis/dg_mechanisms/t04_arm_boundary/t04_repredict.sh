#!/bin/bash
# T04 deviation 12 (research/dg_mechanisms/plans/2026-11-14-T04-repredict-deviation.md): re-predict the headline-arm test
# predictions that are absent on both clusters (T04 roster.tsv, how = absent) for Glioma t1c / t2f and Pelvis ct / mri,
# into the STAGE tree that t04_arms.pred_dir reads (how = staged). Never writes into 8_results_*/01_predictions or 02_metrics.
#
# Predictions go through each dataset's own 05_predict wrapper -> the shared predict_common.sh, with
#   PREDICT_OUTPUT_BASE = STAGE mirror of the wrapper's model-category dir (outputs + _logs land there; model read from RUN_DIR)
#   RUN_JOB_PACK_DIR    = one pack dir per (setting, arm, fold): the driver RECORDS its nnUNetv2_predict command
#                         (checkpoint_best, --disable_tta, the dataset's imagesTs_<item>) instead of submitting it;
# the recorded commands are then submitted here as per-GPU L40S jobs through run_job (no --wait, no partition).
# Pelvis models are one run dir per fold (<m>_f<k>_<ts>); the roster reads the collapsed id (<m>_<ts>/fold<k>), so a relative
# symlink <m>_<ts>/fold<k> -> ../<m>_f<k>_<ts>/fold<k> is added in STAGE.
#
#   bash t04_repredict.sh record <setting> [arm ...]        record (no compute; creates the staged dirs + pelvis symlinks)
#   bash t04_repredict.sh submit <name> <time> <cmdfile...> one GPU job running the recorded commands one after the other
#   bash t04_repredict.sh cmds <setting> [arm ...]          list the recorded cmdfiles
#   bash t04_repredict.sh audit                             staged .nii.gz count per setting x arm x fold x item
# Settings: Glioma_t1c Glioma_t2f Pelvis_ct Pelvis_mri. Arms: baseline auglab_default synthseg_EM synthseg_noEM pa_auglab.
set -euo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
STAGE=/scratch/paulh/dg_mechanisms/T04/staged_predictions
WORK=/scratch/paulh/dg_mechanisms/T04/repredict
ROSTER=${ROOT}/benchmark/01_commun_results/dg_mechanisms/T04/roster.tsv
export RUN_JOB_ACCOUNT=aip-jcohen
cd "${ROOT}"

ARMS="baseline auglab_default synthseg_EM synthseg_noEM pa_auglab"
# setting -> dataset dir (repo-relative), 05_predict dir, model type, training contrast, OOD items (registered E1 items)
setting_info() {
  case "$1" in
    Glioma_t1c) DS=benchmark/02_tasks/brain_tumor/brats2024-glioma; DSN=brats2024-glioma; MT=brats2024_glioma_model; TC=t1c; ITEMS="t1n t2w t2f"; PERFOLD=0 ;;
    Glioma_t2f) DS=benchmark/02_tasks/brain_tumor/brats2024-glioma; DSN=brats2024-glioma; MT=brats2024_glioma_model; TC=t2f; ITEMS="t1c t1n t2w"; PERFOLD=0 ;;
    Pelvis_ct)  DS=benchmark/02_tasks/pelvis_healthy/totalseg-pelvic; DSN=totalseg-pelvic; MT=totalseg_pelvic_model; TC=ct; ITEMS="mri"; PERFOLD=1 ;;
    Pelvis_mri) DS=benchmark/02_tasks/pelvis_healthy/totalseg-pelvic; DSN=totalseg-pelvic; MT=totalseg_pelvic_model; TC=mri; ITEMS="ct"; PERFOLD=1 ;;
    *) echo "unknown setting $1" >&2; exit 2 ;;
  esac
  PRED=${DS}/5_scripts_${DSN}/05_predict
}
# (setting, arm) -> predict wrapper (the dataset's headline wrapper for that method)
wrapper() {
  case "$1:$2" in
    Glioma_t1c:baseline) echo 05_70_predict_t1c_baseline.sh ;;
    Glioma_t1c:auglab_default) echo 05_71_predict_t1c_auglab_default.sh ;;
    Glioma_t1c:synthseg_EM) echo 05_72_predict_t1c_synthseg_EM.sh ;;
    Glioma_t1c:synthseg_noEM) echo 05_73_predict_t1c_synthseg_noEM.sh ;;
    Glioma_t1c:pa_auglab) echo 05_75_predict_t1c_auglabAug_v26_6_2_train050_val000_dualval.sh ;;
    Glioma_t2f:baseline) echo 05_50_predict_t2f_baseline.sh ;;
    Glioma_t2f:auglab_default) echo 05_51_predict_t2f_auglab_default.sh ;;
    Glioma_t2f:synthseg_EM) echo 05_52_predict_t2f_synthseg_EM.sh ;;
    Glioma_t2f:synthseg_noEM) echo 05_53_predict_t2f_synthseg_noEM.sh ;;
    Glioma_t2f:pa_auglab) echo 05_55_predict_t2f_auglabAug_v26_6_2_train050_val000_dualval.sh ;;
    Pelvis_ct:baseline) echo 05_02_predict_ct_baseline.sh ;;
    Pelvis_ct:auglab_default) echo 05_03_predict_ct_auglab_default.sh ;;
    Pelvis_ct:synthseg_EM) echo 05_05_predict_ct_synthseg_EM.sh ;;
    Pelvis_ct:synthseg_noEM) echo 05_04_predict_ct_synthseg_noEM.sh ;;
    Pelvis_ct:pa_auglab) echo 05_07_predict_ct_auglabAug_v26_6_2_val000.sh ;;
    Pelvis_mri:baseline) echo 05_13_predict_mri_baseline.sh ;;
    Pelvis_mri:auglab_default) echo 05_14_predict_mri_auglab_default.sh ;;
    Pelvis_mri:synthseg_EM) echo 05_16_predict_mri_synthseg_EM.sh ;;
    Pelvis_mri:synthseg_noEM) echo 05_15_predict_mri_synthseg_noEM.sh ;;
    Pelvis_mri:pa_auglab) echo 05_18_predict_mri_auglabAug_v26_6_2_val000.sh ;;
    *) echo "no wrapper for $1:$2" >&2; exit 2 ;;
  esac
}
# roster run id of (setting, arm), required to be listed as absent (the deviation's exact scope)
roster_rid() {
  local r; r=$(awk -F'\t' -v s="$1" -v a="$2" 'NR>1 && $1==s && $2==a && $9 ~ /^absent/ {print $3}' "${ROSTER}" | sort -u)
  [ "$(printf '%s\n' "$r" | grep -c .)" = "1" ] || { echo "roster: $1 $2 not exactly one absent run id: '$r'" >&2; exit 3; }
  echo "$r"
}
wrapper_category() { grep -m1 -E '^CATEGORY="' "$1" | sed -E 's/^CATEGORY="([^"]*)".*/\1/'; }

record() {
  local S="$1"; shift; setting_info "$S"
  local arms="${*:-${ARMS}}" A W CAT RID OB k MRID PD
  for A in ${arms}; do
    W=${PRED}/$(wrapper "$S" "$A"); [ -f "$W" ] || { echo "missing $W" >&2; exit 2; }
    CAT=$(wrapper_category "$W"); RID=$(roster_rid "$S" "$A")
    OB=${STAGE}/${DS}/8_results_${DSN}/01_predictions/${MT}/${TC}/${CAT}
    case "${OB}" in "${STAGE}"/*) ;; *) echo "refusing output base outside STAGE: ${OB}" >&2; exit 4 ;; esac
    for k in 0 1 2; do
      if [ "${PERFOLD}" = "1" ]; then local ts=${RID: -15}; MRID="${RID%_${ts}}_f${k}_${ts}"; else MRID=${RID}; fi
      ls "${ROOT}/${DS}/8_results_${DSN}/01_predictions/${MT}/${TC}/${CAT}/${MRID}"/Dataset*/*/fold_${k}/checkpoint_best.pth >/dev/null \
        || { echo "MISSING checkpoint_best ${S} ${A} ${MRID} fold ${k}" >&2; exit 5; }
      PD=${WORK}/packs/${S}_${A}_f${k}
      [ -e "${PD}/index.tsv" ] && { echo "already recorded: ${PD}" >&2; continue; }
      echo "[record] ${S} ${A} fold ${k}: ${W##*/} ${MRID} items ${ITEMS} -> ${OB}"
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
  for A in ${*:-${ARMS}}; do for k in 0 1 2; do cut -f1 "${WORK}/packs/${S}_${A}_f${k}/index.tsv"; done; done
}

submit() {
  local name="$1" time="$2"; shift 2
  mkdir -p "${WORK}/groups" "${WORK}/logs"
  local g="${WORK}/groups/${name}.sh" c
  { echo "#!/bin/bash"; echo "rc=0"; echo "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true"
    for c in "$@"; do [ -f "$c" ] || { echo "no cmdfile $c" >&2; exit 2; }
      echo "echo \"[\$(date '+%F %T')] start ${c##*/}\"; bash '${c}' || rc=1; echo \"[\$(date '+%F %T')] end ${c##*/} rc=\$rc\""; done
    echo "exit \$rc"; } > "${g}"
  source "${ROOT}/scripts/job_runner/run_job.sh"
  local out; out=$(run_job --name "t04r_${name}" --gpus 1 --cpus 8 --mem 64G --time "${time}" --log "${WORK}/logs/${name}.log" -- bash "${g}")
  echo "${out}"; echo "$(date '+%F %T') ${name} ${out##* } ${time} $*" >> "${WORK}/submitted.txt"
}

audit() {
  local S A k it CAT RID OB n
  printf 'setting\tarm\tfold\titem\tn_nii\n'
  for S in Glioma_t1c Glioma_t2f Pelvis_ct Pelvis_mri; do setting_info "$S"
    for A in ${ARMS}; do
      CAT=$(wrapper_category "${PRED}/$(wrapper "$S" "$A")"); RID=$(roster_rid "$S" "$A")
      OB=${STAGE}/${DS}/8_results_${DSN}/01_predictions/${MT}/${TC}/${CAT}
      for k in 0 1 2; do for it in ${ITEMS}; do
        n=$(ls "${OB}/${RID}/fold${k}/${it}"/*.nii.gz 2>/dev/null | wc -l)
        printf '%s\t%s\t%s\t%s\t%s\n' "$S" "$A" "$k" "$it" "$n"
      done; done
    done
  done
}

case "${1:-}" in
  record) shift; record "$@" ;;
  cmds) shift; cmds "$@" ;;
  submit) shift; submit "$@" ;;
  audit) audit ;;
  *) echo "usage: record|cmds|submit|audit" >&2; exit 2 ;;
esac
