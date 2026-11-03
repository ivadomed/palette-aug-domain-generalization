#!/bin/bash
# Step 5 (Vulcan): score the returned predictions of one or more tasks with each dataset's OWN evaluator (the same
# calls as scripts/cluster/rung45_flat/post_run.sh), for both conditions:
#   texrm  = texture-removed item   (<run>/fold{k}/texrm/<item>)
#   origts = originals re-predicted on the same machine (<run>/fold{k}/origts/<item>)
# through the evaluators' CKPT_TAG hook (reads fold{k}/<tag>/<item>, writes <category>_<RUN_ID>_<tag> next to the
# original metrics; on-harmony: CHECKPOINT=checkpoint_<tag>.pth). Mandible's evaluators have no CKPT_TAG: they read
# PRED_SUBDIR=<tag>/sif and write <metrics>/texture_removal_<tag>/<category>_<RUN_ID>.
# Evaluators with a hard-coded item list also score the in-domain item: fold{k}/<tag>/<item> is linked to the ORIGINAL
# prediction for those (tasks.yaml passthrough) -- never read by the analysis.
# One run_job CPU controller per task, evaluators inline inside it.
#   bash 05_evaluate.sh <task|all> [tags]          (tags default "origts texrm")
set -euo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
HERE="${ROOT}/benchmark/00_commun_scripts/00_04_analysis/texture_removal"
OUT=/scratch/paulh/texture_removal_test; mkdir -p "${OUT}/logs"
TASKS="${1:?task|all}"; TAGS="${2:-origts texrm}"
[ "${TASKS}" = all ] && TASKS="MS Glioma Breast HCC Abdomen Brain Mandible Pelvis Pancreas"
cd "${ROOT}"

if [ -z "${TEXRM_EVAL_WORKER:-}" ]; then
  source scripts/job_runner/run_job.sh; export RUN_JOB_ACCOUNT="${RUN_JOB_ACCOUNT:-aip-jcohen}"
  for task in ${TASKS}; do
    run_job --name "texrm_eval_${task}" --gpus 0 --cpus 8 --mem 64G --time 04:00:00 \
      --log "${OUT}/logs/eval_${task}.log" -- \
      env TEXRM_EVAL_WORKER=1 RUN_JOB_INLINE=1 bash "${HERE}/05_evaluate.sh" "${task}" "${TAGS}"
    sleep 1
  done
  exit 0
fi

# ---------------------------------------------------------------- worker (inside the controller job) -------------
T=benchmark/02_tasks
run() { local m="$1"; .venv/bin/python - "$HERE/tasks.yaml" "$task" "$m" <<'EOF'
import sys, yaml, glob
t = yaml.safe_load(open(sys.argv[1]))["tasks"][sys.argv[2]]["models"][sys.argv[3]]
print(t["run_id"], t["category"], t["run_dir"])
EOF
}
passthrough() {   # passthrough <run_dir> <tag> <item...>: link fold{k}/<tag>/<item> -> ../<item> (original predictions)
  local rd="$1" tag="$2"; shift 2
  for k in 0 1 2; do for it in "$@"; do
    [ -d "${rd}/fold${k}/${it}" ] || { echo "ERROR: no original predictions ${rd}/fold${k}/${it}" >&2; exit 1; }
    mkdir -p "${rd}/fold${k}/${tag}"
    [ -e "${rd}/fold${k}/${tag}/${it}" ] || ln -s "../${it}" "${rd}/fold${k}/${tag}/${it}"
  done; done
}
fail=0
for task in ${TASKS}; do for tag in ${TAGS}; do for m in flat texture; do
  read -r RID CAT RDIR < <(run "$m")
  echo "[$(date '+%F %T')] evaluate ${task} ${m} ${tag} ${RID}"
  (
  set -e
  case "${task}" in
    MS)
      passthrough "${RDIR}" "${tag}" flair
      export METRICS_SUBDIR=ablations CKPT_TAG="${tag}"
      bash ${T}/brain_ms/open-ms/5_scripts_open-ms/06_evaluate/06_01_evaluate_run.sh "${RID}" "${CAT}" ;;
    Glioma)
      passthrough "${RDIR}" "${tag}" t1n
      B=${T}/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma
      export TRAINING_CONTRAST=t1n; source ${B}/00_utils/env.sh
      export TRAINING_CONTRAST=t1n DATASET_ID=051 EVAL_INLINE=1 CATEGORY="${CAT}" METRICS_SUBDIR=ablations CKPT_TAG="${tag}"
      for F in 0 1 2; do bash ${B}/06_evaluate/06_01_evaluate_run.sh "${RID}" "${F}"; done ;;
    Pelvis)
      passthrough "${RDIR}" "${tag}" ct
      export TRAINING_CONTRAST=ct METRICS_SUBDIR= CKPT_TAG="${tag}"
      bash ${T}/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic/06_evaluate/06_01_evaluate_run.sh "${RID}" "${CAT}" ;;
    Brain)
      O=${T}/brain_healthy/on-harmony/5_scripts_on-harmony
      source ${O}/00_utils/env.sh; export METRICS_SUBDIR=ablations CHECKPOINT="checkpoint_${tag}.pth"
      for F in 0 1 2; do bash ${O}/06_evaluate/06_01_evaluate_testset.sh "${RID}" "${F}"; done ;;
    Abdomen)
      passthrough "${RDIR}" "${tag}" t1in
      C=${T}/abdomen_healthy/chaos/5_scripts_chaos
      source ${C}/00_utils/env.sh; export DATASET_ID=60 CATEGORY="${CAT}" METRICS_SUBDIR= CKPT_TAG="${tag}"
      bash ${C}/06_evaluate/06_01_evaluate_run.sh "${RID}"
      # amos/sliver07 crops: the shared fov_crop_predict_evaluate.sh (PHASES 3 4) through a scratch mirror whose
      # 01_predictions/<cat>/<RID>_<tag>/fold{k}/<item>_fovcrop/ holds links to the returned fold{k}/<tag>/<item>
      # files and whose 02_metrics is the repo's -> metrics at <ds>/.../t1in/fov_crop/<cat>_<RID>_<tag>
      M="${OUT}/fovmirror"; TR="$(basename "$(ls -d "${RDIR}"/Dataset*/*__nnUNetPlans__3d_fullres)" | sed 's/__.*//')"
      mkdir -p "${M}/chaos"
      ln -sfn "${ROOT}/${T}/abdomen_healthy/chaos/2_nnUNet_chaos" "${M}/chaos/2_nnUNet_chaos"
      ln -sfn "${ROOT}/${T}/abdomen_healthy/chaos/8_results_chaos" "${M}/chaos/8_results_chaos"
      for ds in amos sliver07; do
        A="${ROOT}/${T}/abdomen_healthy/${ds}"
        mkdir -p "${M}/${ds}/8_results_${ds}"
        ln -sfn "${A}/2_nnUNet_${ds}" "${M}/${ds}/2_nnUNet_${ds}"
        ln -sfn "${A}/8_results_${ds}/02_metrics" "${M}/${ds}/8_results_${ds}/02_metrics"
        its="ct"; [ "${ds}" = amos ] && its="ct mri"
        for k in 0 1 2; do for it in ${its}; do
          src="${A}/8_results_${ds}/01_predictions/chaos_model/t1in/${CAT}/${RID}/fold${k}/${tag}/${it}"
          dst="${M}/${ds}/8_results_${ds}/01_predictions/chaos_model/t1in/${CAT}/${RID}_${tag}/fold${k}/${it}_fovcrop"
          mkdir -p "${dst}"; for f in "${src}"/*.nii.gz; do ln -sfn "${f}" "${dst}/"; done
        done; done
        RF="${M}/runs_${ds}_${RID}_${tag}.txt"; echo "t1in|${CAT}|${RID}_${tag}|${TR}|" > "${RF}"
        if [ "${ds}" = amos ]; then X=(DATASET=amos "ITEMS=ct mri" ANCHOR=kidney ANCHOR_IDS=2,3 EVAL_MODE=amos)
        else X=(DATASET=sliver07 ITEMS=ct ANCHOR=liver ANCHOR_IDS=1 EVAL_MODE=generic_labels EVAL_LABELS=liver); fi
        env SCRATCH="${M}" RUNS_FILE="${RF}" CONTRASTS=t1in SKIP_REPORT=1 PHASES="3 4" EVAL_PARALLEL=6 \
          CROP_REUSE_DIR="/scratch/paulh/srcsm_match/fovmirror/${ds}/_fovcrop/vulcanprep" "${X[@]}" \
          bash benchmark/00_commun_scripts/00_02_predict/fov_crop_predict_evaluate.sh
      done ;;
    Mandible)
      MD="${ROOT}/${T}/mandible_healthy"
      HS_PRED=${MD}/hanseg/8_results_hanseg/01_predictions/toothfairy2_model/cbct \
      HS_METRICS=${MD}/hanseg/8_results_hanseg/02_metrics/toothfairy2_model/cbct MO_DIR= FORCE_SUB="texture_removal_${tag}" \
      GT_OVERRIDE=${MD}/hanseg/2_nnUNet_hanseg/raw/labelsTs_ct_sif PRED_SUBDIR="${tag}/sif" ONLY_RUN="${RID}" \
        bash ${MD}/hanseg/5_scripts_hanseg/06_evaluate/06_03_eval_mandible_only.sh
      PD_PRED=${MD}/pddca/8_results_pddca/01_predictions/toothfairy2_model/cbct \
      PD_METRICS=${MD}/pddca/8_results_pddca/02_metrics/toothfairy2_model/cbct MO_DIR= FORCE_SUB="texture_removal_${tag}" \
      GT_OVERRIDE=${MD}/pddca/2_nnUNet_pddca/raw/labelsTs_ct_sif PRED_SUBDIR="${tag}/sif" ONLY_RUN="${RID}" \
        bash ${MD}/pddca/5_scripts_pddca/06_evaluate/06_01_eval_mandible_only.sh ;;
    Pancreas)
      passthrough "${RDIR}" "${tag}" t1wce
      PZ=${T}/pancreas_disease
      ( source ${PZ}/pansegdata/5_scripts_pansegdata/00_utils/env.sh
        export TRAINING_CONTRAST=t1wce METRICS_SUBDIR=ablations CKPT_TAG="${tag}"
        bash ${PZ}/pansegdata/5_scripts_pansegdata/06_evaluate/06_01_evaluate_run.sh "${RID}" "${CAT}" )
      for comp in totalsegmri-pancreas msd-pancreas; do
        LADDER=1 EVAL_MEM=64G EVAL_TIME=3:00:00 CKPT_TAG="${tag}" \
          bash ${PZ}/${comp}/5_scripts_${comp}/06_evaluate/06_01_evaluate_run.sh "${RID}" "${CAT}" t1wce
      done ;;
    Breast)
      passthrough "${RDIR}" "${tag}" t1wce
      BC=${T}/breast_cancer
      ( source ${BC}/ispy2/5_scripts_ispy2/00_utils/env.sh; export METRICS_SUBDIR=ablations CKPT_TAG="${tag}"
        bash ${BC}/ispy2/5_scripts_ispy2/06_evaluate/06_01_evaluate_own_run.sh "${RID}" "${CAT}" 100 all )
      DUKE_ITEM=precontrast_uniap METRICS_SUBDIR=ablations/precontrast_uniap CKPT_TAG="${tag}" \
        bash ${BC}/duke-breast-mri/5_scripts_duke-breast-mri/06_evaluate/06_01_evaluate_ispy2_run.sh "${RID}" "${CAT}" t1wce
      LADDER=1 ITEMS_OVERRIDE=precontrast CKPT_TAG="${tag}" \
        bash ${BC}/ispy1/5_scripts_ispy1/06_evaluate/06_01_evaluate_run.sh "${RID}" "${CAT}" t1wce
      LADDER=1 CKPT_TAG="${tag}" bash ${BC}/acrin6698/5_scripts_acrin6698/06_evaluate/06_01_evaluate_run.sh "${RID}" "${CAT}" t1wce ;;
    HCC)
      export METRICS_SUBDIR=ablations CATEGORY="${CAT}" CKPT_TAG="${tag}"
      bash ${T}/liver_hcc/lld-mmri-hcc/5_scripts_lld-mmri-hcc/06_evaluate/06_01_evaluate_run.sh "${RID}" ;;
    *) echo "unknown task ${task}" >&2; exit 2 ;;
  esac
  ) || { echo "FAILED: ${task} ${m} ${tag}" >&2; fail=1; }
done; done; done
echo "[$(date '+%F %T')] done (fail=${fail})"
exit ${fail}
