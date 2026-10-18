#!/bin/bash
# Vulcan CPU EVAL job of ONE ssem50 companion (afterok behind its 3 predict jobs), submitted by scripts/cluster/ssem50/companions_vulcan.sh.
# Runs the companion's OWN standard evaluate script (run_job inline inside this job) with the explicit ssem50 RUN_ID, flat layout next to
# the headline runs (what the source task's combined config reads), then AUDITS each item / fold eval_all.csv: distinct cases == the
# companion's labelsTs_<item> count (scripts/cluster/ssem50/companion_audit.py).
#   vulcan_companion_eval_job.sh <companion> <RUN_ID> <training contrast> <item...>
#SBATCH --account=aip-jcohen
set -uo pipefail
CN="${1:?companion}"; RID="${2:?run id}"; TC="${3:?training contrast}"; shift 3; ITEMS=("$@")
cd /project/aip-jcohen/paulh/mri_synthesis_project
export RUN_JOB_INLINE=1
case "${CN}" in
  duke-breast-mri|ispy1|acrin6698) S=benchmark/02_tasks/breast_cancer/${CN}/5_scripts_${CN}; SRCM=ispy2_model ;;
  totalsegmri-pancreas|msd-pancreas) S=benchmark/02_tasks/pancreas_disease/${CN}/5_scripts_${CN}; SRCM=pansegdata_model ;;
  *) echo "[ssem50c-eval] unknown companion ${CN}"; exit 2 ;;
esac
echo "[ssem50c-eval] ${CN} run=${RID} tc=${TC} items=${ITEMS[*]} host=$(hostname) job=${SLURM_JOB_ID:-?}"
rc=0
case "${CN}" in
  duke-breast-mri)   # one item per call (DUKE_ITEM), metrics in <tc>/<item>/ (the *_uniap layout the ispy2 combined config reads)
    for it in "${ITEMS[@]}"; do DUKE_ITEM="${it}" METRICS_SUBDIR="${it}" bash "${S}/06_evaluate/06_01_evaluate_ispy2_run.sh" "${RID}" auglab "${TC}" all || rc=1; done ;;
  ispy1|acrin6698)
    ITEMS_OVERRIDE="${ITEMS[*]}" LADDER=0 bash "${S}/06_evaluate/06_01_evaluate_run.sh" "${RID}" auglab "${TC}" all || rc=1 ;;
  *)
    EVAL_INLINE=1 LADDER=0 bash "${S}/06_evaluate/06_01_evaluate_run.sh" "${RID}" auglab "${TC}" all || rc=1 ;;
esac
arc=0
read -r MR RAW < <( unset METRICS_ROOT nnUNet_raw PREDICTIONS_ROOT; source "${S}/00_utils/env.sh" >/dev/null 2>&1; echo "${METRICS_ROOT} ${nnUNet_raw}" )
for it in "${ITEMS[@]}"; do
  echo "[ssem50c-eval] audit ${it}"
  python3 -I scripts/cluster/ssem50/companion_audit.py "${MR}/${SRCM}/${TC}/${it}/auglab_${RID}" --gt "${RAW}" --items "${it}" || arc=1
done
echo "[ssem50c-eval] done eval_rc=${rc} audit_rc=${arc}"
[ "${rc}" = 0 ] && [ "${arc}" = 0 ]
