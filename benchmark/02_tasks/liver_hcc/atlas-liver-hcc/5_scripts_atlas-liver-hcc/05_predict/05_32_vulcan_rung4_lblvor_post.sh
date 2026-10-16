#!/bin/bash
# Post-training controller (Vulcan) for the atlas-liver-hcc rung-4 lblvor retrain: predict + evaluate atlas own t1w and the
# three cross-dataset companions (lld-mmri-hcc, lld-mmri-malignant, liverhccseg), then audit case sets against the OLD
# val100 rung-5 metrics. Queued by 05_31 behind the 3 fold jobs (afterany). Modelled on scripts/cluster/rung5_val000/post_run.sh.
# Status: $SCRATCH/rung5_val000/post/<RUN_ID>.status (last line DONE / FAILED).   Usage: bash 05_30_...sh <RUN_ID>
set -uo pipefail
RID="${1:?usage: $0 <RUN_ID>}"
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project; cd "${ROOT}"
AR=benchmark/02_tasks/liver_hcc; A="${AR}/atlas-liver-hcc"
OUTD="${SCRATCH:?}/rung5_val000/post"; mkdir -p "${OUTD}"; ST="${OUTD}/${RID}.status"
export RUN_JOB_ACCOUNT=aip-jcohen RUN_JOB_GPU_TYPE=l40s RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-rack02-06}"
unset RUN_JOB_DEPENDENCY RUN_JOB_INLINE
log() { echo "[$(date '+%F %T')] $*" | tee -a "${ST}"; }
FAIL=0; fail() { log "ERROR: $*"; FAIL=1; }
log "START ${RID} host=$(hostname) job=${SLURM_JOB_ID:-none}"
CAT=auglab; RDIR="${A}/8_results_atlas-liver-hcc/01_predictions/atlas_liver_hcc_model/t1w/${CAT}/${RID}"
for k in 0 1 2; do ls "${RDIR}"/Dataset*/*/fold_${k}/checkpoint_best.pth >/dev/null 2>&1 || { log "FAILED: no checkpoint_best fold ${k} -- training incomplete; resume with the same RUN_ID, then re-run"; exit 2; }; done
log "checkpoints OK"
OLD=auglab_atlas-liver-hcc_t1w_baseline_kmeans_label_remap_voronoi_20260829_165445
predict() { local tag; tag="$(basename "$(dirname "$(dirname "$2")")")"; "$@" > "${OUTD}/${RID}.predict_${tag}.log" 2>&1 || fail "predict: $*"; }
audit() { "${ROOT}/.venv/bin/python" scripts/cluster/rung5_val000/audit.py "$1" "$2" "$3" 2>&1 | tee -a "${ST}"; [ "${PIPESTATUS[0]}" = 0 ] || FAIL=1; }

# own (flat metrics dir, no old reference: 12 held-out patients)
predict bash "${A}/5_scripts_atlas-liver-hcc/05_predict/05_10_predict_baseline_kmeans_label_remap_voronoi_lblvor.sh" "${RID}"
( export RUN_JOB_INLINE=1; bash "${A}/5_scripts_atlas-liver-hcc/06_evaluate/06_01_evaluate_run.sh" "${RID}" "${CAT}" ) 2>&1 | tail -3 | tee -a "${ST}"
audit "expect:t1w=12" "${A}/8_results_atlas-liver-hcc/02_metrics/atlas_liver_hcc_model/t1w/${CAT}_${RID}" "atlas_own"

# companions (all items rung 4 was scored on; metrics next to rung 4 in ablations/)
for d in lld-mmri-hcc lld-mmri-malignant liverhccseg; do
  S="${AR}/${d}/5_scripts_${d}"; M="${AR}/${d}/8_results_${d}/02_metrics/atlas_liver_hcc_model/t1w"
  predict bash "${S}/05_predict/05_14_predict_atlas_baseline_kmeans_label_remap_voronoi_lblvor.sh" "${RID}"
  ( export RUN_JOB_INLINE=1 CATEGORY="${CAT}" METRICS_SUBDIR=ablations EVAL_MEM=150G EVAL_TIME=3:00:00; bash "${S}/06_evaluate/06_01_evaluate_run.sh" "${RID}" ) 2>&1 | tail -3 | tee -a "${ST}"
  audit "${M}/ablations/${OLD}" "${M}/ablations/${CAT}_${RID}" "${d}"
done
if [ "${FAIL}" = 0 ]; then log "DONE ${RID}"; else log "FAILED ${RID} (see errors above)"; exit 1; fi
