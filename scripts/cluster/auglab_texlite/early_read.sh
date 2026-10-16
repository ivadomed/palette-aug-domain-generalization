#!/usr/bin/env bash
# EARLY READ of the reduced-Auglab arms on TamIA (2026-10-16): predict + evaluate the CURRENT checkpoint_best.pth of runs that
# are still training (or finished but not yet predicted), from a SNAPSHOT in a throwaway results base -- the real run dirs and
# the pending packs/post jobs are never touched. Uses each setting's own predict wrapper (shared predict_common.sh, checkpoint_best,
# on-harmony's in-job RAS->native resample) and its own evaluate script (on-harmony's 06_01_evaluate_testset.sh refuses
# off-grid predictions), with PREDICTIONS_ROOT / METRICS_ROOT pointed at ${EARLY}/<row>/ (common_env.sh keeps preset roots).
#   bash scripts/cluster/auglab_texlite/early_read.sh prep "keepres edgeonly"   # snapshot + record predict cmds (login node, light)
#   bash scripts/cluster/auglab_texlite/early_read.sh submit                    # ONE whole-node H100 job: all predicts (8 slots
#                                                                               #  over 4 GPUs) -> evaluate -> count audit
# Output: ${EARLY}/<arm>_<row>/{01_predictions,02_metrics,best_epochs.txt}; status ${EARLY}/job_<id>.out
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH=/scratch/p/paulh
EARLY="${EARLY:-${SCRATCH}/review_20261016/arms_early}"; mkdir -p "${EARLY}"
T=benchmark/02_tasks
rows_for_arm() {   # prints the run-table rows of one arm with the resolved RUN_ID in field 9
  ( export TEXLITE_ARM="$1"; source scripts/cluster/tamia_texlite_runs.sh
    for r in "${RUNG_ROWS[@]}"; do IFS='|' read -r name pack dir envf tenv tc wrapper args prefix pexp ekind eexp <<<"$r"
      run="$(rung_resolve_run "${pack}" "${prefix}")" || continue
      echo "$1|${name}|${dir}|${envf}|${tenv}|${tc}|${wrapper}|${args}|${run}|${pexp}|${ekind}|${eexp}"; done )
}
envsetup() { export TRAINING_CONTRAST="$1"; source "$T/$2/00_utils/$3"; source "scripts/cluster/$4"; export TRAINING_CONTRAST="$1"; }
case "${1:?prep|submit|_job}" in
prep)
  : > "${EARLY}/ROWS"
  for arm in ${2:?arms}; do rows_for_arm "${arm}" >> "${EARLY}/ROWS"; done
  while IFS='|' read -r arm name dir envf tenv tc wrapper args run pexp ekind eexp; do
    tag="${arm}_${name}"; P="${EARLY}/${tag}/01_predictions"; M="${EARLY}/${tag}/02_metrics"; PK="${EARLY}/${tag}/pack"
    [ -f "${PK}/index.tsv" ] && { echo "[prep] ${tag}: already prepared"; continue; }
    ( envsetup "${tc}" "${dir}" "${envf}" "${tenv}"
      SRC="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${tc}/auglab/${run}"; DST="${P}/${MODEL_TYPE}/${tc}/auglab/${run}"
      [ -d "${SRC}" ] || { echo "[prep] ${tag}: no run dir ${SRC}"; exit 1; }
      mkdir -p "${EARLY}/${tag}"; : > "${EARLY}/${tag}/best_epochs.txt"
      for td in "${SRC}"/Dataset*/*__nnUNetPlans__3d_fullres; do
        rel="${td#${SRC}/}"; mkdir -p "${DST}/${rel}"
        cp "${td}/plans.json" "${td}/dataset.json" "${DST}/${rel}/"; [ -f "${td}/dataset_fingerprint.json" ] && cp "${td}/dataset_fingerprint.json" "${DST}/${rel}/"
        for k in 0 1 2; do
          mkdir -p "${DST}/${rel}/fold_${k}"
          for try in 1 2 3; do cp "${td}/fold_${k}/checkpoint_best.pth" "${DST}/${rel}/fold_${k}/checkpoint_best.pth" \
              && cmp -s "${td}/fold_${k}/checkpoint_best.pth" "${DST}/${rel}/fold_${k}/checkpoint_best.pth" && break; sleep 20; done
          cmp -s "${td}/fold_${k}/checkpoint_best.pth" "${DST}/${rel}/fold_${k}/checkpoint_best.pth" || { echo "[prep] ${tag} fold${k}: copy unstable"; exit 1; }
          python3 - "${td}/fold_${k}" "${k}" "${td}/fold_${k}/checkpoint_final.pth" >> "${EARLY}/${tag}/best_epochs.txt" <<'PY'
import glob, re, sys, os
cur = best = None
for L in sorted(glob.glob(sys.argv[1] + "/training_log_*.txt")):
    for line in open(L, errors="ignore"):
        m = re.search(r"Epoch (\d+)\s*$", line)
        if m: cur = int(m.group(1))
        if "New best" in line: best = cur
print(f"fold{sys.argv[2]} best_at={best} current={cur} final={'yes' if os.path.exists(sys.argv[3]) else 'no'}")
PY
        done
      done
      case "${ekind}" in onh*) mkdir -p "${P}/${MODEL_TYPE}"; ln -sfn "${PREDICTIONS_ROOT}/${MODEL_TYPE}/_test_set" "${P}/${MODEL_TYPE}/_test_set" ;; esac
      export PREDICTIONS_ROOT="${P}" METRICS_ROOT="${M}"; mkdir -p "${M}" "${PK}"
      # shellcheck disable=SC2086
      RUN_JOB_PACK_DIR="${PK}" bash "$T/${dir}/05_predict/${wrapper}" "${run}" ${args} > "${EARLY}/${tag}/record.log" 2>&1 \
        || { echo "[prep] ${tag}: record failed (see record.log)"; exit 1; }
      n=$(grep -c . "${PK}/index.tsv"); bad=0
      while IFS=$'\t' read -r c _r; do grep -q "${EARLY}/${tag}/01_predictions" "$c" || bad=1; done < "${PK}/index.tsv"
      echo "[prep] ${tag}: run ${run}; ${n} fold-predict cmds (expect 3), all into the throwaway base: $([ ${bad} = 0 ] && echo yes || echo NO)"
      sed 's/^/        /' "${EARLY}/${tag}/best_epochs.txt" ) || echo "[prep] ${tag}: FAILED"
  done < "${EARLY}/ROWS" ;;
submit)
  sbatch --parsable --job-name=arms_early_read --account=aip-jcohen --nodes=1 --gpus-per-node=h100:4 --cpus-per-task=48 --mem=0 \
    --time=03:00:00 --output="${EARLY}/job_%j.out" --wrap="bash /project/aip-jcohen/paulh/mri_synthesis_project/scripts/cluster/auglab_texlite/early_read.sh _job" ;;
_job)
  mapfile -t CMDS < <(cat "${EARLY}"/*/pack/index.tsv | cut -f1)
  N=${#CMDS[@]}; SLOTS=8; echo "[early] host=$(hostname) ${N} fold-predict cmds, ${SLOTS} slots on 4 GPUs"
  worker() { local i="$1"; while [ "$i" -lt "$N" ]; do local c="${CMDS[$i]}"
      CUDA_VISIBLE_DEVICES=$(( $1 % 4 )) bash "$c" > "${c%.sh}.log" 2>&1; echo "[early] $(basename "$c") exit=$?"; i=$(( i + SLOTS )); done; }
  for s in $(seq 0 $((SLOTS-1))); do worker "$s" & done; wait
  export RUN_JOB_INLINE=1
  while IFS='|' read -r arm name dir envf tenv tc wrapper args run pexp ekind eexp; do
    tag="${arm}_${name}"; S="$T/${dir}/06_evaluate"
    ( envsetup "${tc}" "${dir}" "${envf}" "${tenv}"; export PREDICTIONS_ROOT="${EARLY}/${tag}/01_predictions" METRICS_ROOT="${EARLY}/${tag}/02_metrics" METRICS_SUBDIR=""
      case "${ekind}" in
        onh_best) for F in 0 1 2; do bash "$S/06_01_evaluate_testset.sh" "${run}" "${F}" || exit 1; done ;;
        chaos:*)  export DATASET_ID="${ekind#chaos:}" CATEGORY=auglab; bash "$S/06_01_evaluate_run.sh" "${run}" ;;
        openms_flair) bash "$S/06_01_evaluate_run.sh" "${run}" auglab ;;
      esac ) > "${EARLY}/${tag}/eval.log" 2>&1; echo "[early] eval ${tag} rc=$?"
    .venv/bin/python - "${EARLY}/${tag}/02_metrics" "${run}" ${eexp} <<'PY'
import csv, glob, sys, collections
root, run, exp = sys.argv[1], sys.argv[2], dict(x.rsplit(":", 1) for x in sys.argv[3:]); bad = 0
for k in (0, 1, 2):
    ps = glob.glob(f"{root}/**/auglab_{run}/fold{k}/eval_all.csv", recursive=True)
    if len(ps) != 1: print(f"  AUDIT FAIL fold{k}: {len(ps)} eval_all.csv"); bad = 1; continue
    c = collections.defaultdict(set)
    for r in csv.DictReader(open(ps[0])): c[r.get("group") or r.get("contrast")].add(r["case"])
    for it, n in exp.items():
        if len(c.get(it, ())) != int(n): print(f"  AUDIT FAIL fold{k} {it}: {len(c.get(it, ()))}/{n}"); bad = 1
print(f"  {run}: AUDIT {'OK' if not bad else 'FAILED'}")
PY
  done < "${EARLY}/ROWS"
  echo "[early] DONE" ;;
esac
