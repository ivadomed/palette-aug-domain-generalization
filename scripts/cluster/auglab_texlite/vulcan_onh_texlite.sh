#!/usr/bin/env bash
# texlite arm, on-harmony T1w, on VULCAN (2026-10-15 evening): the 3 folds would have taken a whole TamIA H100 node with one GPU
# idle (pack C_onh, cancelled), so they train per-GPU on Vulcan L40S instead, through on-harmony's own wrapper
# (04_65_train_t1w_auglabAugTexlite_v26_6_2_train050_val000.sh -> 04_00_common.sh -> train_common.sh -> run_job, 1 GPU per fold,
# 2000 epochs as the wrapper sets, on-harmony env time 96 h; ~65 s/epoch on L40S -> ~36 h, so no resume chain is needed; if a fold
# ever times out: SINGLE_FOLD=k bash <wrapper> <RUN_ID> resumes from checkpoint_latest). Same RUN_ID as the texlite arm on TamIA,
# results in on-harmony's normal tree on Vulcan (8_results_on-harmony/01_predictions/on_harmony_model/T1w/auglab/<RUN_ID>).
#   bash scripts/cluster/auglab_texlite/vulcan_onh_texlite.sh launch      # submit the 3 fold jobs
#   bash scripts/cluster/auglab_texlite/vulcan_onh_texlite.sh queue-post  # CPU controller behind them: predict + eval + count audit
#   bash scripts/cluster/auglab_texlite/vulcan_onh_texlite.sh _post       # (the controller body)
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
RID=on-harmony_T1w_auglabAugTexlite_v26_6_2_train050_val000_20261015_174645
O=benchmark/02_tasks/brain_healthy/on-harmony/5_scripts_on-harmony
OUTD=/scratch/paulh/review_20261015/auglab_ops/vulcan_onh; mkdir -p "${OUTD}"
export RUN_JOB_ACCOUNT=aip-jcohen RUN_JOB_GPU_TYPE=l40s
export RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-rack02-06}"   # GPU jobs die at CUDA init there (2026-10-06)
case "${1:?launch|queue-post|_post}" in
  launch)
    bash "${O}/04_train/04_65_train_t1w_auglabAugTexlite_v26_6_2_train050_val000.sh" "${RID}" 2>&1 | tee "${OUTD}/launch.log" ;;
  queue-post)
    source scripts/job_runner/run_job.sh
    ids="$(for k in 0 1 2; do squeue -u "${USER}" -h -n "fold${k}_${RID}" -o %i | sort -n | tail -1; done | grep . | paste -sd:)"
    [ "$(echo "${ids}" | tr ':' '\n' | grep -c .)" = 3 ] || { echo "expected 3 queued fold jobs, got '${ids}'"; exit 1; }
    RUN_JOB_DEPENDENCY="afterany:${ids}" run_job --name "texlite_onh_post" --gpus 0 --cpus 8 --mem 64G --time 12:00:00 \
        --log "${OUTD}/post.log" -- bash scripts/cluster/auglab_texlite/vulcan_onh_texlite.sh _post
    echo "post controller queued behind ${ids}" ;;
  _post)
    unset RUN_JOB_DEPENDENCY RUN_JOB_INLINE
    ST="${OUTD}/post.status"; log() { echo "[$(date '+%F %T')] $*" | tee -a "${ST}"; }
    source "${O}/00_utils/env.sh"
    RDIR="${PREDICTIONS_ROOT}/${MODEL_TYPE}/T1w/auglab/${RID}"
    for k in 0 1 2; do for ck in checkpoint_best checkpoint_final; do
      ls "${RDIR}"/Dataset*/*/fold_${k}/${ck}.pth >/dev/null 2>&1 || { log "FAILED: missing ${ck} fold ${k} (resume: SINGLE_FOLD=${k} bash ${O}/04_train/04_65_*.sh ${RID}, then re-run queue-post)"; exit 2; }
    done; done; log "checkpoints OK"
    # predict: checkpoint_best, no TTA, RAS->native resample inside each predict job (05_01_predict_common.sh PREDICT_POST_ITEM_CMD)
    bash "${O}/05_predict/05_40_predict_t1w_auglabAugTexlite_v26_6_2_train050_val000.sh" "${RID}" > "${OUTD}/predict.log" 2>&1 || { log "FAILED: predict"; exit 3; }
    log "predict done"
    # evaluate inside this job (06_01_evaluate_testset.sh refuses predictions off the native GT grid)
    ( export RUN_JOB_INLINE=1; for F in 0 1 2; do bash "${O}/06_evaluate/06_01_evaluate_testset.sh" "${RID}" "${F}" || exit 1; done ) > "${OUTD}/eval.log" 2>&1 \
      || { log "FAILED: eval"; exit 4; }
    # count audit (same expected counts as every on-harmony run: gre has 8 predictions but 7 scored cases)
    .venv/bin/python - "${METRICS_ROOT}/${MODEL_TYPE}/T1w/auglab_${RID}" <<'PY' 2>&1 | tee -a "${ST}"
import csv, sys, collections
exp = {"T1w": 8, "T2w": 8, "bold": 8, "dwi_ap": 8, "epi_ap": 4, "gre_echo1_mag": 7}; bad = 0
for k in (0, 1, 2):
    c = collections.defaultdict(set)
    for r in csv.DictReader(open(f"{sys.argv[1]}/fold{k}/eval_all.csv")):
        c[r.get("group") or r.get("contrast")].add(r["case"])
    for it, n in exp.items():
        if len(c.get(it, ())) != n: print(f"AUDIT FAIL fold{k} {it}: {len(c.get(it, ()))}/{n}"); bad = 1
print("AUDIT OK" if not bad else "AUDIT FAILED"); sys.exit(bad)
PY
    [ "${PIPESTATUS[0]}" = 0 ] && log "DONE ${RID}" || { log "FAILED: audit"; exit 5; } ;;
esac
