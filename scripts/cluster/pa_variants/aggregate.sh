#!/usr/bin/env bash
# Vulcan tail of the PA variants round 2 (scripts/cluster/pa_variants/launch.sh on TamIA):
#   bash scripts/cluster/pa_variants/aggregate.sh runids      # run table from the TamIA pack root -> ${W}/run_ids.tsv
#   round 3: export PA_ROUND=3 W=/scratch/paulh/review_20261017/weaken/round3 AGG_TITLE="PA variants round 3"; then runids|fetch|aggregate
#   (reference PA_REF_ARM = round 2's `ctrl`, whose rows runids adds to the round-3 table; round 3 runs at 8 workers + compile, ctrl
#   at the dataset defaults (on-harmony / chaos 0/0): that comparison carries the loader change, accepted by the user as negligible)
#   bash scripts/cluster/pa_variants/aggregate.sh fetch       # md5-verified copy of each evaluated run's METRICS dir from TamIA
#                                                             # scratch into the real results tree (flat, next to the headline PA)
#   bash scripts/cluster/pa_variants/aggregate.sh aggregate   # CPU run_job: write_configs.py (scratch configs + symlinked metrics)
#                                                             # + aggregate_from_config.py -> ${W}/agg/<setting>/vs_<ref>[_ho]/01_results_summary.md
#                                                             # + heldout_deltas.py -> ${W}/agg/HELDOUT_DELTAS.md
# W (work dir with run_ids.tsv) and AGG_TITLE (table title prefix) can be overridden: round 1 re-read and round 3 reuse this tail
#   (round1_reread.sh, ROUND=3 launch.sh runids writes ${W}/run_ids.tsv under /scratch/paulh/review_20261017/weaken/).
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
W="${W:-/scratch/paulh/review_20261016/new_variants}"; T=benchmark/02_tasks; SC=/scratch/p/paulh
case "${1:?runids|fetch|aggregate|_job}" in
  runids)   # PA_ROUND=3: the round-3 table + round 2's ctrl rows (same folds) = the reference
    TW="${SC}${W#/scratch/paulh}"; mkdir -p "${W}"
    ssh tamia.alliancecan.ca "bash -lc 'mkdir -p ${TW} && cd /project/aip-jcohen/paulh/mri_synthesis_project && PA_ROUND=${PA_ROUND:-2} bash scripts/cluster/pa_variants/launch.sh runids ${TW}/run_ids.tsv'" >/dev/null
    ssh tamia.alliancecan.ca "cat ${TW}/run_ids.tsv" > "${W}/run_ids.tsv"
    if [ "${PA_ROUND:-2}" != 2 ]; then
      folds="$(tail -n +2 "${W}/run_ids.tsv" | cut -f2 | sort -u | tr '\n' ' ')"
      awk -F'\t' -v F=" ${folds}" '$5=="ctrl" && index(F, " "$2" ")' /scratch/paulh/review_20261016/new_variants/run_ids.tsv >> "${W}/run_ids.tsv"
    fi
    cat "${W}/run_ids.tsv" ;;
  fetch)
    declare -A SRC=([onh_t1w]="on-harmony/8_results/02_metrics/on_harmony_model/T1w" [openms_flair]="open-ms/8_results_open-ms/02_metrics/open_ms_model/flair"
                    [chaos_t2spir]="chaos/8_results_chaos/02_metrics/chaos_model/t2spir")
    declare -A DST=([onh_t1w]="brain_healthy/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model/T1w"
                    [openms_flair]="brain_ms/open-ms/8_results_open-ms/02_metrics/open_ms_model/flair"
                    [chaos_t2spir]="abdomen_healthy/chaos/8_results_chaos/02_metrics/chaos_model/t2spir")
    tail -n +2 "${W}/run_ids.tsv" | cut -f1,3 | sort -u | while IFS=$'\t' read -r s run; do
      [ -z "${FETCH_ONLY:-}" ] || [[ "${s}" =~ ${FETCH_ONLY} ]] || continue
      d="auglab_${run}"; src="${SC}/${SRC[$s]}"; dst="${T}/${DST[$s]}"
      ssh -n tamia.alliancecan.ca "test -d '${src}/${d}'" || { echo "[fetch] not evaluated yet: ${s} ${run}"; continue; }
      mkdir -p "${dst}"; ssh -n tamia.alliancecan.ca "cd '${src}' && tar cf - '${d}'" | tar xf - -C "${dst}"
      a=$(cd "${dst}/${d}" && find . -type f | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-32)
      b=$(ssh -n tamia.alliancecan.ca "cd '${src}/${d}' && find . -type f | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-32")
      [ "$a" = "$b" ] && echo "[fetch] OK $(find "${dst}/${d}" -name eval_all.csv | wc -l) eval_all.csv  ${s} ${run}" || { echo "[fetch] MISMATCH ${run}"; exit 1; }
    done ;;
  aggregate)
    source scripts/job_runner/run_job.sh
    RUN_JOB_ACCOUNT=aip-jcohen run_job --name pa_variants_agg --gpus 0 --cpus 8 --mem 32G --time 01:00:00 \
        --log "${W}/agg_%j.log" --wait -- env W="${W}" AGG_TITLE="${AGG_TITLE:-PA variants round 2}" PA_REF_ARM="${PA_REF_ARM:-ctrl}" bash scripts/cluster/pa_variants/aggregate.sh _job ;;
  _job)
    .venv/bin/python scripts/cluster/pa_variants/write_configs.py "${W}/run_ids.tsv" "${W}/agg" "${AGG_TITLE:-PA variants round 2}"
    for c in "${W}"/agg/*/config_vs_*.yaml; do
      .venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/aggregate_from_config.py "${c}" > "${c%.yaml}.log" 2>&1 \
        && echo "[agg] OK ${c}" || { echo "[agg] FAILED ${c}"; tail -5 "${c%.yaml}.log"; }
    done
    .venv/bin/python scripts/cluster/pa_variants/heldout_deltas.py "${W}/agg" "${W}/run_ids.tsv" ;;
esac
