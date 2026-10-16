#!/usr/bin/env bash
# Vulcan tail of the PA variants round 2 (scripts/cluster/pa_variants/launch.sh on TamIA):
#   bash scripts/cluster/pa_variants/aggregate.sh runids      # run table from the TamIA pack root -> ${W}/run_ids.tsv
#   bash scripts/cluster/pa_variants/aggregate.sh fetch       # md5-verified copy of each evaluated run's METRICS dir from TamIA
#                                                             # scratch into the real results tree (flat, next to the headline PA)
#   bash scripts/cluster/pa_variants/aggregate.sh aggregate   # CPU run_job: write_configs.py (scratch configs + symlinked metrics)
#                                                             # + aggregate_from_config.py -> ${W}/agg/<setting>/vs_<ref>/01_results_summary.md
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
W=/scratch/paulh/review_20261016/new_variants; T=benchmark/02_tasks; SC=/scratch/p/paulh
case "${1:?runids|fetch|aggregate|_job}" in
  runids)
    ssh tamia.alliancecan.ca "bash -lc 'cd /project/aip-jcohen/paulh/mri_synthesis_project && bash scripts/cluster/pa_variants/launch.sh runids ${SC}/review_20261016/new_variants/run_ids.tsv'" >/dev/null
    ssh tamia.alliancecan.ca "cat ${SC}/review_20261016/new_variants/run_ids.tsv" > "${W}/run_ids.tsv"; cat "${W}/run_ids.tsv" ;;
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
        --log "${W}/agg_%j.log" --wait -- bash scripts/cluster/pa_variants/aggregate.sh _job ;;
  _job)
    .venv/bin/python scripts/cluster/pa_variants/write_configs.py "${W}/run_ids.tsv" "${W}/agg"
    for c in "${W}"/agg/*/config_vs_*.yaml; do
      .venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/aggregate_from_config.py "${c}" > "${c%.yaml}.log" 2>&1 \
        && echo "[agg] OK ${c}" || { echo "[agg] FAILED ${c}"; tail -5 "${c%.yaml}.log"; }
    done ;;
esac
