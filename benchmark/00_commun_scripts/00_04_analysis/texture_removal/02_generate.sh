#!/bin/bash
# Step 2: texture-removed copies of every OOD input (gen_jobs.tsv written by 01_prepare.sh), one run_job CPU job per
# (task, item), output <work_root>/images/<task>/<item>/ (linked as <input>_texrm; same file names, same
# geometry). Per-case QC rows (case ids: scratch only) in /scratch/paulh/texture_removal_test/gen_stats/. Rerunning skips files already written.
#   bash 02_generate.sh [--wait] [task ...]      (default: every task in gen_jobs.tsv)
set -euo pipefail
ROOT=/project/aip-jcohen/paulh/mri_synthesis_project
HERE="${ROOT}/benchmark/00_commun_scripts/00_04_analysis/texture_removal"
OUT=/scratch/paulh/texture_removal_test; mkdir -p "${OUT}/logs" "${OUT}/gen_stats"
K="$(awk '/^partition:/{gsub(/[{},:]/," "); for(i=1;i<=NF;i++){if($i=="k")print $(i+1)}}' "${HERE}/tasks.yaml")"
CELLS="$(awk '/^partition:/{gsub(/[{},:]/," "); for(i=1;i<=NF;i++){if($i=="cells")print $(i+1)}}' "${HERE}/tasks.yaml")"
WAIT=""; [ "${1:-}" = "--wait" ] && { WAIT="--wait"; shift; }
ONLY=" $* "
cd "${ROOT}"
source scripts/job_runner/run_job.sh
export RUN_JOB_ACCOUNT="${RUN_JOB_ACCOUNT:-aip-jcohen}"
while IFS=$'\t' read -r task key input output lo hi restricted; do
  [ "${ONLY}" = "  " ] || [[ "${ONLY}" == *" ${task} "* ]] || continue
  win=""; [ "${lo}" != "-" ] && win="--ct-window ${lo} ${hi}"   # "-" = no CT window (tab IFS collapses empty fields)
  n=$(ls "${input}" | grep -c '_0000.nii.gz')
  t="01:00:00"; [ "${n}" -gt 150 ] && t="03:00:00"
  run_job --name "texrm_gen_${key:0:40}" --gpus 0 --cpus 8 --mem 48G --time "${t}" ${WAIT} \
    --log "${OUT}/logs/gen_${task}_${key}.log" -- \
    .venv/bin/python "${HERE}/texture_removal.py" --in-dir "${input}" --out-dir "${output}" ${win} \
      --k "${K}" --cells "${CELLS}" --salt "${task}/${key}" --workers 6 --stats "${OUT}/gen_stats/${task}__${key}.tsv" &
  sleep 2
done < <(tail -n +2 "${OUT}/gen_jobs.tsv")
wait
echo "submitted; logs ${OUT}/logs/gen_*.log"
