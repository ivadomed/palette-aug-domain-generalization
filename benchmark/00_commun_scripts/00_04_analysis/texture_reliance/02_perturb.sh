#!/bin/bash
# Step 2: perturbation T of every OOD input dir (gen_jobs.tsv), one CPU job per (task, input dir), output
# ${WORK}/images/<task>/<dataset>__<item>__<hash>/ (same file names and geometry), QC rows ${WORK}/gen_stats/.
# Prints the job ids (one per line) on stdout. Reruns skip images already written.
#   bash 02_perturb.sh [task ...]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
mkdir -p "${WORK}/gen_stats"
ONLY=" $* "
while IFS=$'\t' read -r task input ds item output salt open; do
  [ "${ONLY}" = "  " ] || [[ "${ONLY}" == *" ${task} "* ]] || continue
  n=$(ls "${input}" | grep -c '_0000.nii.gz')
  mins=$(( 30 + n * 2 ))                   # ~1-2 min per large CT with 4 workers, generous
  t=$(printf '%02d:%02d:00' $(( mins / 60 )) $(( mins % 60 )))
  submit --name "trel_T_${task}_${ds}_${item}" --gpus 0 --cpus 5 --mem 64G --time "${t}" \
    --log "${WORK}/logs/perturb_${task}__$(basename "${output}").log" -- \
    "${PY}" "${HERE}/perturb.py" --in-dir "${input}" --out-dir "${output}" --salt "${salt}" --workers 4 \
      --stats "${WORK}/gen_stats/T__${task}__$(basename "${output}").tsv"
done < <(tail -n +2 "${WORK}/gen_jobs.tsv")
