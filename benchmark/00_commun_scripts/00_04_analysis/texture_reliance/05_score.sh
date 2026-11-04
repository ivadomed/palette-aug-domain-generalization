#!/bin/bash
# Step 5: score every domain (score.py: agreement Dice, GT Dice, validation vs the ladder's eval_all.csv), one CPU job
# per domain. RUN_JOB_DEPENDENCY may hold the predict jobs. Prints the job ids.
#   [RUN_JOB_DEPENDENCY=afterany:...] bash 05_score.sh [domain ...]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ONLY=" $* "
for dom in $(tail -n +2 "${WORK}/pred_jobs.tsv" | cut -f1 | uniq); do
  [ "${ONLY}" = "  " ] || [[ "${ONLY}" == *" ${dom} "* ]] || continue
  submit --name "trel_S_${dom}" --gpus 0 --cpus 8 --mem 48G --time 03:00:00 --log "${WORK}/logs/score_${dom}.log" -- \
    "${PY}" "${HERE}/score.py" --domain "${dom}" --work "${WORK}" --workers 7
done
