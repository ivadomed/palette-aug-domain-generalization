#!/bin/bash
# Step 4: one GPU job per (domain, model, fold) of pred_jobs.tsv: original + perturbed inputs of every OOD item
# (predict.py). RUN_JOB_DEPENDENCY may hold the perturbation jobs. Prints the job ids on stdout.
#   [RUN_JOB_DEPENDENCY=afterok:...] bash 04_predict.sh [domain ...]
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
ONLY=" $* "
while IFS=$'\t' read -r dom model fold n; do
  [ "${ONLY}" = "  " ] || [[ "${ONLY}" == *" ${dom} "* ]] || continue
  mins=$(( 30 + n * 2 * 10 / 60 ))           # ~10 s per prediction incl. preprocessing/export, 2 conditions
  [ "${mins}" -gt 1440 ] && mins=1440
  t=$(printf '%02d:%02d:00' $(( mins / 60 )) $(( mins % 60 )))
  submit --name "trel_P_${dom}_${model}_f${fold}" --gpus 1 --cpus 8 --mem 64G --time "${t}" \
    --log "${WORK}/logs/predict_${dom}_${model}_f${fold}.log" -- \
    "${PY}" "${HERE}/predict.py" --domain "${dom}" --model "${model}" --fold "${fold}" --work "${WORK}"
done < <(tail -n +2 "${WORK}/pred_jobs.tsv")
