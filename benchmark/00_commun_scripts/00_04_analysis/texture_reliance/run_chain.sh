#!/bin/bash
# Whole experiment after 01_build.sh, unattended, with Slurm dependencies:
#   02 perturb -> (afterok) 03 QC and 04 predict -> (afterany) 05 score -> (afterany) 06 analyze.
# Optional env (to resume / extend a partly submitted run): PERTURB_TASKS (tasks to perturb, default all),
# PRED_DOMAINS (domains to submit predictions for, default all), PRED_EXISTING (colon list of predict job ids already
# queued, added to the score dependency).
# Job ids -> ${WORK}/chain_jobs.txt. Final report: ${WORK}/analysis/REPORT.md.
#   bash run_chain.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
G=$(bash "${HERE}/02_perturb.sh" ${PERTURB_TASKS:-} | paste -sd:)
[ -n "${G}" ] || { echo "no perturbation jobs submitted" >&2; exit 1; }
Q=$(RUN_JOB_DEPENDENCY="afterok:${G}" bash "${HERE}/03_qc.sh")
P=$(RUN_JOB_DEPENDENCY="afterok:${G}" bash "${HERE}/04_predict.sh" ${PRED_DOMAINS:-} | paste -sd:)
PA="${P}${PRED_EXISTING:+:${PRED_EXISTING}}"
S=$(RUN_JOB_DEPENDENCY="afterany:${PA}" bash "${HERE}/05_score.sh" | paste -sd:)
A=$(RUN_JOB_DEPENDENCY="afterany:${S}:${Q}" bash "${HERE}/06_analyze.sh")
{ echo "commit $(git -C "${ROOT}" rev-parse HEAD) $(date '+%F %T')"; echo "perturb ${G}"; echo "qc ${Q}"; echo "predict ${PA}"
  echo "score ${S}"; echo "analyze ${A}"; } | tee "${WORK}/chain_jobs.txt"
