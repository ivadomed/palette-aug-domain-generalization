#!/bin/bash
# Whole experiment after 01_build.sh, unattended, with Slurm dependencies:
#   02 perturb -> (afterok) 03 QC and 04 predict -> (afterany) 05 score -> (afterany) 06 analyze.
# Job ids -> ${WORK}/chain_jobs.txt. Final report: ${WORK}/analysis/REPORT.md.
#   bash run_chain.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
G=$(bash "${HERE}/02_perturb.sh" | paste -sd:)
[ -n "${G}" ] || { echo "no perturbation jobs submitted" >&2; exit 1; }
Q=$(RUN_JOB_DEPENDENCY="afterok:${G}" bash "${HERE}/03_qc.sh")
P=$(RUN_JOB_DEPENDENCY="afterok:${G}" bash "${HERE}/04_predict.sh" | paste -sd:)
S=$(RUN_JOB_DEPENDENCY="afterany:${P}" bash "${HERE}/05_score.sh" | paste -sd:)
A=$(RUN_JOB_DEPENDENCY="afterany:${S}:${Q}" bash "${HERE}/06_analyze.sh")
{ echo "commit $(git -C "${ROOT}" rev-parse HEAD) $(date '+%F %T')"; echo "perturb ${G}"; echo "qc ${Q}"; echo "predict ${P}"
  echo "score ${S}"; echo "analyze ${A}"; } | tee "${WORK}/chain_jobs.txt"
