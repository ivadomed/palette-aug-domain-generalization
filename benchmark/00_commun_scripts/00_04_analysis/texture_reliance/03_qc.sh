#!/bin/bash
# Step 3: QC of the operator. (a) positive control: the same gates on 3 images per openly licensed input dir scrambled
# at EVERY scale (perturb.py --mode control_all) must fail; (b) QC slice PNGs from openly licensed cohorts only
# (qc_slices.py; restricted/DUA/NC-ND cohorts never shown); (c) the QC table (analyze.py --qc-only).
# One CPU job; RUN_JOB_DEPENDENCY may hold the perturbation jobs. Prints the job id.
#   [RUN_JOB_DEPENDENCY=afterok:...] bash 03_qc.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
submit --name trel_qc --gpus 0 --cpus 4 --mem 48G --time 03:00:00 --log "${WORK}/logs/qc.log" -- \
  bash -c "set -e; cd ${ROOT}
    tail -n +2 ${WORK}/gen_jobs.tsv | awk -F'\t' '\$7==1' | while IFS=\$'\t' read -r task input ds item output salt open; do
      cs=\$(ls \${input} | grep '_0000.nii.gz' | head -3 | sed 's/_0000.nii.gz//' | paste -sd,)
      ${PY} ${HERE}/perturb.py --in-dir \${input} --out-dir ${WORK}/control_all/\$(basename \${output}) --salt \${salt} \
        --mode control_all --cases \${cs} --workers 3 --stats ${WORK}/gen_stats/control__\$(basename \${output}).tsv
    done
    ${PY} ${HERE}/qc_slices.py --work ${WORK}
    ${PY} ${HERE}/analyze.py --work ${WORK} --qc-only"
