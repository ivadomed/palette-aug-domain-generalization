#!/bin/bash
# CPU job: build an LAS copy of the SSA inputs (scratch) and run qc_orientation.py on the current (LPS)
# and the LAS inputs against the glioma test set.
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
R=benchmark/03_archive/brats-ssa2024/2_nnUNet_brats-ssa2024/raw
O="${SCRATCH}/review_20261020/brats_ssa"
G=benchmark/02_tasks/brain_tumor/brats2024-glioma
.venv/bin/python scripts/cluster/brats_ssa_pa_alone/reorient_inputs.py "$R" "$O/inputs_las" LAS
for v in lps:$R las:$O/inputs_las; do
  echo "################ ${v%%:*}"
  .venv/bin/python scripts/cluster/brats_ssa_pa_alone/qc_orientation.py "${v#*:}" "$G/2_nnUNet_brats2024-glioma/raw" \
     "$G/4_splits_brats2024-glioma/test_cases.json" "$O/qc/${v%%:*}" || echo "qc exit $?"
done
