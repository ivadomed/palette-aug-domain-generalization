#!/bin/bash
# Label-level difficulty vs target type (preregistered in README.md here).
#   bash run.sh            submit the whole computation as one CPU job (run_job, Vulcan)
#   bash run.sh --inline   do the work in this shell (what the job runs)
# Outputs: outputs/ (gitignored): <setting>.json, label_table.csv, setting_tests.csv, SUMMARY.md, gain_vs_difficulty.png
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"
cd "${ROOT}"
OUT="${LABEL_DIFFICULTY_OUT:-${HERE}/outputs}"
if [ "${1:-}" != "--inline" ]; then
    source scripts/job_runner/run_job.sh
    mkdir -p "${OUT}"
    run_job --name label_difficulty --gpus 0 --cpus 4 --mem 16G --time 02:00:00 \
        --log "${OUT}/run.log" -- bash "${HERE}/run.sh" --inline
    exit $?
fi
export PROJECT_ROOT="${ROOT}"
T=benchmark/02_tasks
PY="${ROOT}/.venv/bin/python ${HERE}/label_difficulty.py"
rc=0
# key | type | ladder script (the one run_ladders.sh uses) | training nnU-Net raw dataset | env file to source ("" = none)
while IFS='|' read -r key typ script raw envf; do
    [ -z "${key}" ] && continue
    echo "=== ${key} $(date +%T)"
    ( [ -n "${envf}" ] && source "${envf}" >/dev/null 2>&1
      ${PY} compute --setting "${key}" --type "${typ}" --script "${script}" --raw-dataset "${raw}" --out-dir "${OUT}" ) \
        || { echo "!!! ${key} FAILED"; rc=1; }
done <<TABLE
glioma_t1n|appearance|$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_13_ladder_summary.py|$T/brain_tumor/brats2024-glioma/2_nnUNet_brats2024-glioma/raw/Dataset051_BraTS2024GliomaT1n|
glioma_t2w|appearance|$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_14_ladder_summary_t2w.py|$T/brain_tumor/brats2024-glioma/2_nnUNet_brats2024-glioma/raw/Dataset052_BraTS2024GliomaT2w|
glioma_t2f|appearance|$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_16_ladder_summary_t2f.py|$T/brain_tumor/brats2024-glioma/2_nnUNet_brats2024-glioma/raw/Dataset053_BraTS2024GliomaT2f|
glioma_t1c|appearance|$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_33_ladder_summary_t1c.py|$T/brain_tumor/brats2024-glioma/2_nnUNet_brats2024-glioma/raw/Dataset054_BraTS2024GliomaT1c|
brain_t1w|interface|$T/brain_healthy/on-harmony/5_scripts_on-harmony/06_evaluate/06_10_ladder_summary.py|$T/brain_healthy/on-harmony/2_nnUNet_on-harmony/raw/Dataset031_OnHarmonyT1w31|
brain_t2w|interface|$T/brain_healthy/on-harmony/5_scripts_on-harmony/06_evaluate/06_11_ladder_summary_t2w.py|$T/brain_healthy/on-harmony/2_nnUNet_on-harmony/raw/Dataset032_OnHarmonyT2w31|
brain_dwi|interface|$T/brain_healthy/on-harmony/5_scripts_on-harmony/06_evaluate/06_12_ladder_summary_dwi_ap.py|$T/brain_healthy/on-harmony/2_nnUNet_on-harmony/raw/Dataset033_OnHarmonyDWI31|
abdomen_t1in|interface|$T/abdomen_healthy/chaos/5_scripts_chaos/06_evaluate/06_34_ladder_summary_t1in.py|$T/abdomen_healthy/chaos/2_nnUNet_chaos/raw/Dataset060_CHAOS_MR_T1in|$T/abdomen_healthy/chaos/5_scripts_chaos/00_utils/env.sh
abdomen_t2spir|interface|$T/abdomen_healthy/chaos/5_scripts_chaos/06_evaluate/06_35_ladder_summary_t2spir.py|$T/abdomen_healthy/chaos/2_nnUNet_chaos/raw/Dataset061_CHAOS_MR_T2spir|$T/abdomen_healthy/chaos/5_scripts_chaos/00_utils/env_t2spir.sh
pelvis_ct|interface|$T/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic/06_evaluate/06_10_ladder_summary_ct.py|$T/pelvis_healthy/totalseg-pelvic/2_nnUNet_totalseg-pelvic/raw/Dataset130_TotalsegPelvic_CT|
pelvis_mri|interface|$T/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic/06_evaluate/06_11_ladder_summary_mri.py|$T/pelvis_healthy/totalseg-pelvic/2_nnUNet_totalseg-pelvic/raw/Dataset131_TotalsegPelvic_MRI|
TABLE
${PY} analyze --out-dir "${OUT}" || rc=1
echo "label_difficulty rc=${rc}"; exit ${rc}
