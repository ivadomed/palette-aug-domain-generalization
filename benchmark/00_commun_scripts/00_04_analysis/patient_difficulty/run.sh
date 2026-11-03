#!/bin/bash
# Patient-level difficulty-matched test (preregistered in README.md here).
#   bash run.sh                 submit compute + analyze as one CPU job (run_job, Vulcan)
#   bash run.sh --inline        do the work in this shell (what the job runs)
#   bash run.sh --inline compute-only   only the per-setting patient tables (+ the difficulty look)
# Outputs: outputs/ (gitignored): <setting>.json, patient_table.csv, setting_table.csv, SUMMARY.md, *.png
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../../.." && pwd)"
cd "${ROOT}"
OUT="${PATIENT_DIFFICULTY_OUT:-${HERE}/outputs}"
if [ "${1:-}" != "--inline" ]; then
    source scripts/job_runner/run_job.sh
    mkdir -p "${OUT}"
    run_job --name patient_difficulty --gpus 0 --cpus 2 --mem 16G --time 01:00:00 \
        --log "${OUT}/run.log" -- bash "${HERE}/run.sh" --inline "$@"
    exit $?
fi
export PROJECT_ROOT="${ROOT}"
T=benchmark/02_tasks
PY="${ROOT}/.venv/bin/python ${HERE}/patient_difficulty.py"
rc=0
# key | type | ladder script (the one each task's ladder runner uses) | env file to source ("" = none)
while IFS='|' read -r key typ script envf; do
    [ -z "${key}" ] && continue
    echo "=== ${key} $(date +%T)"
    ( [ -n "${envf}" ] && source "${envf}" >/dev/null 2>&1
      ${PY} compute --setting "${key}" --type "${typ}" --script "${script}" --out-dir "${OUT}" ) \
        || { echo "!!! ${key} FAILED"; rc=1; }
done <<TABLE
glioma_t1n|appearance|$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_13_ladder_summary.py|
glioma_t2w|appearance|$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_14_ladder_summary_t2w.py|
glioma_t2f|appearance|$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_16_ladder_summary_t2f.py|
glioma_t1c|appearance|$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/06_evaluate/06_33_ladder_summary_t1c.py|
ms_flair|appearance|$T/brain_ms/open-ms/5_scripts_open-ms/06_evaluate/06_19_ladder_summary_ood.py|
ms_t1w|appearance|$T/brain_ms/open-ms/5_scripts_open-ms/06_evaluate/06_18_ladder_summary_t1w.py|
breast_t1wce|appearance|$T/breast_cancer/ispy2/5_scripts_ispy2/06_evaluate/06_10_ladder_summary_t1wce.py|
breast_t2w|appearance|$T/breast_cancer/ispy2/5_scripts_ispy2/06_evaluate/06_11_ladder_summary_t2w.py|
hcc_t1w|appearance|$T/liver_hcc/atlas-liver-hcc/5_scripts_atlas-liver-hcc/06_evaluate/06_13_ladder_summary.py|
brain_t1w|interface|$T/brain_healthy/on-harmony/5_scripts_on-harmony/06_evaluate/06_10_ladder_summary.py|
brain_t2w|interface|$T/brain_healthy/on-harmony/5_scripts_on-harmony/06_evaluate/06_11_ladder_summary_t2w.py|
brain_dwi|interface|$T/brain_healthy/on-harmony/5_scripts_on-harmony/06_evaluate/06_12_ladder_summary_dwi_ap.py|
abdomen_t1in|interface|$T/abdomen_healthy/chaos/5_scripts_chaos/06_evaluate/06_34_ladder_summary_t1in.py|$T/abdomen_healthy/chaos/5_scripts_chaos/00_utils/env.sh
abdomen_t2spir|interface|$T/abdomen_healthy/chaos/5_scripts_chaos/06_evaluate/06_35_ladder_summary_t2spir.py|$T/abdomen_healthy/chaos/5_scripts_chaos/00_utils/env_t2spir.sh
pelvis_ct|interface|$T/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic/06_evaluate/06_10_ladder_summary_ct.py|
pelvis_mri|interface|$T/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic/06_evaluate/06_11_ladder_summary_mri.py|
mandible_cbct|interface|$T/mandible_healthy/toothfairy2/5_scripts_toothfairy2/06_evaluate/06_05_ladder_summary.py|$T/mandible_healthy/toothfairy2/5_scripts_toothfairy2/00_utils/env.sh
pancreas_t1wce|interface|$T/pancreas_disease/pansegdata/5_scripts_pansegdata/06_evaluate/06_10_ladder_summary_t1wce.py|
pancreas_t2w|interface|$T/pancreas_disease/pansegdata/5_scripts_pansegdata/06_evaluate/06_11_ladder_summary_t2w.py|
TABLE
if [ "${2:-}" = "compute-only" ]; then
    ${PY} difficulty --out-dir "${OUT}" > "${OUT}/difficulty_look.txt" || rc=1
else
    ${PY} analyze --out-dir "${OUT}" || rc=1
fi
echo "patient_difficulty rc=${rc}"; exit ${rc}
