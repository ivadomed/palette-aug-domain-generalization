#!/usr/bin/env bash
# Pull the final-rung (OURS recipe + boundary PV, run ids `*_auglabAug_v26_6_2_pv_train050_val000_*`) METRICS run dirs from
# TamIA scratch into this (Vulcan) repo, flat next to the headline OURS metrics (category auglab, no ablations/ subdir).
# Targeted (only these run dirs), md5-verified like fetch_rung6_pv_metrics.sh. Mandible arms: TamIA mandible_only[_sif]/ -> Vulcan cbct/.
#   bash scripts/cluster/fetch_rung7_metrics.sh
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
T=benchmark/02_tasks; SC=/scratch/${USER:0:1}/${USER}; PAT='*_auglabAug_v26_6_2_pv_train050_val000_*'
# <TamIA metrics parent (under ${SC})>|<Vulcan metrics parent (under ${T})>
ROWS=(
"brats2024-glioma/8_results/02_metrics/brats2024_glioma_model/t1n|brain_tumor/brats2024-glioma/8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/t1n"
"brats2024-glioma/8_results/02_metrics/brats2024_glioma_model/t2w|brain_tumor/brats2024-glioma/8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/t2w"
"brats2024-glioma/8_results/02_metrics/brats2024_glioma_model/t2f|brain_tumor/brats2024-glioma/8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/t2f"
"brats2024-glioma/8_results/02_metrics/brats2024_glioma_model/t1c|brain_tumor/brats2024-glioma/8_results_brats2024-glioma/02_metrics/brats2024_glioma_model/t1c"
"on-harmony/8_results/02_metrics/on_harmony_model/T1w|brain_healthy/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model/T1w"
"on-harmony/8_results/02_metrics/on_harmony_model/T2w|brain_healthy/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model/T2w"
"on-harmony/8_results/02_metrics/on_harmony_model/dwi_ap|brain_healthy/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model/dwi_ap"
"open-ms/8_results_open-ms/02_metrics/open_ms_model/flair|brain_ms/open-ms/8_results_open-ms/02_metrics/open_ms_model/flair"
"open-ms/8_results_open-ms/02_metrics/open_ms_model/t1w|brain_ms/open-ms/8_results_open-ms/02_metrics/open_ms_model/t1w"
"toothfairy2/8_results/02_metrics/toothfairy2_model/cbct_mandible_only|mandible_healthy/toothfairy2/8_results_toothfairy2/02_metrics/toothfairy2_model/cbct"
"hanseg/8_results/02_metrics/toothfairy2_model/cbct/mandible_only_sif|mandible_healthy/hanseg/8_results_hanseg/02_metrics/toothfairy2_model/cbct"
"pddca/8_results/02_metrics/toothfairy2_model/cbct/mandible_only_sif|mandible_healthy/pddca/8_results_pddca/02_metrics/toothfairy2_model/cbct"
"ispy2/8_results/02_metrics/ispy2_model/t1wce|breast_cancer/ispy2/8_results_ispy2/02_metrics/ispy2_model/t1wce"
"ispy2/8_results/02_metrics/ispy2_model/t2w|breast_cancer/ispy2/8_results_ispy2/02_metrics/ispy2_model/t2w"
"chaos/8_results_chaos/02_metrics/chaos_model/t1in|abdomen_healthy/chaos/8_results_chaos/02_metrics/chaos_model/t1in"
"chaos/8_results_chaos/02_metrics/chaos_model/t2spir|abdomen_healthy/chaos/8_results_chaos/02_metrics/chaos_model/t2spir"
"totalseg-pelvic/8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/ct|pelvis_healthy/totalseg-pelvic/8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/ct"
"totalseg-pelvic/8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/mri|pelvis_healthy/totalseg-pelvic/8_results_totalseg-pelvic/02_metrics/totalseg_pelvic_model/mri"
)
for r in "${ROWS[@]}"; do IFS='|' read -r src dst <<<"$r"
  run="$(ssh tamia.alliancecan.ca "cd '${SC}/${src}' && ls -d ${PAT} 2>/dev/null")"
  [ "$(echo "${run}" | grep -c .)" = 1 ] || { echo "[fetch] expected exactly 1 run in ${src}, got: ${run}"; exit 1; }
  mkdir -p "${T}/${dst}"
  ssh tamia.alliancecan.ca "cd '${SC}/${src}' && tar cf - '${run}'" | tar xf - -C "${T}/${dst}"
  a=$(cd "${T}/${dst}/${run}" && find . -type f | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-32)
  b=$(ssh tamia.alliancecan.ca "cd '${SC}/${src}/${run}' && find . -type f | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-32")
  n=$(find "${T}/${dst}/${run}" -type f | wc -l)
  [ "$a" = "$b" ] && echo "[fetch] OK ${n} files  ${run}" || { echo "[fetch] MISMATCH ${run}"; exit 1; }
done
