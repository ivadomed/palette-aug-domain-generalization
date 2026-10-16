#!/usr/bin/env bash
# Pull the PALETTE + REDUCED Auglab ("texlite", run ids `*_auglabAugTexlite_v26_6_2_train050_val000_*`, tamia_pack_texlite.sh)
# METRICS run dirs from TamIA scratch into this (Vulcan) repo, flat next to the headline OURS metrics (category auglab).
# Targeted (only these run dirs), md5-verified; clone of fetch_rung7_metrics.sh. Mandible arms: TamIA mandible_only[_sif]/ ->
# Vulcan cbct/; liver: atlas own + lld-mmri-hcc + liverhccseg (TamIA metrics on scratch, written by 05_35).
#   bash scripts/cluster/fetch_texlite_metrics.sh                         # texlite arm
#   TEXLITE_ARM=keepres bash scripts/cluster/fetch_texlite_metrics.sh     # texlite_keepres arm (*_auglabAugTexliteKeepres_*)
#   TEXLITE_ARM=edgeonly bash scripts/cluster/fetch_texlite_metrics.sh    # edgeonly arm (*_auglabAugEdgeonly_*)
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
T=benchmark/02_tasks; SC=/scratch/${USER:0:1}/${USER}
case "${TEXLITE_ARM:-}" in
  "") PAT='*_auglabAugTexlite_v26_6_2_train050_val000_*' ;;
  keepres) PAT='*_auglabAugTexliteKeepres_v26_6_2_train050_val000_*' ;;
  edgeonly) PAT='*_auglabAugEdgeonly_v26_6_2_train050_val000_*' ;;
  *) echo "unknown TEXLITE_ARM='${TEXLITE_ARM}'"; exit 2 ;;
esac
# <TamIA metrics parent (under ${SC})>|<Vulcan metrics parent (under ${T})>
# development settings only (glioma, mandible, HCC held out and cancelled 2026-10-15 evening).
# A source "VULCAN" = the run was trained/predicted/evaluated ON VULCAN (texlite on-harmony T1w, see
# scripts/cluster/auglab_texlite/vulcan_onh_texlite.sh): nothing to copy, the metrics are checked in place.
ONH_SRC="on-harmony/8_results/02_metrics/on_harmony_model/T1w"; [ -z "${TEXLITE_ARM:-}" ] && ONH_SRC=VULCAN
ROWS=(
"${ONH_SRC}|brain_healthy/on-harmony/8_results_on-harmony/02_metrics/on_harmony_model/T1w"
"open-ms/8_results_open-ms/02_metrics/open_ms_model/flair|brain_ms/open-ms/8_results_open-ms/02_metrics/open_ms_model/flair"
"chaos/8_results_chaos/02_metrics/chaos_model/t2spir|abdomen_healthy/chaos/8_results_chaos/02_metrics/chaos_model/t2spir"
)
# FETCH_ONLY=<regex> (optional): only the rows whose Vulcan destination matches (e.g. FETCH_ONLY='open-ms|chaos' while the
# on-harmony run is still training)
for r in "${ROWS[@]}"; do IFS='|' read -r src dst <<<"$r"
  [ -z "${FETCH_ONLY:-}" ] || [[ "${dst}" =~ ${FETCH_ONLY} ]] || continue
  if [ "${src}" = VULCAN ]; then
    run="$(cd "${T}/${dst}" && ls -d ${PAT} 2>/dev/null)"
    [ "$(echo "${run}" | grep -c .)" = 1 ] || { echo "[fetch] expected exactly 1 local (Vulcan) run in ${dst}, got: ${run}"; exit 1; }
    for k in 0 1 2; do [ -s "${T}/${dst}/${run}/fold${k}/eval_all.csv" ] || { echo "[fetch] LOCAL ${run}: fold${k}/eval_all.csv missing (Vulcan post job not done?)"; exit 1; }; done
    echo "[fetch] LOCAL (Vulcan) ${run}"; continue
  fi
  run="$(ssh tamia.alliancecan.ca "cd '${SC}/${src}' && ls -d ${PAT} 2>/dev/null")"
  [ "$(echo "${run}" | grep -c .)" = 1 ] || { echo "[fetch] expected exactly 1 run in ${src}, got: ${run}"; exit 1; }
  mkdir -p "${T}/${dst}"
  ssh tamia.alliancecan.ca "cd '${SC}/${src}' && tar cf - '${run}'" | tar xf - -C "${T}/${dst}"
  a=$(cd "${T}/${dst}/${run}" && find . -type f | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-32)
  b=$(ssh tamia.alliancecan.ca "cd '${SC}/${src}/${run}' && find . -type f | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-32")
  n=$(find "${T}/${dst}/${run}" -type f | wc -l)
  [ "$a" = "$b" ] && echo "[fetch] OK ${n} files  ${run}" || { echo "[fetch] MISMATCH ${run}"; exit 1; }
done
