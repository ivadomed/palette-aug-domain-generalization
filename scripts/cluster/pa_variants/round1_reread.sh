#!/usr/bin/env bash
# Round-1 re-read (2026-10-17): the reduced-Auglab arms of 2026-10-15 (texlite, texlite_keepres, edgeonly; AugLab 153c618 / 5776263 /
# 8e682d8 configs, scripts/cluster/auglab_texlite/) on MS (open-ms flair) and Abdomen (chaos t2spir), fold 0, against the round-2
# TamIA control (arm ctrl of scripts/cluster/pa_variants/, same trainer, loader settings, epochs) instead of the July headline.
# Brain is skipped (round-1 Brain runs were cancelled mid-training).
# Metrics: texlite = its official post jobs (already in the Vulcan results tree); keepres / edgeonly = the 2026-10-16 early read
# (scripts/cluster/auglab_texlite/early_read.sh, TamIA job 622012: each setting's own predict wrapper + evaluate script, count audit
# OK), which ran on the FINAL checkpoint_best of finished runs (best_epochs.txt: final=yes; snapshot checkpoint md5 == run dir),
# so they are the official fold metrics; `fetch` copies them (md5-verified) into the Vulcan results tree next to texlite's.
#   bash scripts/cluster/pa_variants/round1_reread.sh fetch      # keepres / edgeonly metrics TamIA -> Vulcan results tree
#   bash scripts/cluster/pa_variants/round1_reread.sh runids     # fold-0 run table (ctrl + 3 arms) -> ${W}/run_ids.tsv
#   bash scripts/cluster/pa_variants/round1_reread.sh aggregate  # = W=${W} aggregate.sh aggregate -> ${W}/agg/HELDOUT_DELTAS.md
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
W=/scratch/paulh/review_20261017/weaken/round1; mkdir -p "${W}"
TE=/scratch/p/paulh/review_20261016/arms_early; T=benchmark/02_tasks
declare -A SUB=([openms_flair]=open_ms_model/flair [chaos_t2spir]=chaos_model/t2spir)
declare -A DST=([openms_flair]=brain_ms/open-ms/8_results_open-ms/02_metrics/open_ms_model/flair
                [chaos_t2spir]=abdomen_healthy/chaos/8_results_chaos/02_metrics/chaos_model/t2spir)
declare -A PFX=([openms_flair]=open-ms_flair [chaos_t2spir]=chaos_t2spir)
declare -A METH=([texlite]=auglabAugTexlite_v26_6_2_train050_val000_20261015_174645
                 [keepres]=auglabAugTexliteKeepres_v26_6_2_train050_val000_20261015_205134
                 [edgeonly]=auglabAugEdgeonly_v26_6_2_train050_val000_20261015_210451)
case "${1:?fetch|runids|aggregate}" in
  fetch)
    for a in keepres edgeonly; do for s in openms_flair chaos_t2spir; do
      d="auglab_${PFX[$s]}_${METH[$a]}"; src="${TE}/${a}_${s}/02_metrics/${SUB[$s]}"; dst="${T}/${DST[$s]}"
      [ -e "${dst}/${d}" ] && { echo "[fetch] exists, not overwritten: ${dst}/${d}"; continue; }
      ssh -n tamia.alliancecan.ca "grep -q 'final=no' '${TE}/${a}_${s}/best_epochs.txt' && echo NOTFINAL" | grep -q NOTFINAL \
        && { echo "[fetch] ${a} ${s}: early read was not on a final checkpoint -- refused"; exit 1; }
      ssh -n tamia.alliancecan.ca "cd '${src}' && tar cf - '${d}'" | tar xf - -C "${dst}"
      x=$(cd "${dst}/${d}" && find . -type f | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-32)
      y=$(ssh -n tamia.alliancecan.ca "cd '${src}/${d}' && find . -type f | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-32")
      [ "$x" = "$y" ] && echo "[fetch] OK $(find "${dst}/${d}" -name eval_all.csv | wc -l) eval_all.csv  ${a} ${s}" || { echo "[fetch] MISMATCH ${a} ${s}"; exit 1; }
    done; done
    for s in openms_flair chaos_t2spir; do d="${T}/${DST[$s]}/auglab_${PFX[$s]}_${METH[texlite]}"
      [ -f "${d}/fold0/eval_all.csv" ] && echo "[fetch] texlite ${s}: in place (official post job)" || echo "[fetch] texlite ${s}: MISSING"; done ;;
  runids)
    R2=/scratch/paulh/review_20261016/new_variants/run_ids.tsv
    { printf 'setting\tfold\trun_id\tpack\tarm\n'
      awk -F'\t' '$5=="ctrl" && $2==0 && ($1=="openms_flair" || $1=="chaos_t2spir")' "${R2}"
      for s in openms_flair chaos_t2spir; do for a in texlite keepres edgeonly; do
        printf '%s\t0\t%s\tround1\t%s\n' "${s}" "${PFX[$s]}_${METH[$a]}" "${a}"; done; done; } > "${W}/run_ids.tsv"
    cat "${W}/run_ids.tsv" ;;
  aggregate)
    W="${W}" AGG_TITLE="Round-1 reduced-Auglab arms vs TamIA control" bash scripts/cluster/pa_variants/aggregate.sh aggregate ;;
esac
