#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M
#SBATCH --time=01:00:00
#SBATCH --job-name=ssem50_scan
# READ-ONLY scan of TamIA scratch for the closed augmentation-variant rounds (rounds 1-3 + probes). Deletes NOTHING.
# Output: one TSV row per candidate dir: path  kind  run_id_or_tag  size_KiB  n_files  eval_all.csv md5 per fold (fold:md5;...)
set -uo pipefail
S=/scratch/p/paulh; OUT=${OUT:-$S/review_20261018/ssem50/tamia_variant_scan.tsv}
TSS="20261015_174645 20261015_205134 20261015_210451 20261016_140429 20261017_122851 20261017_122523 20261017_121231 20261017_121832 20261016_135050 20261015_172751"
ROOTS="on-harmony open-ms chaos brats2024-glioma toothfairy2 atlas-liver-hcc hanseg pddca lld-mmri-hcc liverhccseg amos sliver07"
row() { local p="$1" kind="$2" tag="$3" sz nf ev=""
  sz=$(du -sk "$p" 2>/dev/null | cut -f1); nf=$(find "$p" -type f 2>/dev/null | wc -l)
  while read -r c; do [ -n "$c" ] || continue; f=$(basename "$(dirname "$c")"); ev="${ev}${f}:$(md5sum < "$c" | cut -c1-32);"; done < <(find "$p" -name eval_all.csv 2>/dev/null | sort)
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$p" "$kind" "$tag" "$sz" "$nf" "${ev:--}"; }
{ printf 'path\tkind\trun_id_or_tag\tsize_KiB\tn_files\teval_all_md5\n'
for r in ${ROOTS}; do [ -d "$S/$r" ] || continue
  for ts in ${TSS}; do
    find "$S/$r" -maxdepth 9 -type d -name "*_${ts}" -prune 2>/dev/null | while read -r d; do
      b=$(basename "$d"); case "$d" in *02_metrics*) k=metrics;; *01_predictions*|*/nnUNet/*|*/auglab/*) k=run_or_predictions;; *) k=other;; esac
      row "$d" "$k" "${b#auglab_}"; done
  done
done
for d in $S/_packruns/pa_variants_* $S/_packruns/texlite_* $S/toDelete_r3_ctrl81_cancelled_20261017 $S/_pa_variants_probe $S/review_20261015/latency $S/review_20261016/arms_early; do
  [ -e "$d" ] && row "$d" pack_root_or_throwaway "$(basename "$d")"; done
} > "${OUT}"
echo "rows: $(($(wc -l < "${OUT}") - 1))"; awk -F'\t' 'NR>1{s+=$4; n+=$5} END{printf "total %.1f GiB, %d files (may double-count nested rows)\n", s/1048576, n}' "${OUT}"
