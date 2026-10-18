# Phase 1 (check) and 2 (move) of the ssem50 TamIA scratch cleanup. MODE=check|move. List = del_paths.tsv (path, size KiB, files).
set -uo pipefail
S=/scratch/p/paulh; L=$S/review_20261018/ssem50/del_paths.tsv; D=$S/toDelete_ssem50_20261018; MODE=$1
squeue -u paulh -h | grep -q . && { echo "jobs in queue -- abort"; squeue -u paulh -h; exit 1; }
bad=0; n=0
while IFS=$'\t' read -r p sz nf; do n=$((n+1))
  case "$p" in /scratch/p/paulh/?*) ;; *) echo "OUTSIDE SCRATCH: $p"; bad=1; continue;; esac
  [ -e "$p" ] || { echo "MISSING $p"; bad=1; continue; }
  csz=$(du -sk "$p" | cut -f1); cnf=$(find "$p" -type f | wc -l)
  [ "$csz" = "$sz" ] && [ "$cnf" = "$nf" ] || { echo "CHANGED $p size $sz->$csz files $nf->$cnf"; bad=1; }
done < "$L"
echo "[check] $n rows, problems=$bad"
[ "$MODE" = move ] || exit $bad
[ "$bad" = 0 ] || { echo "refusing to move"; exit 1; }
[ -e "$D" ] && { echo "$D already exists -- refusing"; exit 1; }
mkdir -p "$D"; moved=0
while IFS=$'\t' read -r p sz nf; do rel="${p#/scratch/p/paulh/}"; mkdir -p "$D/$(dirname "$rel")"; mv "$p" "$D/$rel" && moved=$((moved+1)); done < "$L"
echo "[move] moved $moved"
# verify: every listed path gone from its place and present under D; nothing else under D
ok=0; while IFS=$'\t' read -r p sz nf; do rel="${p#/scratch/p/paulh/}"; [ ! -e "$p" ] && [ -e "$D/$rel" ] && ok=$((ok+1)); done < "$L"
cut -f1 "$L" | sed 's|^/scratch/p/paulh/||' | sort > $S/review_20261018/ssem50/del_expected.txt
( cd "$D" && while IFS=$'\t' read -r p sz nf; do :; done < /dev/null; find . -mindepth 1 -type d | sed 's|^\./||' | sort ) > $S/review_20261018/ssem50/del_dirs_under_D.txt
# entries under D that are neither a listed path, inside one, nor a parent created for one
extra=$(python3 - "$S/review_20261018/ssem50/del_expected.txt" "$S/review_20261018/ssem50/del_dirs_under_D.txt" <<'PY'
import sys
exp=[l.strip() for l in open(sys.argv[1]) if l.strip()]; got=[l.strip() for l in open(sys.argv[2]) if l.strip()]
bad=[g for g in got if not any(g==e or g.startswith(e+'/') or e.startswith(g+'/') for e in exp)]
print(len(bad)); [print(b) for b in bad[:10]]
PY
)
echo "[verify] listed paths moved and present under toDelete: $ok / $(wc -l < "$L"); entries under toDelete outside the list: $extra"
du -sk "$D" | awk '{printf "[verify] toDelete size %.2f GiB\n", $1/1048576}'; find "$D" -type f | wc -l | sed 's/^/[verify] toDelete files /'
