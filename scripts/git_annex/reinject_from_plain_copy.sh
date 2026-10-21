#!/bin/bash
# Fill a git-annex clone on Vulcan with content taken from an existing plain copy of the same files
# (the BIDS leaf that was rsynced to romane and pushed to data.neuro). Hashes every file of a hardlink
# copy of <plain_dir> and moves it into the annex when its key is known, so <plain_dir> stays readable
# and no extra space is used. Then verifies and writes a report next to the annex repo.
#
#   bash scripts/git_annex/reinject_from_plain_copy.sh <plain_dir> <annex_repo>          # submit (Slurm CPU job)
#   bash scripts/git_annex/reinject_from_plain_copy.sh --worker <plain_dir> <annex_repo> # what the job runs
set -euo pipefail

GA_BIN=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/git-annex/10.20231129/bin
SELF="$(realpath "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SELF")/../.." && pwd)"

if [ "${1:-}" != "--worker" ]; then
    SRC="$(realpath "$1")"; ANNEX="$(realpath "$2")"
    [ -d "$SRC" ] && [ -d "$ANNEX/.git/annex" ] || { echo "need <plain_dir> and an initialised <annex_repo>" >&2; exit 1; }
    source "${PROJECT_ROOT}/scripts/job_runner/run_job.sh"
    LOG="${ANNEX}.reinject.log"
    run_job --name "reinject_$(basename "$ANNEX")" --gpus 0 --cpus 4 --mem 8G --time 06:00:00 \
        --log "$LOG" -- bash "$SELF" --worker "$SRC" "$ANNEX"
    echo "log: $LOG"
    exit 0
fi

SRC="$(realpath "$2")"; ANNEX="$(realpath "$3")"
export PATH="${GA_BIN}:$PATH"
HL="${SRC}_hl_reinject"
REPORT="${ANNEX}.reinject_report.txt"
cd "$ANNEX"
echo "[$(date -Is)] src=$SRC annex=$ANNEX hl=$HL host=$(hostname)"

if [ ! -d "$HL" ]; then
    echo "[$(date -Is)] hardlink copy"
    cp -al "$SRC" "${HL}.tmp" && mv "${HL}.tmp" "$HL"
fi

echo "[$(date -Is)] reinject --known"
find "$HL" -type f -print0 | xargs -0 -n 200 -P 4 git annex reinject --known >/dev/null || true   # judged by the checks below

echo "[$(date -Is)] verify"
missing=$(git annex find --not --in here | wc -l)
broken=$(find . -path ./.git -prune -o -xtype l -print | wc -l)
dirty=$(git status --porcelain | wc -l)

# Files left in the hardlink copy = not annexed. Each must be a git-tracked (non-annex) file or a
# deliberately excluded leftover; anything else would be content that silently missed the upload.
git ls-files -s | awk '$1 != "120000" {sub(/^[^\t]*\t/, ""); print}' | sort > "${REPORT}.gittracked"
(cd "$HL" && find . -type f | sed 's|^\./||' | sort) > "${REPORT}.leftover"
EXCL='^(derivatives/masks/|derivatives/labels/synthseg_qc_rank[0-9]+\.csv$|\.datalad/|\.gitkeep$|\.gitattributes$|README$)'
unexpected=$(comm -23 "${REPORT}.leftover" "${REPORT}.gittracked" | grep -Ev "$EXCL" || true)

# Git-tracked small files must match the plain copy (README.md was README there).
diffs=""
while IFS= read -r f; do
    o="$SRC/$f"; [ "$f" = README.md ] && [ ! -e "$o" ] && o="$SRC/README"
    [ -e "$o" ] || continue
    cmp -s "$f" "$o" || diffs+="$f"$'\n'
done < "${REPORT}.gittracked"

{
    echo "annexed files without content here: $missing"
    echo "broken symlinks in worktree:        $broken"
    echo "git status lines:                   $dirty"
    echo "leftover files total:               $(wc -l < "${REPORT}.leftover")"
    echo "leftovers neither git-tracked nor excluded: $(printf '%s' "$unexpected" | grep -c . || true)"
    printf '%s\n' "$unexpected" | head -50
    echo "git-tracked files differing from the plain copy:"
    printf '%s' "$diffs"
} | tee "$REPORT"
echo "[$(date -Is)] done"
