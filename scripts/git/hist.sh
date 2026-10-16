#!/usr/bin/env bash
# Readable one-line history (skill git-workflow): `bash scripts/git/hist.sh [git-log args]`,
# or `git hist [args]` once the alias is set (`git config alias.hist '!bash scripts/git/hist.sh'`).
#
# Commits made before 2026-10-16 have 150-700-character logbook subjects. A clean
# `<area>: <summary>` (<= 72 chars) for each of them is stored as a git note under
# refs/notes/history (no commit was rewritten). This prints that note when it exists,
# otherwise the commit's own subject.
#   git hist -20                     last 20 commits
#   git hist --since=2026-09-01      a period
#   git hist -- benchmark/02_tasks/pancreas_disease/pansegdata     one dataset
#   git hist | grep '^.\{19\}ladder:'                              one area
# Other clones: `git fetch origin refs/notes/history:refs/notes/history`.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
git log --notes=history --date=short --format='%h %ad %x1f%N%x1f%s%x1e' "$@" \
    | tr -d '\n' | tr '\036' '\n' \
    | awk -F'\037' 'NF >= 3 { s = ($2 != "") ? $2 : $3; if (length(s) > 90) s = substr(s, 1, 87) "..."; print $1 s }'
