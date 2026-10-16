#!/usr/bin/env bash
# Readable one-line history (skill git-workflow): `bash scripts/git/hist.sh [git-log args]`,
# or `git hist [args]` once the alias is set (`git config alias.hist '!bash scripts/git/hist.sh'`).
#
# Since the 2026-10-16 history rebuild every subject is a clean `<area>: <summary>`; this is
# `git log --oneline` with dates, subjects cut at 90 chars.
#   git hist -20                     last 20 commits
#   git hist --since=2026-09-01      a period
#   git hist -- benchmark/02_tasks/pancreas_disease/pansegdata     one dataset
#   git hist | grep '^.\{19\}ladder:'                              one area
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1
git log --date=short --format='%h %ad %s' "$@" | cut -c1-110
