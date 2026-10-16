#!/usr/bin/env bash
# Read-only git/GitHub health report (skill git-workflow, "Audit"). Light enough for a
# login node: no object walks over the 25 GB history.
#   bash scripts/git/audit.sh [days]     (default window: 30 days)
# Compares against origin/main (the local `main` ref may be stale: fetched first).
set -uo pipefail
top="$(git rev-parse --show-toplevel)" || exit 1
cd "$top" || exit 1
export PATH="$HOME/bin:$PATH"
days="${1:-30}"
br="$(git branch --show-current)"
h() { printf '\n== %s\n' "$*"; }

git fetch -q origin 2>/dev/null || echo "(fetch failed: remote numbers may be stale)"

h "Branches"
echo "current: $br  ahead/behind origin/main: $(git rev-list --left-right --count HEAD...origin/main | awk '{print "+"$1" / -"$2}')"
echo "origin/$br: $(git rev-list --left-right --count HEAD...origin/"$br" 2>/dev/null | awk '{print "unpushed "$1", missing "$2}')"
echo "local main vs origin/main: $(git rev-list --left-right --count main...origin/main 2>/dev/null | awk '{print "+"$1" / -"$2}')"
git for-each-ref --sort=-committerdate --format='  %(refname:short)  %(committerdate:short)' refs/heads refs/remotes
echo "worktrees:"; git worktree list | sed 's/^/  /'
echo "merge-base with origin/main: $(git log -1 --format='%h %cd' --date=short "$(git merge-base HEAD origin/main)")"

h "Commits, last $days days"
git log --since="$days.days" --format=%s | awk -v d="$days" '{n++; L+=length; if(length>72)o++; if(length>150)x++}
    END {printf "commits %d (%.1f/day), mean subject %.0f chars, >72: %d, >150: %d\n", n, n/d, (n?L/n:0), o, x}'
echo "prefixes (top 8):"; git log --since="$days.days" --format=%s | grep -oE '^[A-Za-z0-9_./+-]+:' | sort | uniq -c | sort -rn | head -8 | sed 's/^/  /'
echo "biggest commits by files:"
git log --since="$days.days" --format='@%h %s' --shortstat | awk '/^@/{c=$0; next} /files? changed/{print $1, substr(c,1,90)}' | sort -rn | head -5 | sed 's/^/  /'
echo "code+output mixed commits: $(git log --since="$days.days" --format=tformat:@ --name-only | awk '
    /^@/{if(c&&o)m++; c=o=0; next} /\.(py|sh|yaml)$/{c=1} /\.(png|pdf|csv|svg)$|8_results_|\/outputs\//{o=1} END{if(c&&o)m++; print m+0}')"

h "Working tree"
echo "modified tracked: $(git diff --name-only | wc -l), staged: $(git diff --cached --name-only | wc -l)"
echo "untracked scripts (.sh/.py) not ignored: $(git ls-files --others --exclude-standard -- '*.sh' '*.py' | wc -l)"
git ls-files --others --exclude-standard -- '*.sh' '*.py' | xargs -r -n1 dirname | sort | uniq -c | sort -rn | head -6 | sed 's/^/  /'

h "Tracked content"
echo "tracked files: $(git ls-files | wc -l); backups/toDelete tracked: $(git ls-files | grep -cE '\.bak|_bak$|toDelete/')"
echo "tracked files > 1 MB:"; git ls-tree -r -l HEAD | awk '$4 > 1048576 {printf "  %6.1f MB  %s\n", $4/1048576, $5}' | sort -rn | head -8
if [ -x "$HOME/bin/ruff" ] || command -v ruff >/dev/null; then
    echo "ruff E9/F63/F7/F82 errors in tracked .py: $(git ls-files -z '*.py' | xargs -0 ruff check --no-cache --isolated --select E9,F63,F7,F82 --output-format concise 2>/dev/null | grep -c ':[0-9]*:[0-9]*: ')"
fi

h "Sibling repos"
for d in sub-workspaces/auglab_workspace/AugLab paper .claude/skills; do
    if [ -d "$d" ] && [ "$(git -C "$d" rev-parse --show-toplevel 2>/dev/null)" = "$(cd "$d" && pwd -P)" ]; then
        up=$(git -C "$d" rev-list --left-right --count HEAD...@{u} 2>/dev/null | awk '{print "unpushed "$1}');
        echo "$d: $(git -C "$d" branch --show-current) $(git -C "$d" log -1 --format='%h %cd' --date=short) ${up:-no upstream}, dirty $(git -C "$d" status --porcelain | wc -l)"
    else
        echo "$d: NOT a git repo (unversioned)"
    fi
done

h "Tags"
last_tag=$(git describe --tags --abbrev=0 2>/dev/null) && echo "latest: $last_tag (+$(git rev-list --count "$last_tag"..HEAD) commits)" || echo "no tags"

if command -v gh >/dev/null; then
    h "GitHub"
    gh run list --limit 10 --json conclusion --jq '[.[].conclusion] | group_by(.) | map("\(.[0]): \(length)") | join(", ")' 2>/dev/null | sed 's/^/last 10 CI runs: /'
    gh pr list --state open --json number,title --jq '"open PRs: \(length)"' 2>/dev/null
    gh issue list --state open --limit 200 --json number,labels,createdAt \
        --jq '"open issues: \(length), unlabelled: \([.[] | select(.labels|length==0)] | length), oldest: \(min_by(.createdAt).createdAt[0:10] // "-")"' 2>/dev/null
    gh api "repos/{owner}/{repo}/branches/main/protection" >/dev/null 2>&1 && echo "main protected: yes" || echo "main protected: NO"
fi
