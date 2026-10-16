#!/usr/bin/env bash
# Run BEFORE `git push` (skill git-workflow, "Push"). Read-only.
#   bash scripts/git/check_push.sh [branch]      (default: current branch)
# Checks the commits about to go public (origin/<branch>..HEAD):
#   BLOCK: forbidden paths or > 5 MB files anywhere in the range; ruff E9/F63/F7/F82
#          errors in .py files the range touches (errors YOU would turn CI red with).
#   info : commit count, long subjects, CI state of the last pushed commit (a red CI
#          that predates the range is reported, not blocking: it needs an issue + owner).
# Then prints the push command that works here (gh as credential helper).
set -uo pipefail
top="$(git rev-parse --show-toplevel)" || exit 1
cd "$top" || exit 1
export PATH="$HOME/bin:$PATH"
branch="${1:-$(git branch --show-current)}"
RUFF="${RUFF:-$(command -v ruff || echo "$HOME/bin/ruff")}"
fail=0

git fetch -q origin "$branch" 2>/dev/null || echo "check_push: (no origin/$branch yet: whole branch vs origin/main)"
base="origin/$branch"; git rev-parse -q --verify "$base" >/dev/null || base="origin/main"
n=$(git rev-list --count "$base..HEAD")
echo "check_push: $n commit(s) in $base..HEAD"
[ "$n" -eq 0 ] && exit 0

forbidden='(\.nii(\.gz)?|\.dcm|\.mha|\.mhd|\.nrrd|\.pth|\.pt|\.ckpt|\.h5|\.hdf5|\.npz|\.npy|\.pkl|\.zip|\.tar|\.tar\.gz|\.tgz|\.7z)$|\.bak|_bak$|(^|/)toDelete/|^paper/|^CLAUDE\.md$|(^|/)\.claude/|(^|/)\.env$|id_rsa|\.pem$|(^|/)REPO_AUDIT'
while IFS= read -r f; do
    [[ "$f" =~ $forbidden ]] && { echo "check_push: BLOCK: forbidden path in range: $f" >&2; fail=1; }
done < <(git log --format= --name-only --diff-filter=AM "$base..HEAD" | sort -u)
while IFS= read -r path; do
    size=$(git cat-file -s "HEAD:$path" 2>/dev/null || echo 0)
    [ "$size" -gt $((5 * 1024 * 1024)) ] && { echo "check_push: BLOCK: $path is $((size / 1048576)) MB" >&2; fail=1; }
done < <(git diff --name-only --diff-filter=AM "$base..HEAD")

mapfile -t py < <(git diff --name-only --diff-filter=AM "$base..HEAD" -- '*.py')
if [ "${#py[@]}" -gt 0 ] && [ -x "$RUFF" ]; then
    tmp="$(mktemp -d "${SCRATCH:-${TMPDIR:-/tmp}}/check_push.XXXXXX")"; trap 'rm -rf -- "$tmp"' EXIT
    git archive HEAD -- "${py[@]}" | tar -x -C "$tmp"
    (cd "$tmp" && "$RUFF" check --no-cache --isolated --select E9,F63,F7,F82 --output-format concise "${py[@]}" >&2) \
        || { echo "check_push: BLOCK: ruff errors in files this push touches (CI would go red)" >&2; fail=1; }
elif [ "${#py[@]}" -gt 0 ]; then
    echo "check_push: warn: ruff missing, .py files in range not linted" >&2
fi

long=$(git log --format=%s "$base..HEAD" | awk 'length > 72' | wc -l)
[ "$long" -gt 0 ] && echo "check_push: info: $long subject(s) > 72 chars in the range"

if command -v gh >/dev/null; then
    last=$(gh run list --branch "$branch" --limit 1 --json conclusion,headSha,url \
           --jq '.[0] | "\(.conclusion) \(.headSha[0:7]) \(.url)"' 2>/dev/null)
    [ -n "$last" ] && echo "check_push: info: last CI run on $branch: $last"
    case "$last" in failure*) echo "check_push: info: CI already red before this push -> make sure an issue with an owner exists (skill: 'CI')";; esac
fi

if [ "$fail" -eq 0 ]; then
    echo "check_push: OK. Push with:"
    echo "  git -c credential.helper= -c credential.https://github.com.helper='!gh auth git-credential' push origin $branch"
fi
exit "$fail"
