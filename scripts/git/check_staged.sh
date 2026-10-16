#!/usr/bin/env bash
# Checks the STAGED snapshot before a commit. Called by scripts/git/hooks/pre-commit,
# and runnable by hand: `bash scripts/git/check_staged.sh`. Rules: skill git-workflow.
#
# Blocks (exit 1):
#   - forbidden paths: imaging data, checkpoints, archives, backups, toDelete/,
#     paper/, CLAUDE.md, .claude/, env/credential files
#   - any staged file > 5 MB
#   - Python syntax errors / undefined names (ruff E9,F63,F7,F82 = the CI lint step)
#   - shell syntax errors (bash -n = the CI shell step)
# Warns only:
#   - more than 40 files staged, or code mixed with generated outputs in one commit
#
# ruff: ~/bin/ruff (unzipped from the Alliance wheelhouse, see the skill). Without
# it only py_compile runs and undefined names are NOT checked (said loudly).
set -uo pipefail

top="$(git rev-parse --show-toplevel)" || exit 1
cd "$top" || exit 1
RUFF="${RUFF:-$(command -v ruff || echo "$HOME/bin/ruff")}"
MAX_BYTES=$((5 * 1024 * 1024))

mapfile -d '' files < <(git diff --cached --name-only --diff-filter=ACMR -z)
[ "${#files[@]}" -eq 0 ] && exit 0

fail=0
err() { echo "check_staged: BLOCK: $*" >&2; fail=1; }
warn() { echo "check_staged: warn: $*" >&2; }

forbidden='(\.nii(\.gz)?|\.dcm|\.mha|\.mhd|\.nrrd|\.pth|\.pt|\.ckpt|\.h5|\.hdf5|\.npz|\.npy|\.pkl|\.zip|\.tar|\.tar\.gz|\.tgz|\.7z)$|\.bak|_bak$|(^|/)toDelete/|^paper/|^CLAUDE\.md$|(^|/)\.claude/|(^|/)\.env$|id_rsa|\.pem$|(^|/)REPO_AUDIT'
n_code=0; n_out=0; py=(); sh=()
for f in "${files[@]}"; do
    if [[ "$f" =~ $forbidden ]]; then err "forbidden path staged: $f"; fi
    size=$(git cat-file -s ":$f" 2>/dev/null || echo 0)
    if [ "$size" -gt "$MAX_BYTES" ]; then err "$f is $((size / 1024 / 1024)) MB (> 5 MB)"; fi
    case "$f" in
        *.py) py+=("$f"); n_code=$((n_code + 1)) ;;
        *.sh) sh+=("$f"); n_code=$((n_code + 1)) ;;
        *.yaml|*.yml) n_code=$((n_code + 1)) ;;
        *.png|*.pdf|*.csv|*.svg|*.mp4) n_out=$((n_out + 1)) ;;
        *8_results_*|*/outputs/*) n_out=$((n_out + 1)) ;;
    esac
done

# Lint the STAGED content (not the working tree), in a private temp dir.
if [ "${#py[@]}" -gt 0 ] || [ "${#sh[@]}" -gt 0 ]; then
    tmp="$(mktemp -d "${SCRATCH:-${TMPDIR:-/tmp}}/check_staged.XXXXXX")" || exit 1
    trap 'rm -rf -- "$tmp"' EXIT
    printf '%s\0' "${py[@]}" "${sh[@]}" | git checkout-index -z --stdin --prefix="$tmp/"
    if [ "${#py[@]}" -gt 0 ]; then
        if [ -x "$RUFF" ]; then
            if ! (cd "$tmp" && "$RUFF" check --no-cache --isolated --select E9,F63,F7,F82 \
                    --output-format concise "${py[@]}" >&2); then
                err "ruff E9/F63/F7/F82 failed (the same check CI runs)"
            fi
        else
            warn "ruff not found ($RUFF): only syntax checked, undefined names NOT checked"
            for f in "${py[@]}"; do
                python3 -I -m py_compile "$tmp/$f" 2>&1 >/dev/null | sed 's/^/  /' >&2
                [ "${PIPESTATUS[0]}" -ne 0 ] && err "python syntax error in $f"
            done
        fi
    fi
    for f in "${sh[@]}"; do
        bash -n "$tmp/$f" || err "bash syntax error in $f"
    done
fi

[ "${#files[@]}" -gt 40 ] && warn "${#files[@]} files staged: is this ONE logical change?"
[ "$n_code" -gt 0 ] && [ "$n_out" -gt 0 ] && warn "code ($n_code) and generated outputs ($n_out) in one commit: split them (outputs go in a 'results:' commit)"

if [ "$fail" -ne 0 ]; then
    echo "check_staged: fix the above, or unstage with 'git restore --staged <path>'. Rules: .claude/skills/git-workflow/SKILL.md" >&2
fi
exit "$fail"
