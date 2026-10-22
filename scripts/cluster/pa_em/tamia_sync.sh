#!/usr/bin/env bash
# Bring TamIA's outer checkout up to Vulcan's pushed `main` for the pa_em launch, per skill git-workflow reference/cluster-sync.md.
# Run ON VULCAN:   bash scripts/cluster/pa_em/tamia_sync.sh            # --check (default): report blockers, change nothing
#                  bash scripts/cluster/pa_em/tamia_sync.sh --pull     # do the pull (refuses on any blocker)
#                  bash scripts/cluster/pa_em/tamia_sync.sh --pull-scripts-only   # pull although our jobs are pending/running, allowed ONLY
#                      when every path of the range is a cluster / predict / evaluate script (scripts/cluster/, */05_predict/, */06_evaluate/):
#                      nothing a training job imports (trainers, 04_train, 00_01_train, src/, AugLab) can change under a running job.
# Steps: (1) Vulcan HEAD must be pushed (origin/main == HEAD); (2) TamIA must have no pending/running job of ours (jobs import trainers
# from the working tree at start); (3) every TamIA-local modified file and every untracked file at a path the pull creates is md5-compared
# with Vulcan's committed version: identical -> listed for `git checkout --` / move to $SCRATCH/toDelete_tamia_pull_<date>/, differing ->
# BLOCKER (stop and ask); (4) CLAUDE.md, .claude/settings.json, .claude/hooks are copied aside to $SCRATCH/pa_em_20261022/pull_aside_<ts>/
# and restored after the pull (they are untracked on purpose); (5) `git pull --ff-only origin main`, then TamIA HEAD must equal Vulcan HEAD.
# TamIA /project is near its FILE quota (498K/500K on 2026-10-18): the pull adds ~60 files; the script prints the quota line first.
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
MODE="${1:---check}"
VH="$(git rev-parse HEAD)"; git fetch -q origin main 2>/dev/null || true
[ "$(git rev-parse origin/main)" = "${VH}" ] || { echo "BLOCKER: Vulcan HEAD ${VH:0:7} is not pushed (origin/main $(git rev-parse --short origin/main)); push first (check_push.sh)"; if [ "${MODE}" != --check ]; then exit 1; fi; }
TH="$(ssh tamia.alliancecan.ca 'cd /project/aip-jcohen/paulh/mri_synthesis_project && git rev-parse HEAD')"
echo "[sync] Vulcan HEAD ${VH:0:7}  TamIA HEAD ${TH:0:7}"
git merge-base --is-ancestor "${TH}" "${VH}" || { echo "BLOCKER: TamIA HEAD is not an ancestor of Vulcan HEAD (no fast-forward)"; exit 1; }
# Vulcan md5 of every path the range touches (committed content)
LIST="$(mktemp -p "${SCRATCH:-/scratch/paulh}")"
git diff --name-only "${TH}" "${VH}" | while read -r f; do
  if git cat-file -e "${VH}:${f}" 2>/dev/null; then printf '%s  %s\n' "$(git show "${VH}:${f}" | md5sum | cut -c1-32)" "$f"; else printf 'DELETED  %s\n' "$f"; fi
done > "${LIST}"
echo "[sync] the pull touches $(wc -l < "${LIST}") paths"
if [ "${MODE}" = --pull-scripts-only ]; then
  nonscript="$(awk '{print $2}' "${LIST}" | grep -vE '^scripts/cluster/|/5_scripts_[^/]+/0(5_predict|6_evaluate)/' || true)"
  [ -z "${nonscript}" ] || { echo "BLOCKER (--pull-scripts-only): the range touches non-script paths:"; echo "${nonscript}"; rm -f "${LIST}"; exit 1; }
  echo "[sync] --pull-scripts-only: every path is a cluster / predict / evaluate script"
fi
REMOTE="$(cat <<'EOS'
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"
MODE="$1"; LIST="$2"; VH="$3"
diskusage_report 2>/dev/null | grep -i project || true
n=$(squeue -u "$USER" -h | wc -l); echo "[tamia] jobs of ours in the queue: $n"; squeue -u "$USER" -h | head
blk=0; [ "$n" = 0 ] || [ "$MODE" = --pull-scripts-only ] || { echo "BLOCKER: jobs pending/running (they import trainers from this tree at start)"; blk=1; }
mods="$(git status --porcelain | grep -v '^??' | awk '{print $2}')"; untr="$(git status --porcelain | grep '^??' | awk '{print $2}')"
aside_mod=(); aside_untr=()
while read -r h f; do
  [ -n "$f" ] || continue
  if printf '%s\n' "$mods" | grep -qxF "$f"; then
    t=$(md5sum < "$f" | cut -c1-32); if [ "$t" = "$h" ]; then echo "  identical local mod: $f"; aside_mod+=("$f"); else echo "BLOCKER: locally modified, differs from Vulcan: $f"; blk=1; fi
  elif [ -e "$f" ] && ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    t=$(md5sum < "$f" | cut -c1-32); if [ "$t" = "$h" ]; then echo "  identical untracked: $f"; aside_untr+=("$f"); else echo "BLOCKER: untracked file the pull creates, differs: $f"; blk=1; fi
  fi
done < "$LIST"
echo "[tamia] other local modifications (not touched by the pull): $(printf '%s\n' "$mods" | grep -c . || true); untracked (not touched): $(printf '%s\n' "$untr" | grep -c . || true)"
if [ "$MODE" != --pull ] && [ "$MODE" != --pull-scripts-only ]; then echo "[tamia] --check only. blockers=$blk; identical mods ${#aside_mod[@]}, identical untracked ${#aside_untr[@]}"; exit 0; fi
[ "$blk" = 0 ] || { echo "[tamia] REFUSING to pull (blockers)"; exit 1; }
TS=$(date +%Y%m%d_%H%M%S); A="$SCRATCH/pa_em_20261022/pull_aside_$TS"; D="$SCRATCH/toDelete_tamia_pull_$(date +%Y%m%d)"; mkdir -p "$A" "$D"
for p in CLAUDE.md .claude/settings.json .claude/hooks; do [ -e "$p" ] && cp -a --parents "$p" "$A/"; done
for f in "${aside_mod[@]}"; do git checkout -- "$f"; done
for f in "${aside_untr[@]}"; do mkdir -p "$D/$(dirname "$f")"; mv "$f" "$D/$f"; done
git pull --ff-only origin main || { echo "[tamia] pull FAILED"; exit 1; }
for p in CLAUDE.md .claude/settings.json .claude/hooks; do [ -e "$A/$p" ] && [ ! -e "$p" ] && { mkdir -p "$(dirname "$p")"; cp -a "$A/$p" "$p"; echo "  restored $p"; }; done
[ "$(git rev-parse HEAD)" = "$VH" ] && echo "[tamia] HEAD == Vulcan HEAD ${VH:0:7}: SYNC OK" || { echo "[tamia] HEAD $(git rev-parse --short HEAD) != Vulcan"; exit 1; }
EOS
)"
scp -q "${LIST}" tamia.alliancecan.ca:/scratch/p/paulh/pa_em_20261022/pull_list.txt
printf '%s\n' "${REMOTE}" | ssh tamia.alliancecan.ca "cat > /scratch/p/paulh/pa_em_20261022/tamia_sync_remote.sh && bash /scratch/p/paulh/pa_em_20261022/tamia_sync_remote.sh ${MODE} /scratch/p/paulh/pa_em_20261022/pull_list.txt ${VH}"
rm -f "${LIST}"
