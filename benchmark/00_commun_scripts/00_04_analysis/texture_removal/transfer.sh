#!/bin/bash
# Run ON tassan: move one task's data of the texture-removal experiment between Vulcan and this checkout, in parallel.
#   bash transfer.sh pull <Task> [N]   # Vulcan -> here: <Task>.sizes (stage_list.py) split into N size-balanced chunks,
#                                      # N concurrent rsync -rlptL --partial (no -z: .nii.gz/.pth are compressed), then a
#                                      # single dry-run pass (size+mtime) that must find nothing left to transfer
#   bash transfer.sh push <Task> [N]   # here -> Vulcan: every <run>/fold{0,1,2}/{origts,texrm} prediction dir of the
#                                      # task's two runs, one rsync -R --ignore-existing per dir, N at a time, then the
#                                      # same verification pass
# N defaults to 8 (keep <= 8: one ssh session each over the existing ControlMaster; sshd MaxSessions is often 10).
# Env: REPO (default the checkout holding this file), VULCAN (vulcan.alliancecan.ca),
#      VREPO (/project/aip-jcohen/paulh/mri_synthesis_project), VSTAGE (/scratch/paulh/texture_removal_test/stage).
# Pull the next task while the current one predicts: `bash transfer.sh pull Glioma &`.
set -euo pipefail
MODE="${1:?pull|push}"; TASK="${2:?Task}"; N="${3:-8}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "${HERE}/../../../.." && pwd)}"
VULCAN="${VULCAN:-vulcan.alliancecan.ca}"
VREPO="${VREPO:-/project/aip-jcohen/paulh/mri_synthesis_project}"
VSTAGE="${VSTAGE:-/scratch/paulh/texture_removal_test/stage}"
TMP="$(mktemp -d -t "texrm_transfer_${TASK}_XXXX")"; trap 'rm -rf "${TMP}"' EXIT
t0=$(date +%s)
case "${MODE}" in
  pull)
    rsync -q "${VULCAN}:${VSTAGE}/${TASK}.sizes" "${TMP}/sizes"
    # greedy size balancing: largest file first into the currently smallest chunk
    sort -t$'\t' -k1,1nr "${TMP}/sizes" | awk -F'\t' -v n="${N}" -v d="${TMP}" '
      { b = 0; for (i = 1; i < n; i++) if (tot[i] < tot[b]) b = i; tot[b] += $1; print $2 > (d "/chunk" b) }
      END { for (i = 0; i < n; i++) printf "chunk %d: %.2f GB\n", i, tot[i] / 1e9 }'
    cut -f2 "${TMP}/sizes" > "${TMP}/all"
    pids=()
    for c in "${TMP}"/chunk*; do
      rsync -rlptL --partial --files-from="${c}" "${VULCAN}:${VREPO}/" "${REPO}/" & pids+=($!)
    done
    rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
    [ "${rc}" = 0 ] || { echo "ERROR: a pull chunk failed (rerun: --partial resumes)" >&2; exit 1; }
    left=$(rsync -rlptL --dry-run --itemize-changes --files-from="${TMP}/all" "${VULCAN}:${VREPO}/" "${REPO}/" | grep -c '^>f' || true)
    echo "[transfer] pull ${TASK}: $(wc -l < "${TMP}/all") files, $(awk -F'\t' '{s+=$1} END {printf "%.2f", s/1e9}' "${TMP}/sizes") GB in $(( $(date +%s) - t0 )) s; verification: ${left} file(s) still differ"
    [ "${left}" = 0 ] ;;
  push)
    PY="${REPO}/.venv/bin/python"
    mapfile -t RIDS < <("${PY}" -c 'import sys, yaml; t = yaml.safe_load(open(sys.argv[1]))["tasks"][sys.argv[2]]["models"]; print(t["flat"]["run_id"]); print(t["texture"]["run_id"])' "${HERE}/tasks.yaml" "${TASK}")
    cd "${REPO}"
    : > "${TMP}/dirs"
    for rid in "${RIDS[@]}"; do
      for d in benchmark/02_tasks/*/*/8_results_*/01_predictions/*/*/*/"${rid}"/fold[0-2]/{origts,texrm}; do
        [ -d "${d}" ] && echo "${d}" >> "${TMP}/dirs"
      done
    done
    [ -s "${TMP}/dirs" ] || { echo "ERROR: no origts/texrm prediction dirs for ${TASK}" >&2; exit 1; }
    xargs -P "${N}" -I{} rsync -rlptR --partial --ignore-existing ./{} "${VULCAN}:${VREPO}/" < "${TMP}/dirs"
    left=0
    while read -r d; do
      left=$(( left + $(rsync -rlptR --ignore-existing --dry-run --itemize-changes "./${d}" "${VULCAN}:${VREPO}/" | grep -c '^<f' || true) ))
    done < "${TMP}/dirs"
    echo "[transfer] push ${TASK}: $(wc -l < "${TMP}/dirs") prediction dirs, $(cat "${TMP}/dirs" | xargs -I{} find {} -name '*.nii.gz' | wc -l) files in $(( $(date +%s) - t0 )) s; verification: ${left} file(s) not on Vulcan"
    [ "${left}" = 0 ] ;;
  *) echo "usage: $0 pull|push <Task> [N]" >&2; exit 2 ;;
esac
