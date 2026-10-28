#!/bin/bash
# Run the legacy bids-validator (the version NeuroPoly's romane uses) on one or more BIDS leaves as a
# Slurm CPU job, one report per leaf next to the log. Read-only on the leaves.
#
#   bash scripts/git_annex/validate_bids_leaves.sh <out_dir> <leaf> [<leaf> ...]          # submit
#   BV_CONFIG=scripts/git_annex/bids_validator_ct_only.json bash ... # CT-only datasets
#   bash scripts/git_annex/validate_bids_leaves.sh --worker <out_dir> <leaf> [<leaf> ...] # what the job runs
#
# Validator install (login node, once): npm install bids-validator@1.9.7 in $SCRATCH/tools/bidsval197
set -euo pipefail

BV="${BIDS_VALIDATOR:-$SCRATCH/tools/bidsval197/node_modules/.bin/bids-validator}"
SELF="$(realpath "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "$SELF")/../.." && pwd)"

if [ "${1:-}" != "--worker" ]; then
    OUT="$(realpath -m "$1")"; shift
    mkdir -p "$OUT"
    [ -x "$BV" ] || { echo "bids-validator not found at $BV" >&2; exit 1; }
    leaves=(); for l in "$@"; do leaves+=("$(realpath "$l")"); done
    source "${PROJECT_ROOT}/scripts/job_runner/run_job.sh"
    run_job --name bids_validate --gpus 0 --cpus 2 --mem 8G --time 01:00:00 \
        --log "$OUT/validate.log" -- bash "$SELF" --worker "$OUT" "${leaves[@]}"
    echo "reports: $OUT"
    exit 0
fi

shift; OUT="$1"; shift
command -v node >/dev/null || module load StdEnv/2023 nodejs/22.23.4   # compute nodes have no system node
"$BV" --version
for L in "$@"; do
    name="$(basename "$L")"
    echo "[$(date -Is)] $name"
    # BV_CONFIG: optional validator config, e.g. {"ignore": [45]} for datasets whose every image is CT
    # (CT has no BIDS suffix yet, so the images are .bidsignored and the validator then sees no subjects).
    "$BV" --verbose ${BV_CONFIG:+-c "$BV_CONFIG"} "$L" > "$OUT/${name}.txt" 2>&1 || true
    grep -E "^\s*[0-9]+: \[(ERR|WARN)\]|Summary:|files,|errors|warnings" "$OUT/${name}.txt" | head -40 || true
done
echo "[$(date -Is)] done"
