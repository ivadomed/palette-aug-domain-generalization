#!/bin/bash
# TamIA side of scripts/cluster/flatauglab/metrics_sync.sh (piped over ssh, no checkout needed):
#   ssh -o BatchMode=yes tamia.alliancecan.ca 'bash -s' -- <job id,...> <metrics dir>... < tamia_metrics_status.sh
# Prints  JOB <id> <state> <exit code> <pending reason>   (sacct; the squeue reason for a pending job, e.g. DependencyNeverSatisfied)
#         DIR <dir> present|missing
#         MD5 <md5> <file>                                (every <dir>/fold*/eval_all.csv)
set -uo pipefail
ids="${1:?job ids}"; shift
for j in ${ids//,/ }; do
  st="$(sacct -j "$j" -X -n -P -o State,ExitCode 2>/dev/null | head -1)"
  r="$(squeue -h -j "$j" -o %r 2>/dev/null | head -1)"
  printf 'JOB\t%s\t%s\t%s\t%s\n' "$j" "${st%%|*}" "${st#*|}" "${r:--}"
done
for d in "$@"; do
  if [ -d "$d" ]; then printf 'DIR\t%s\tpresent\n' "$d"
    for f in "$d"/fold*/eval_all.csv; do [ -f "$f" ] && printf 'MD5\t%s\t%s\n' "$(md5sum < "$f" | cut -c1-32)" "$f"; done
  else printf 'DIR\t%s\tmissing\n' "$d"; fi
done
