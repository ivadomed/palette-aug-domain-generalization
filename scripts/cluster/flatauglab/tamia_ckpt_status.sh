#!/bin/bash
# Training state of every flatauglab fold on TamIA, read from the pack recording (no repo checkout needed: piped over ssh by
# companions_vulcan.sh / companions_poller.sh on Vulcan:  ssh -o BatchMode=yes tamia.alliancecan.ca 'bash -s' < this [-- md5 <run_id>]).
# Output, one line per recorded fold:  ST <run_id> <fold> <state> <run dir> <fold dir relative to the run dir>
#   state done     = checkpoint_final + checkpoint_best present, checkpoint_latest gone (nnU-Net removes it right after saving the final
#                    checkpoint, on_train_end) and checkpoint_final older than 10 min;  training / settling otherwise.
# With "md5 <run_id>": MD5 <md5> <bytes> <path> for the checkpoint_final + checkpoint_best of each of that run's folds.
set -uo pipefail
ROOT="$(cat /scratch/p/paulh/_packruns_flatauglab_root.txt)"
now=$(date +%s)
for idx in "${ROOT}"/*/index.tsv; do
  while IFS=$'\t' read -r _cmd _log name done_; do
    fold="${name#fold}"; fold="${fold%%_*}"; rid="${name#fold[0-9]_}"
    d="$(dirname "${done_}")"; rundir="${done_%%/Dataset[0-9]*}"
    if [ "${1:-}" = md5 ]; then
      [ "${rid}" = "$2" ] || continue
      for ck in checkpoint_final checkpoint_best; do f="${d}/${ck}.pth"
        [ -f "$f" ] && printf 'MD5\t%s\t%s\t%s\n' "$(md5sum < "$f" | cut -c1-32)" "$(stat -c %s "$f")" "$f"; done
      continue
    fi
    st=training
    if [ -f "${d}/checkpoint_final.pth" ]; then
      st=settling
      [ -f "${d}/checkpoint_best.pth" ] && [ ! -e "${d}/checkpoint_latest.pth" ] \
        && [ $(( now - $(stat -c %Y "${d}/checkpoint_final.pth") )) -gt 600 ] && st=done
    fi
    printf 'ST\t%s\t%s\t%s\t%s\t%s\n' "${rid}" "${fold}" "${st}" "${rundir}" "${d#${rundir}/}"
  done < "${idx}"
done
