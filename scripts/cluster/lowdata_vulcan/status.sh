#!/bin/bash
# Status of the Vulcan low-data texture-step runs: per setting / arm / fold the Slurm state, the epochs logged in the
# fold's nnU-Net training log (fold_N/training_log_*.txt) and the last epoch time; then the post controller status.
#   bash scripts/cluster/lowdata_vulcan/status.sh [setting ...]
cd /project/aip-jcohen/paulh/mri_synthesis_project
source scripts/cluster/lowdata_vulcan/common.sh
for S in ${@:-${LD_SETTINGS}}; do
  ld_setting "${S}"; [ -f "${LD_TABLE}" ] || continue
  for ARM in flat texture; do
    rid="$(ld_rid "${S}" "${ARM}")"; [ -n "${rid}" ] || continue
    read -r -a J < <(awk -F'\t' -v r="${rid}" '$4==r {print $5, $6, $7}' "${LD_TABLE}" | tail -1)
    for k in 0 1 2; do
      st="$(sacct -n -X -j "${J[$k]}" -o State%12 2>/dev/null | head -1 | tr -d ' ')"
      L="$(ls -t "${LD_PRED_ROOT}/${rid}"/Dataset*/*/fold_${k}/training_log_*.txt 2>/dev/null | head -1)"
      ep="-"; et="-"
      if [ -n "${L}" ]; then ep="$(grep -c 'Epoch time' "${L}")"; et="$(grep 'Epoch time' "${L}" | tail -1 | awk '{print $(NF-1)}')"; fi
      printf '%-13s %-7s fold%s job %-8s %-11s epochs %-5s last %ss  %s\n' "${S}" "${ARM}" "${k}" "${J[$k]}" "${st:-?}" "${ep}" "${et}" "${rid}"
    done
  done
  [ -f "${LD_POST}/${S}/POST.queued" ] && tail -1 "${LD_POST}/${S}/POST.queued"
  [ -f "${LD_POST}/${S}/POST.status" ] && tail -2 "${LD_POST}/${S}/POST.status"
done
