#!/bin/bash
# Run ON TASSAN: progress of every low-data run in the run table (epoch reached per fold, last epoch time, checkpoint_final)
# + GPU memory / utilisation. Reads fold_N/training_log_*.txt (the real progress record, not the wrapper logs).
#   bash scripts/cluster/lowdata_tassan/status.sh            (MRI-trained runs)
#   LOWDATA_CONTRAST=ct bash scripts/cluster/lowdata_tassan/status.sh   (CT-trained runs)
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"; cd "${REPO}"
D=benchmark/02_tasks/pelvis_healthy/totalseg-pelvic/8_results_totalseg-pelvic
C="${LOWDATA_CONTRAST:-mri}"; TSUF=""; [ "${C}" = ct ] && TSUF="_ct"
TABLE="${D}/_launch/RUN_IDS_lowdata_tassan${TSUF}.tsv"
[ -f "${TABLE}" ] || { echo "no run table yet (${TABLE})"; exit 0; }
while IFS=$'\t' read -r nn arm rid slot started head al aldirty probe; do
  line="${TSUF:+ct }n${nn} ${arm} slot ${slot}${probe:+ probe=${probe}}:"
  for f in 0 1 2; do
    fd="$(ls -d ${D}/01_predictions/totalseg_pelvic_model/${C}/auglab/${rid}/*/*/fold_${f} 2>/dev/null | head -1)"
    log="$(ls ${fd}/training_log_*.txt 2>/dev/null | tail -1)"
    ep="$(grep -oE 'Epoch [0-9]+' "${log}" 2>/dev/null | tail -1 | cut -d' ' -f2)"
    et="$(grep -oE 'Epoch time: [0-9.]+' "${log}" 2>/dev/null | tail -1 | awk '{print $3}')"
    done_=""; [ -f "${fd}/checkpoint_final.pth" ] && done_=" FINAL"
    line+="  f${f} ep=${ep:--} ${et:+${et}s/ep}${done_}"
  done
  echo "${line}  (${rid})"
done < "${TABLE}"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
