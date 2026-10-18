#!/usr/bin/env bash
# VULCAN (L40S, one GPU per fold, run_job) training of the ssem50 arm on pansegdata t1wce + t2w, folds 0-2 = 6 jobs. Pansegdata's data
# lives on Vulcan only (not staged on TamIA; TamIA's venv has no pansegdata trainer shim). Wrappers: 04_35 / 04_36 (make_wrappers.py).
# No AugLab pin: training imports the venv's editable AugLab (Vulcan checkout), the same tree that holds the arm's config (record its SHA).
#   bash scripts/cluster/ssem50/vulcan_pansegdata.sh --dry-run          # print what would be submitted (default)
#   TL=<hh:mm:ss> bash scripts/cluster/ssem50/vulcan_pansegdata.sh --launch
# Time limit: L40S 2000-epoch single-fold walls measured 2026-10-04 (latency of the pansegdata headline runs): auglab_default 14.5 h,
# synthseg_EM 17.8 h -> this arm ~15-18 h; 22:00:00 leaves margin. A job that hits its limit resumes with the SAME RUN_ID (re-run the
# wrapper with it). RUN_IDs: pansegdata_<tc>_auglabAug_synthseg_EM_train050_val000_<TS>, TS fixed in ${STATE} at the first --launch.
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
MODE="${1:---dry-run}"
D=benchmark/02_tasks/pancreas_disease/pansegdata/5_scripts_pansegdata/04_train
W=("${D}/04_35_train_t1wce_auglabAug_synthseg_EM_train050_val000.sh" "${D}/04_36_train_t2w_auglabAug_synthseg_EM_train050_val000.sh")
METHOD=auglabAug_synthseg_EM_train050_val000
STATE=benchmark/02_tasks/pancreas_disease/pansegdata/8_results_pansegdata/_launch/RUN_IDS_ssem50_vulcan.tsv
export RUN_JOB_ACCOUNT="${RUN_JOB_ACCOUNT:-aip-jcohen}" RUN_JOB_GPU_TYPE=l40s
unset CE_EXTRA_PYTHONPATH
TS="$( [ -f "${STATE}" ] && awk -F'\t' 'NR==2{print $4}' "${STATE}" || date +%Y%m%d_%H%M%S )"
for w in "${W[@]}"; do [ -f "$w" ] || { echo "missing $w" >&2; exit 1; }
  grep -q "^METHOD=\"${METHOD}\"" "$w" || { echo "$w: wrong METHOD" >&2; exit 1; }; done
echo "[ssem50-vulcan] repo $(git rev-parse --short HEAD) dirty=$(git status --porcelain | wc -l) AugLab $(git -C sub-workspaces/auglab_workspace/AugLab rev-parse --short HEAD) TS=${TS}"
case "${MODE}" in
  --dry-run)
    for w in "${W[@]}"; do tc="$(basename "$w" | sed -E 's/^04_[0-9]+_train_(t1wce|t2w)_.*/\1/')"
      echo "[dry-run] ${tc}: TRAIN_FOLDS='0 1 2' RUN_JOB_TIME_DEFAULT=\${TL} bash $w pansegdata_${tc}_${METHOD}_${TS}"; done ;;
  --launch)
    : "${TL:?set TL (per-fold time limit, e.g. 22:00:00)}"
    mkdir -p "$(dirname "${STATE}")"
    [ -f "${STATE}" ] || printf 'contrast\trun_id\tmethod\tts\trepo_head\tauglab_head\n' > "${STATE}"
    for w in "${W[@]}"; do tc="$(basename "$w" | sed -E 's/^04_[0-9]+_train_(t1wce|t2w)_.*/\1/')"; rid="pansegdata_${tc}_${METHOD}_${TS}"
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$tc" "$rid" "${METHOD}" "${TS}" "$(git rev-parse HEAD)" "$(git -C sub-workspaces/auglab_workspace/AugLab rev-parse HEAD)" >> "${STATE}"
      ( export RUN_JOB_TIME_DEFAULT="${TL}" TRAIN_FOLDS="0 1 2"; bash "$w" "$rid" ) < /dev/null
      sleep 3; done ;;
  *) echo "usage: $0 --dry-run | --launch" >&2; exit 2 ;;
esac
