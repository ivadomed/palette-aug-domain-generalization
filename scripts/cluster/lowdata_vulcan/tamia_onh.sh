#!/bin/bash
# TamIA training of the on-harmony low-data texture-step runs (Brain T1w / T2w / dwi_ap, N=5 sessions of 5 subjects, flat + texture,
# folds 0-2 = 18 folds) on 4 whole H100 nodes, ONE fold per GPU except two dwi_ap folds sharing GPU 3 on units u1 and u2 (dwi_ap is
# the cheapest contrast: 34-42 s/epoch at 2 folds per H100 in its full-data runs). Prediction + evaluation run on Vulcan (onh_sync_poller.sh
# brings the finished folds over and queues post_run.sh onh_<contrast>).
#   unit u1: T1w flat f0 f1 f2 | dwi_ap flat f0 + f1 (GPU 3)       unit u3: T2w flat f0 f1 f2    | dwi_ap texture f1 (GPU 3)
#   unit u2: T1w texture f0-f2 | dwi_ap flat f2 + texture f0 (GPU 3) unit u4: T2w texture f0 f1 f2 | dwi_ap texture f2 (GPU 3)
# Run ON TamIA from the repo (relay or direct ssh), with the committed code:
#   bash scripts/cluster/lowdata_vulcan/tamia_onh.sh record <TS>   # records every fold through 04_91 (RUN_JOB_PACK_DIR, TRAIN_FOLDS=<f>,
#                                                                  # tamia_env_onharmony.sh) with RUN_ID on-harmony_<c>_lowdata_n05_<arm>_<TS>
#   bash scripts/cluster/lowdata_vulcan/tamia_onh.sh submit        # run_job_pack_submit.sh per unit, PACK_CHAIN=2 x 23:59 h
#   bash scripts/cluster/lowdata_vulcan/tamia_onh.sh status        # per fold: epochs logged, last epoch time, checkpoint_final present
# Writes the run table (launch.sh format, job_fold* = the unit's chain job ids) to <root>/RUN_IDS_lowdata_vulcan.tsv; copy its rows to
# Vulcan's benchmark/02_tasks/brain_healthy/on-harmony/8_results_on-harmony/_launch/RUN_IDS_lowdata_vulcan.tsv (read by post_run.sh).
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/p/paulh}"
ROOT="${LDT_ROOT:-${SCRATCH}/lowdata_tamia}"; mkdir -p "${ROOT}"
O=benchmark/02_tasks/brain_healthy/on-harmony/5_scripts_on-harmony
UNITS="u1 u2 u3 u4"
rows() {   # unit -> "contrast arm fold" per GPU slot, in index.tsv order
  case "$1" in
    u1) echo "T1w flat 0|T1w flat 1|T1w flat 2|dwi_ap flat 0|dwi_ap flat 1" ;;
    u2) echo "T1w texture 0|T1w texture 1|T1w texture 2|dwi_ap flat 2|dwi_ap texture 0" ;;
    u3) echo "T2w flat 0|T2w flat 1|T2w flat 2|dwi_ap texture 1" ;;
    u4) echo "T2w texture 0|T2w texture 1|T2w texture 2|dwi_ap texture 2" ;;
  esac
}
gpumap() { case "$1" in u1|u2) echo "0 1 2 3 3" ;; *) echo "0 1 2 3" ;; esac; }
MODE="${1:?record <TS> | submit}"
case "${MODE}" in
record)
  TS="${2:?record needs the RUN_ID timestamp}"; echo "${TS}" > "${ROOT}/TS"
  [ -z "$(git status --porcelain -- "${O}" benchmark/00_commun_scripts scripts/cluster | grep -v '^??' || true)" ] || { echo "uncommitted changes"; exit 1; }
  for u in ${UNITS}; do
    UD="${ROOT}/${u}"; [ -f "${UD}/index.tsv" ] && { echo "[record] ${u} already recorded"; continue; }; mkdir -p "${UD}"
    IFS='|' read -r -a R <<< "$(rows "${u}")"
    for r in "${R[@]}"; do
      read -r c a f <<< "${r}"; rid="on-harmony_${c}_lowdata_n05_${a}_${TS}"
      case "${c}" in T1w) envf=env.sh ;; T2w) envf=env_t2w.sh ;; dwi_ap) envf=env_dwi.sh ;; esac
      ( source "${O}/00_utils/${envf}"; source scripts/cluster/tamia_env_onharmony.sh
        LOWDATA_CONTRAST="${c}" TRAIN_FOLDS="${f}" RUN_JOB_PACK_DIR="${UD}" bash "${O}/04_train/04_91_train_lowdata_texture_step.sh" "${a}" 5 "${rid}" < /dev/null
      ) > "${UD}/record_${c}_${a}_f${f}.log" 2>&1 || { echo "record FAILED ${u} ${r}"; tail -5 "${UD}/record_${c}_${a}_f${f}.log"; exit 1; }
      echo "[record] ${u} <- ${rid} fold ${f}"
    done
    [ "$(wc -l < "${UD}/index.tsv")" = "${#R[@]}" ] || { echo "${u}: index.tsv rows != ${#R[@]}"; exit 1; }
  done
  echo "=== recorded commands (check paths) ==="
  for u in ${UNITS}; do while IFS=$'\t' read -r cmd log name done_; do
    echo "${u} ${name}: $(grep -o "nnUNetv2_train [0-9]* 3d_fullres [0-9] *-tr [A-Za-z]*" "${cmd}" | tr -s ' ') | results $(grep -o "nnUNet_results=[^ ]*" "${cmd}" | head -1) | raw $(grep -o "nnUNet_preprocessed=[^ ]*" "${cmd}" | head -1)"
  done < "${ROOT}/${u}/index.tsv"; done ;;
submit)
  TS="$(cat "${ROOT}/TS")"; HEAD="$(git rev-parse HEAD)"
  AL=sub-workspaces/auglab_workspace/AugLab; AH="$(git -C "${AL}" rev-parse --short HEAD)"; AD="$(git -C "${AL}" status --porcelain | wc -l)"
  T="${ROOT}/RUN_IDS_lowdata_vulcan.tsv"
  for u in ${UNITS}; do
    UD="${ROOT}/${u}"; [ -f "${UD}/SUBMITTED" ] && { echo "[submit] ${u} already submitted: $(cat "${UD}/SUBMITTED")"; continue; }
    out="$(PACK_GPU_MAP="$(gpumap "${u}")" PACK_CHAIN="${PACK_CHAIN:-2}" PACK_TIME=23:59:00 PACK_JOB_NAME="ldonh_${u}" \
           bash scripts/job_runner/run_job_pack_submit.sh "${UD}" 2>&1)"; echo "${out}" > "${UD}/submit.log"
    jids="$(echo "${out}" | grep -oE '(Submitted batch job |job )[0-9]+' | grep -oE '[0-9]+' | tr '\n' ' ')"
    [ -n "${jids}" ] || { echo "${out}"; echo "submit ${u} FAILED"; exit 1; }
    echo "${jids}" > "${UD}/SUBMITTED"; echo "[submit] ${u}: chain ${jids}"
  done
  # run table: one row per (setting, arm); job columns = chain job ids of the units holding its folds
  for c in T1w T2w dwi_ap; do for a in flat texture; do
    j=(); for k in 0 1 2; do
      u="$(for uu in ${UNITS}; do rows "${uu}" | tr '|' '\n' | grep -x "${c} ${a} ${k}" >/dev/null && echo "${uu}"; done || true)"
      j+=("tamia:${u}:$(tr ' ' ',' < "${ROOT}/${u}/SUBMITTED" | sed 's/,$//')"); done
    grep -q "on-harmony_${c}_lowdata_n05_${a}_${TS}" "${T}" 2>/dev/null || printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "onh_${c}" "${a}" 5 \
      "on-harmony_${c}_lowdata_n05_${a}_${TS}" "${j[@]}" "tamia_2x23:59" "${HEAD}" "${AH}" "${AD}" "$(date '+%F %T')" >> "${T}"
  done; done
  cat "${T}" ;;
status)
  set +e +o pipefail   # missing fold dirs / logs (not started yet) are normal here
  TS="$(cat "${ROOT}/TS")"
  for c in T1w T2w dwi_ap; do for a in flat texture; do for k in 0 1 2; do
    d="$(ls -d "${SCRATCH}"/on-harmony/8_results/01_predictions/on_harmony_model/${c}/auglab/on-harmony_${c}_lowdata_n05_${a}_${TS}/Dataset*/*/fold_${k} 2>/dev/null | head -1)"
    L="$(ls -t "${d}"/training_log_*.txt 2>/dev/null | head -1)"
    printf '%-7s %-8s fold%s epochs %-5s last %-6s final %s\n' "${c}" "${a}" "${k}" "$([ -n "${L}" ] && grep -c 'Epoch time' "${L}" || echo -)" \
      "$([ -n "${L}" ] && grep 'Epoch time' "${L}" | tail -1 | awk '{print $(NF-1)}')" "$([ -f "${d}/checkpoint_final.pth" ] && echo yes || echo no)"
  done; done; done
  squeue -u "${USER}" -h -o "%i %j %T %M %R" | grep ldonh || true ;;
*) echo "mode: record <TS> | submit | status"; exit 1 ;;
esac
