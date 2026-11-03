#!/bin/bash
# LAUNCH on Vulcan (L40S, per-GPU run_job fold jobs) the low-data texture-step trainings of one or more settings, both arms
# (flat = rung 4.5, texture = rung 5 PALETTE alone), folds 0-2, through the dataset wrappers (chaos 04_93, pansegdata 04_45,
# on-harmony 04_91),
# then queue the post-training controller (post_run.sh <setting>: predict + evaluate + summary) with afterany on its 6 fold jobs.
#   bash scripts/cluster/lowdata_vulcan/launch.sh <setting> [<setting> ...]      settings: chaos_t1in chaos_t2spir panseg_t1wce panseg_t2w onh_T1w onh_T2w onh_dwi_ap
# Refuses on uncommitted tracked changes in the files it runs. Records per arm: setting, arm, N, RUN_ID, fold job ids, time limit,
# main HEAD, AugLab HEAD + dirty count, submit time -> <dataset>/8_results_<dataset>/_launch/RUN_IDS_lowdata_vulcan.tsv.
# Resume a fold that hit its limit: LOWDATA_CONTRAST=<c> RUN_JOB_TIME_DEFAULT=<t> TRAIN_FOLDS=<k> bash <wrapper> <arm> <N> <RUN_ID>.
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
source scripts/cluster/lowdata_vulcan/common.sh
HEAD="$(git rev-parse HEAD)"
AL=sub-workspaces/auglab_workspace/AugLab
AHEAD="$(git -C "${AL}" rev-parse --short HEAD)"; ADIRTY="$(git -C "${AL}" status --porcelain | wc -l)"
for S in "$@"; do
  ld_setting "${S}"
  dirty="$(git status --porcelain -- "${LD_SCRIPTS}" benchmark/00_commun_scripts scripts/cluster/lowdata_vulcan | grep -v '^??' || true)"
  [ -z "${dirty}" ] || { echo "ERROR: uncommitted changes, commit before launching:"; echo "${dirty}"; exit 1; }
  mkdir -p "$(dirname "${LD_TABLE}")" "${LD_POST}/${S}"
  [ -f "${LD_TABLE}" ] || printf 'setting\tarm\tN\trun_id\tjob_fold0\tjob_fold1\tjob_fold2\ttime_limit\tmain_head\tauglab_head\tauglab_dirty\tsubmitted\n' > "${LD_TABLE}"
  jobs_all=()
  for ARM in flat texture; do
    out="$(LOWDATA_CONTRAST="${LD_TC}" RUN_JOB_TIME_DEFAULT="${LD_TIME}" bash "${LD_SCRIPTS}/${LD_TRAIN}" "${ARM}" "${LD_N}" 2>&1)" \
      || { echo "${out}"; echo "ERROR: launch ${S} ${ARM} failed"; exit 1; }
    echo "${out}" > "${LD_POST}/${S}/launch_${ARM}.log"
    rid="$(echo "${out}" | sed -n 's/.*RUN_ID=\([^ ]*\).*/\1/p' | head -1)"
    mapfile -t jids < <(echo "${out}" | sed -n 's/^Submitted batch job \([0-9]*\)$/\1/p')
    [ -n "${rid}" ] && [ "${#jids[@]}" = 3 ] || { echo "${out}"; echo "ERROR: ${S} ${ARM}: RUN_ID '${rid}' / ${#jids[@]} job ids"; exit 1; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${S}" "${ARM}" "${LD_N}" "${rid}" "${jids[@]}" "${LD_TIME}" \
      "${HEAD}" "${AHEAD}" "${ADIRTY}" "$(date '+%F %T')" >> "${LD_TABLE}"
    echo "[launch] ${S} ${ARM} N=${LD_N} ${rid} jobs ${jids[*]}"
    jobs_all+=("${jids[@]}"); sleep 2
  done
  case "${S}" in chaos_*|onh_*) PT=20:00:00 ;; *) PT=40:00:00 ;; esac
  dep="afterany:$(IFS=:; echo "${jobs_all[*]}")"
  pj="$(sbatch --parsable --account=aip-jcohen --cpus-per-task=16 --mem=96G --time="${PT}" --dependency="${dep}" \
        --job-name="ldpost_${S}" --output="${LD_POST}/${S}/post_job_%j.log" \
        --wrap "bash /project/aip-jcohen/paulh/mri_synthesis_project/scripts/cluster/lowdata_vulcan/post_run.sh ${S}")"
  echo "${S} post ${pj} ${dep}" >> "${LD_POST}/${S}/POST.queued"
  echo "[launch] ${S} post controller ${pj} (${dep})"
done
