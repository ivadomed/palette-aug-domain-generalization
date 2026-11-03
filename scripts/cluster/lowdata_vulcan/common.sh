#!/bin/bash
# Shared settings of the Vulcan low-data texture-step runs (chaos N=2, pansegdata N=5, on-harmony N=5 sessions of 5 subjects); sourced by launch.sh / post_run.sh / status.sh.
# setting -> dataset scripts dir, training contrast, wrappers, run table, per-fold time limit.
#   chaos_t1in chaos_t2spir   : CHAOS, N=2 of 12 training cases per fold, 200 epochs (~43 s/epoch L40S full-data -> ~2.5 h)
#   panseg_t1wce panseg_t2w   : PanSegData, N=5 of 113-114 per fold, 2000 epochs (~26 s/epoch L40S full-data -> ~14.5 h)
#   onh_T1w onh_T2w onh_dwi_ap: ON-Harmony, N=5 sessions (5 subjects) of 51-68 per fold, 2000 epochs (full-data runs: 34-63 s/epoch
#                               on TamIA H100 packed 2/GPU; no L40S measurement for these arms -> 3-day limit)
LD_REPO=/project/aip-jcohen/paulh/mri_synthesis_project
LD_POST="${LD_POST:-/scratch/paulh/lowdata_vulcan/post}"
LD_SETTINGS="chaos_t1in chaos_t2spir panseg_t1wce panseg_t2w onh_T1w onh_T2w onh_dwi_ap"
export RUN_JOB_ACCOUNT=aip-jcohen RUN_JOB_GPU_TYPE=l40s
export RUN_JOB_EXCLUDE_NODES="${RUN_JOB_EXCLUDE_NODES:-rack02-06}"
ld_setting() {   # $1 setting -> LD_DS LD_SCRIPTS LD_TC LD_N LD_TRAIN LD_PREDICT LD_SUMMARY LD_TABLE LD_TIME LD_CAT LD_PRED_ROOT LD_MODEL
  case "$1" in
    chaos_t1in|chaos_t2spir)
      LD_DS=chaos; LD_TC="${1#chaos_}"; LD_N=2; LD_CAT=nnUNet; LD_TIME=06:00:00; LD_MODEL=chaos_model
      LD_SCRIPTS=benchmark/02_tasks/abdomen_healthy/chaos/5_scripts_chaos
      LD_TRAIN=04_train/04_93_train_lowdata_texture_step.sh; LD_PREDICT=05_predict/05_79_predict_lowdata_texture_step.sh
      LD_SUMMARY=06_evaluate/06_38_lowdata_texture_step.sh
      LD_RES=benchmark/02_tasks/abdomen_healthy/chaos/8_results_chaos ;;
    panseg_t1wce|panseg_t2w)
      LD_DS=pansegdata; LD_TC="${1#panseg_}"; LD_N=5; LD_CAT=auglab; LD_TIME=22:00:00; LD_MODEL=pansegdata_model
      LD_SCRIPTS=benchmark/02_tasks/pancreas_disease/pansegdata/5_scripts_pansegdata
      LD_TRAIN=04_train/04_45_train_lowdata_texture_step.sh; LD_PREDICT=05_predict/05_42_predict_lowdata_texture_step.sh
      LD_SUMMARY=06_evaluate/06_21_lowdata_texture_step.sh
      LD_RES=benchmark/02_tasks/pancreas_disease/pansegdata/8_results_pansegdata ;;
    onh_T1w|onh_T2w|onh_dwi_ap)
      LD_DS=on-harmony; LD_TC="${1#onh_}"; LD_N=5; LD_CAT=auglab; LD_TIME=3-00:00:00; LD_MODEL=on_harmony_model
      LD_SCRIPTS=benchmark/02_tasks/brain_healthy/on-harmony/5_scripts_on-harmony
      LD_TRAIN=04_train/04_91_train_lowdata_texture_step.sh; LD_PREDICT=05_predict/05_97_predict_lowdata_texture_step.sh
      LD_SUMMARY=06_evaluate/06_21_lowdata_texture_step.sh
      LD_RES=benchmark/02_tasks/brain_healthy/on-harmony/8_results_on-harmony ;;
    *) echo "unknown setting '$1' (${LD_SETTINGS})" >&2; return 1 ;;
  esac
  LD_TABLE="${LD_REPO}/${LD_RES}/_launch/RUN_IDS_lowdata_vulcan.tsv"
  LD_PRED_ROOT="${LD_REPO}/${LD_RES}/01_predictions/${LD_MODEL}/${LD_TC}/${LD_CAT}"
}
ld_rid() {   # $1 setting $2 arm -> RUN_ID of the last launch row
  awk -F'\t' -v s="$1" -v a="$2" '$1==s && $2==a {r=$4} END {print r}' "${LD_TABLE}"
}
