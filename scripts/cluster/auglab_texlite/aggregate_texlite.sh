#!/usr/bin/env bash
# Vulcan, after training + predict + eval on TamIA: fetch the texlite metrics, write the configs, aggregate. Heavy steps run
# as a CPU job (run_job), never on the login node.
#   bash scripts/cluster/auglab_texlite/aggregate_texlite.sh fetch       # md5-verified copy of the run dirs from TamIA scratch (both arms)
#   bash scripts/cluster/auglab_texlite/aggregate_texlite.sh aggregate   # submits the CPU job below
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
RIDS=/scratch/paulh/review_20261015/auglab_ops/run_ids.tsv            # texlite arm
RIDS_KR=/scratch/paulh/review_20261015/auglab_ops/run_ids_keepres.tsv   # texlite_keepres arm (arm column = keepres)
RIDS_EO=/scratch/paulh/review_20261015/auglab_ops/run_ids_edgeonly.tsv  # edgeonly arm (arm column = edgeonly)
# Development settings only (on-harmony T1w, open-ms flair, chaos t2spir); glioma / mandible / HCC are held out.
T=benchmark/02_tasks
case "${1:?fetch|aggregate|_job}" in
  fetch) bash scripts/cluster/fetch_texlite_metrics.sh
         [ ! -f "${RIDS_KR}" ] || TEXLITE_ARM=keepres bash scripts/cluster/fetch_texlite_metrics.sh
         [ ! -f "${RIDS_EO}" ] || TEXLITE_ARM=edgeonly bash scripts/cluster/fetch_texlite_metrics.sh ;;
  aggregate)
    source scripts/job_runner/run_job.sh
    RUN_JOB_ACCOUNT=aip-jcohen run_job --name texlite_aggregate --gpus 0 --cpus 8 --mem 32G --time 01:00:00 \
        --log /scratch/paulh/review_20261015/auglab_ops/aggregate_%j.log -- bash scripts/cluster/auglab_texlite/aggregate_texlite.sh _job ;;
  _job)
    .venv/bin/python scripts/cluster/auglab_texlite/write_configs.py "${RIDS}" $(for f in "${RIDS_KR}" "${RIDS_EO}"; do [ -f "$f" ] && echo "$f"; done)
    bash $T/brain_healthy/on-harmony/5_scripts_on-harmony/06_evaluate/06_02_aggregate_from_config.sh configs/on-harmony_T1w_texlite_01_results.yaml
    bash $T/brain_ms/open-ms/5_scripts_open-ms/06_evaluate/06_02_aggregate_from_config.sh configs/open-ms_flair_texlite_01_results.yaml
    bash $T/abdomen_healthy/chaos/5_scripts_chaos/06_evaluate/06_02_aggregate_from_config.sh configs/chaos_t2spir_texlite_01_results.yaml
    ;;
esac
