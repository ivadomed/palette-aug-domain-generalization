#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=00:45:00
#SBATCH --job-name=ssem50_selftests
# Roster selftests (shared drivers, synthetic metrics, throwaway workdirs) -- unchanged shared code; run as part of the ssem50 checks.
cd /project/aip-jcohen/paulh/mri_synthesis_project
W=/scratch/p/paulh/review_20261018/ssem50/selftests; rm -rf "$W"; mkdir -p "$W"
.venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/selftest_roster_pipeline.py --dataset pansegdata --task pancreas_disease \
   --model-type pansegdata_model --contrasts t1wce t2w --items t1wce t2w --label pancreas --workdir "$W/roster"; echo "ROSTER_SELFTEST_RC=$?"
.venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/selftest_companion_pipeline.py --source ispy2 --companion ispy1 \
   --model-type ispy2_model --prefix ISPY2 --contrasts t1wce t2w --items t1wce precontrast --workdir "$W/companion"; echo "COMPANION_SELFTEST_RC=$?"
