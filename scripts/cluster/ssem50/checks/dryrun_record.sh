# Dry-run RECORD (no submit) of the ssem50 packs + the sizing probe from the code-only staging tree. Run on the TamIA login node.
set -uo pipefail
export SCRATCH=/scratch/p/paulh
TREE=$SCRATCH/review_20261018/ssem50/tree
TS=$(date +%Y%m%d_%H%M%S)
for kind in main probe; do
  R=$SCRATCH/_packruns/ssem50_dryrun_staging_${kind}_${TS}; mkdir -p $R; echo $TS > $R/TS
  echo $R > $SCRATCH/review_20261018/ssem50/dryrun_state_${kind}.txt
done
cd $TREE
SSEM50_REPO=$TREE SSEM50_STATE=$SCRATCH/review_20261018/ssem50/dryrun_state_main.txt bash scripts/cluster/ssem50/launch.sh record > $SCRATCH/review_20261018/ssem50/dryrun_record_main.log 2>&1; echo "main rc=$?"
PROBE=1 SSEM50_REPO=$TREE SSEM50_STATE=$SCRATCH/review_20261018/ssem50/dryrun_state_probe.txt bash scripts/cluster/ssem50/launch.sh record > $SCRATCH/review_20261018/ssem50/dryrun_record_probe.log 2>&1; echo "probe rc=$?"
tail -3 $SCRATCH/review_20261018/ssem50/dryrun_record_main.log; tail -3 $SCRATCH/review_20261018/ssem50/dryrun_record_probe.log
