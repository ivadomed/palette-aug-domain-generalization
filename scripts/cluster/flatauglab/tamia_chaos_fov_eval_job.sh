#!/bin/bash
# CPU EVAL job of the flatauglab arm's chaos -> amos / sliver07 CROP-before-predict companions (TamIA), queued by
# `scripts/cluster/flatauglab/launch.sh queue-companions` afterok behind the 06_36 predict job. Runs the shared driver's eval phases through
# chaos 06_36_fov_crop_ladder_rungs.sh (RUNS_FILE = the arm's chaos_fov_crop_flatauglab_runs.txt), then AUDITS every new run dir against the
# headline PA+Auglab run's fov_crop metrics (same crops, same case set) with scripts/cluster/flatauglab/companion_audit.py.
#SBATCH --account=aip-jcohen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
set -uo pipefail
: "${RUNS_FILE:?}"
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"
echo "[flatauglab-fov-eval] host=$(hostname) job=${SLURM_JOB_ID:-?} RUNS_FILE=${RUNS_FILE}"
bash benchmark/02_tasks/abdomen_healthy/chaos/5_scripts_chaos/06_evaluate/06_36_fov_crop_ladder_rungs.sh eval; rc=$?
declare -A REF=([t1in]=chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615 [t2spir]=chaos_t2spir_auglabAug_v26_6_2_train050_val000_20260710_054202)
arc=0
for tc in t1in t2spir; do
  rid="$(awk -F'|' -v c="$tc" '$1==c && $0 !~ /^#/ {print $3}' "${RUNS_FILE}")"
  [ -n "${rid}" ] || { echo "[flatauglab-fov-eval] no ${tc} run in ${RUNS_FILE}"; arc=1; continue; }
  for ds in amos sliver07; do
    M="${SCRATCH}/${ds}/8_results_${ds}/02_metrics/chaos_model/${tc}/fov_crop"
    echo "[flatauglab-fov-eval] audit ${ds} ${tc} ${rid}"
    .venv/bin/python scripts/cluster/flatauglab/companion_audit.py "${M}/auglab_${rid}" --ref "${M}/auglab_${REF[$tc]}" || arc=1
  done
done
echo "[flatauglab-fov-eval] done eval_rc=${rc} audit_rc=${arc}"
[ "${rc}" = 0 ] && [ "${arc}" = 0 ]
