#!/bin/bash
# CPU EVAL job of the sem50 arm's chaos -> amos / sliver07 CROP-before-predict companions (TamIA), queued by
# `scripts/cluster/sem50/launch.sh queue-companions` afterok behind the 06_36 predict job. Runs the shared driver's eval phases through
# chaos 06_36_fov_crop_ladder_rungs.sh (RUNS_FILE = the arm's chaos_fov_crop_sem50_runs.txt), then AUDITS every new run dir against the
# headline synthseg_EM run's fov_crop metrics (same crops, same case set) with scripts/cluster/sem50/companion_audit.py.
#SBATCH --account=aip-jcohen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
set -uo pipefail
: "${RUNS_FILE:?}"
cd /project/aip-jcohen/paulh/mri_synthesis_project
export SCRATCH="${SCRATCH:-/scratch/${USER:0:1}/${USER}}"
echo "[sem50-fov-eval] host=$(hostname) job=${SLURM_JOB_ID:-?} RUNS_FILE=${RUNS_FILE}"
bash benchmark/02_tasks/abdomen_healthy/chaos/5_scripts_chaos/06_evaluate/06_36_fov_crop_ladder_rungs.sh eval; rc=$?
declare -A REF=([t1in]=chaos_t1in_synthseg_EM_train100_val000_20260611_120000 [t2spir]=chaos_t2spir_synthseg_EM_20260620_112357)
arc=0
for tc in t1in t2spir; do
  rid="$(awk -F'|' -v c="$tc" '$1==c && $0 !~ /^#/ {print $3}' "${RUNS_FILE}")"
  [ -n "${rid}" ] || { echo "[sem50-fov-eval] no ${tc} run in ${RUNS_FILE}"; arc=1; continue; }
  for ds in amos sliver07; do
    M="${SCRATCH}/${ds}/8_results_${ds}/02_metrics/chaos_model/${tc}/fov_crop"
    echo "[sem50-fov-eval] audit ${ds} ${tc} ${rid}"
    .venv/bin/python scripts/cluster/sem50/companion_audit.py "${M}/auglab_${rid}" --ref "${M}/auglab_${REF[$tc]}" || arc=1
  done
done
echo "[sem50-fov-eval] done eval_rc=${rc} audit_rc=${arc}"
[ "${rc}" = 0 ] && [ "${arc}" = 0 ]
