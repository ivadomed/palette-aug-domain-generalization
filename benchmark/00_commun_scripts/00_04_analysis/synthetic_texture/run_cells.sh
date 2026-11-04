#!/usr/bin/env bash
# Slurm array task for the synthetic texture experiment (Killarney L40S, one GPU per task).
# Task i -> (lam, seed) from the LAMS x SEEDS grid; the three arms (plain, flat, texture) of that
# cell train concurrently on the same GPU (tiny 2D U-Net, the GPU is far from saturated).
#
# Launch (from the repo root, after the code is committed):
#   OUT=$SCRATCH/synthetic_texture/runs_<tag> sbatch --array=0-47 --account=aip-jcohen \
#     --gres=gpu:l40s:1 --cpus-per-task=6 --mem=24G --time=02:00:00 \
#     --output=$SCRATCH/synthetic_texture/logs/%x_%A_%a.out --job-name=syntex \
#     benchmark/00_commun_scripts/00_04_analysis/synthetic_texture/run_cells.sh
# Env: OUT (required), ITERS (default 4000), ARMS (default "plain flat texture"),
#      LAMS / SEEDS (default the preregistered grid), EXTRA (extra train.py args, e.g. --no-test).
set -euo pipefail
: "${OUT:?set OUT}"
ITERS="${ITERS:-4000}"
ARMS="${ARMS:-plain flat texture}"
read -r -a LAMS <<< "${LAMS:-0.0 0.2 0.4 0.6 0.8 1.0}"
read -r -a SEEDS <<< "${SEEDS:-0 1 2 3 4 5 6 7}"
i="${SLURM_ARRAY_TASK_ID:-0}"
lam="${LAMS[$(( i % ${#LAMS[@]} ))]}"
seed="${SEEDS[$(( i / ${#LAMS[@]} ))]}"
REPO="${REPO:-/project/6102268/paulh/mri_synthesis_project}"
# sbatch runs a spooled copy, so the code dir is passed explicitly (a copy of the committed files).
SCRIPT_DIR="${SCRIPT_DIR:-$REPO/benchmark/00_commun_scripts/00_04_analysis/synthetic_texture}"
PY="${PY:-$REPO/.venv/bin/python}"
module load python/3.11 >/dev/null 2>&1 || true
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=2
echo "[syntex] host=$(hostname) task=$i lam=$lam seed=$seed iters=$ITERS commit=${SYNTEX_COMMIT:-?} $(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
pids=()
for arm in $ARMS; do
  if [ -s "$OUT/lam${lam}_seed${seed}_${arm}.json" ]; then echo "skip $arm (done)"; continue; fi
  "$PY" "$SCRIPT_DIR/train.py" --lam "$lam" --seed "$seed" --arm "$arm" --iters "$ITERS" \
      --out "$OUT" ${EXTRA:-} > "$OUT/logs/lam${lam}_seed${seed}_${arm}.log" 2>&1 &
  pids+=($!)
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[syntex] done rc=$rc $(date -Is)"
exit $rc
