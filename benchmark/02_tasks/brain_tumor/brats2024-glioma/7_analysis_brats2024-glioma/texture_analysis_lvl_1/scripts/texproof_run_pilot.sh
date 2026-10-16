#!/usr/bin/env bash
# texproof pilot: GT-free global smoothing (+ noise control, + one flat-fill occlusion map) on the rung-4.5 (flat) vs rung-5 (real) fold-0
# models. One GPU job per dataset via run_job (never on the login node). Usage:
#   bash texproof_run_pilot.sh openms            # 8 Open-MS T2w test cases, sigmas 0.5/1/1.5/2.5 mm, noise 0.25/0.5, occlusion on patient12
#   bash texproof_run_pilot.sh pelvis            # 3 CT cases, sigmas 1/2/3.5/5 mm (large volumes, no occlusion)
set -euo pipefail
LVL1="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "${LVL1}/../../../../../.." && pwd)"
PY="${REPO}/.venv/bin/python"
LOGDIR="${LVL1}/outputs/logs"; mkdir -p "${LOGDIR}"
DS="${1:?openms|pelvis}"
source "${REPO}/benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma/00_utils/env.sh"
case "${DS}" in
  openms) EXTRA="--occlusion patient12"; TIME="01:00:00"; MEM="48G" ;;
  pelvis) EXTRA=""; TIME="01:00:00"; MEM="64G" ;;
  *) echo "unknown dataset ${DS}" >&2; exit 1 ;;
esac
run_job --name "texproof_${DS}" --gpus 1 --cpus 8 --mem "${MEM}" --time "${TIME}" --wait \
    --log "${LOGDIR}/texproof_pilot_${DS}.log" -- \
    bash -c "
    export NNUNET_PROJECT_ROOT='${PROJECT_ROOT}'
    export PYTHONPATH='${PYTHONPATH:-}'
    export CUDA_VISIBLE_DEVICES=0
    export nnUNet_raw='${nnUNet_raw}'
    export nnUNet_preprocessed='${nnUNet_preprocessed}'
    export SCRATCH='${SCRATCH:-/scratch/${USER}}'
    cd '${PROJECT_ROOT}'
    '${PY}' '${LVL1}/scripts/texproof_perturbation_pilot.py' --dataset ${DS} ${EXTRA}
    "
tail -n 60 "${LOGDIR}/texproof_pilot_${DS}.log"
