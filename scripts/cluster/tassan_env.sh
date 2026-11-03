# tassan (NeuroPoly lab workstation, 2x RTX PRO 6000 96 GB, 48 cores, no Slurm): source AFTER the dataset's env.sh
# (or export these before calling a pipeline script; no env.sh resets them). Same override pattern as tamia_env.sh:
# cluster differences are env overrides only, run_job.sh / the shared drivers are never forked.
#   source scripts/cluster/tassan_env.sh
# run_job auto-selects its set_slot backend (set_slot is on PATH). tassan's set_slot (systemd slice, 20 cores / ~55 GB
# per slot, slots 0 and 1, blocking) does not pin GPUs, so each slot is paired with its GPU, and fold fan-outs wider than 2
# (predict_common.sh "all" = folds 0 1 2) wrap onto the two slots (fold 2 -> slot 0). See run_job_set_slot.sh.
export RUN_JOB_BACKEND=set_slot
export RUN_JOB_SET_SLOT_NSLOTS=2
export RUN_JOB_SET_SLOT_CUDA_FROM_SLOT=1
# tassan's set_slot (systemd-run --wait) starts the unit with an EMPTY environment and blocks until it ends:
# forward every exported variable (nnUNet_*, NNUNET_PROJECT_ROOT, PYTHONPATH, PATH, ...) as `env NAME=VALUE`.
export RUN_JOB_SET_SLOT_FORWARD_ENV=1
export WANDB_MODE="${WANDB_MODE:-offline}"
_tassan_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# the shared drivers call <repo>/.venv/bin/...; on tassan the venv lives at $WORK/venv -> link it as <repo>/.venv
[ -x "${_tassan_root}/.venv/bin/python" ] || echo "tassan_env.sh: ${_tassan_root}/.venv/bin/python missing (ln -s <venv> ${_tassan_root}/.venv)" >&2
unset _tassan_root
