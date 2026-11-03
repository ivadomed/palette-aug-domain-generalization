#!/usr/bin/env bash
# run_job backend: set_slot (the original persistent 4-GPU workstation).
# A slot already carries a fixed GPU+CPU bundle, so --gpus/--cpus/--mem/--time
# are accepted (for call-site compatibility with the slurm backend) and
# ignored here. --wait is also a no-op: set_slot hands off to a detached
# systemd unit and returns right away on its own, with or without an
# explicit wait. WANDB_MODE is left untouched (today's online behavior).

run_job() {
    local slot="" log=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --name) shift 2 ;;
            --gpus) shift 2 ;;
            --cpus) shift 2 ;;
            --mem)  shift 2 ;;
            --time) shift 2 ;;
            --slot) slot="$2"; shift 2 ;;
            --log)  log="$2"; shift 2 ;;
            --wait) shift ;;
            --) shift; break ;;
            *) echo "run_job (set_slot): unknown arg '$1'" >&2; return 1 ;;
        esac
    done
    if [ -z "${slot}" ]; then
        echo "run_job (set_slot): --slot is required for this backend" >&2
        return 1
    fi

    # Opt-in, for hosts whose set_slot does not pin GPUs and has fewer slots than a driver's fold fan-out
    # (tassan, set by scripts/cluster/tassan_env.sh; unset = unchanged behaviour):
    #   RUN_JOB_SET_SLOT_NSLOTS=N          a numeric --slot S runs on slot S % N (predict_common.sh "all": fold 2 -> 0)
    #   RUN_JOB_SET_SLOT_CUDA_FROM_SLOT=1  the command runs as `env CUDA_VISIBLE_DEVICES=<slot> ...`
    #   RUN_JOB_SET_SLOT_FORWARD_ENV=1     for a set_slot that starts the unit with an empty environment (tassan:
    #                                      systemd-run): every exported variable of this shell is passed as
    #                                      `env NAME=VALUE ...` (minus session/systemd-specific ones)
    if [ -n "${RUN_JOB_SET_SLOT_NSLOTS:-}" ] && [[ "${slot}" =~ ^[0-9]+$ ]]; then
        slot=$(( slot % RUN_JOB_SET_SLOT_NSLOTS ))
    fi
    local _pre=() _v
    if [ "${RUN_JOB_SET_SLOT_FORWARD_ENV:-0}" = "1" ]; then
        _pre=(env)
        for _v in $(compgen -e); do
            case "${_v}" in
                _|PWD|OLDPWD|SHLVL|INVOCATION_ID|JOURNAL_STREAM|NOTIFY_SOCKET|MANAGERPID|LISTEN_*|SYSTEMD_*|DBUS_*|\
                SSH_*|TMUX*|STY|WINDOW|TERM_SESSION_ID|CUDA_VISIBLE_DEVICES) continue ;;
            esac
            _pre+=("${_v}=${!_v}")
        done
    fi
    if [ "${RUN_JOB_SET_SLOT_CUDA_FROM_SLOT:-0}" = "1" ]; then
        [ ${#_pre[@]} -eq 0 ] && _pre=(env)
        _pre+=("CUDA_VISIBLE_DEVICES=${slot}")
    fi

    if [ -n "${log}" ]; then
        set_slot ${slot} "${_pre[@]}" "$@" > "${log}" 2>&1
    else
        set_slot ${slot} "${_pre[@]}" "$@"
    fi
}
