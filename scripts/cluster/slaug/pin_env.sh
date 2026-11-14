# SLAug arm (slaug): SOURCE this (wrappers, slaug.sh, job bodies). Reads scripts/cluster/slaug/PIN.conf (or $SLAUG_PIN_CONF),
# picks this cluster's pin directory and REFUSES (exit 1) unless the pin on disk is at the pinned commit, clean, and its config has the
# pinned md5. Optional SLAUG_EXPECT_COMMIT / SLAUG_EXPECT_CONFIG_MD5 (set from the run table by slaug.sh) must match too.
# Exports: SLAUG_CLUSTER (vulcan|tamia|killarney; override by pre-setting it), SLAUG_PIN, SLAUG_COMMIT, SLAUG_CONFIG, SLAUG_CONFIG_MD5.
_gp_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_gp_conf="${SLAUG_PIN_CONF:-${_gp_dir}/PIN.conf}"
_gp_die() { echo "[slaug pin] REFUSING: $*" >&2; exit 1; }
[ -f "${_gp_conf}" ] || _gp_die "no ${_gp_conf} (run scripts/cluster/slaug/repin.sh on Killarney)"
_gp_get() { awk -F'\t' -v k="$1" '$1==k {print $2}' "${_gp_conf}"; }
if [ -z "${SLAUG_CLUSTER:-}" ]; then
  if [ -d /scratch/p/paulh ]; then SLAUG_CLUSTER=tamia
  elif [ -d /project/6102268/paulh ]; then SLAUG_CLUSTER=killarney
  else SLAUG_CLUSTER=vulcan; fi
fi
export SLAUG_CLUSTER
SLAUG_COMMIT="$(_gp_get auglab_commit)"; SLAUG_CONFIG="$(_gp_get config)"; SLAUG_CONFIG_MD5="$(_gp_get config_md5)"
SLAUG_PIN="$(_gp_get "pin_${SLAUG_CLUSTER}")"
[ -n "${SLAUG_COMMIT}" ] && [ -n "${SLAUG_CONFIG}" ] && [ -n "${SLAUG_CONFIG_MD5}" ] && [ -n "${SLAUG_PIN}" ] || _gp_die "incomplete ${_gp_conf}"
[ -z "${SLAUG_EXPECT_COMMIT:-}" ] || [ "${SLAUG_EXPECT_COMMIT}" = "${SLAUG_COMMIT}" ] \
  || _gp_die "run table commit ${SLAUG_EXPECT_COMMIT} != PIN.conf ${SLAUG_COMMIT} (re-pin / regenerate the run table)"
[ -z "${SLAUG_EXPECT_CONFIG_MD5:-}" ] || [ "${SLAUG_EXPECT_CONFIG_MD5}" = "${SLAUG_CONFIG_MD5}" ] \
  || _gp_die "run table config md5 ${SLAUG_EXPECT_CONFIG_MD5} != PIN.conf ${SLAUG_CONFIG_MD5}"
[ -d "${SLAUG_PIN}/.git" ] || _gp_die "pin ${SLAUG_PIN} missing on ${SLAUG_CLUSTER}"
[ "$(git -C "${SLAUG_PIN}" rev-parse HEAD 2>/dev/null)" = "${SLAUG_COMMIT}" ] || _gp_die "pin ${SLAUG_PIN} HEAD != ${SLAUG_COMMIT}"
[ -z "$(git -C "${SLAUG_PIN}" status --porcelain 2>/dev/null)" ] || _gp_die "pin ${SLAUG_PIN} has local changes"
[ "$(md5sum < "${SLAUG_PIN}/auglab/configs/${SLAUG_CONFIG}" 2>/dev/null | cut -c1-32)" = "${SLAUG_CONFIG_MD5}" ] \
  || _gp_die "config ${SLAUG_CONFIG} md5 in ${SLAUG_PIN} != ${SLAUG_CONFIG_MD5}"
export SLAUG_PIN SLAUG_COMMIT SLAUG_CONFIG SLAUG_CONFIG_MD5
unset -f _gp_get _gp_die; unset _gp_dir _gp_conf
