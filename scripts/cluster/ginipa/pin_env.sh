# GIN-IPA arm (ginipa): SOURCE this (wrappers, ginipa.sh, job bodies). Reads scripts/cluster/ginipa/PIN.conf (or $GINIPA_PIN_CONF),
# picks this cluster's pin directory and REFUSES (exit 1) unless the pin on disk is at the pinned commit, clean, and its config has the
# pinned md5. Optional GINIPA_EXPECT_COMMIT / GINIPA_EXPECT_CONFIG_MD5 (set from the run table by ginipa.sh) must match too.
# Exports: GINIPA_CLUSTER (vulcan|tamia|killarney; override by pre-setting it), GINIPA_PIN, GINIPA_COMMIT, GINIPA_CONFIG, GINIPA_CONFIG_MD5.
_gp_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_gp_conf="${GINIPA_PIN_CONF:-${_gp_dir}/PIN.conf}"
_gp_die() { echo "[ginipa pin] REFUSING: $*" >&2; exit 1; }
[ -f "${_gp_conf}" ] || _gp_die "no ${_gp_conf} (run scripts/cluster/ginipa/repin.sh on Killarney)"
_gp_get() { awk -F'\t' -v k="$1" '$1==k {print $2}' "${_gp_conf}"; }
if [ -z "${GINIPA_CLUSTER:-}" ]; then
  if [ -d /scratch/p/paulh ]; then GINIPA_CLUSTER=tamia
  elif [ -d /project/6102268/paulh ]; then GINIPA_CLUSTER=killarney
  else GINIPA_CLUSTER=vulcan; fi
fi
export GINIPA_CLUSTER
GINIPA_COMMIT="$(_gp_get auglab_commit)"; GINIPA_CONFIG="$(_gp_get config)"; GINIPA_CONFIG_MD5="$(_gp_get config_md5)"
GINIPA_PIN="$(_gp_get "pin_${GINIPA_CLUSTER}")"
[ -n "${GINIPA_COMMIT}" ] && [ -n "${GINIPA_CONFIG}" ] && [ -n "${GINIPA_CONFIG_MD5}" ] && [ -n "${GINIPA_PIN}" ] || _gp_die "incomplete ${_gp_conf}"
[ -z "${GINIPA_EXPECT_COMMIT:-}" ] || [ "${GINIPA_EXPECT_COMMIT}" = "${GINIPA_COMMIT}" ] \
  || _gp_die "run table commit ${GINIPA_EXPECT_COMMIT} != PIN.conf ${GINIPA_COMMIT} (re-pin / regenerate the run table)"
[ -z "${GINIPA_EXPECT_CONFIG_MD5:-}" ] || [ "${GINIPA_EXPECT_CONFIG_MD5}" = "${GINIPA_CONFIG_MD5}" ] \
  || _gp_die "run table config md5 ${GINIPA_EXPECT_CONFIG_MD5} != PIN.conf ${GINIPA_CONFIG_MD5}"
[ -d "${GINIPA_PIN}/.git" ] || _gp_die "pin ${GINIPA_PIN} missing on ${GINIPA_CLUSTER}"
[ "$(git -C "${GINIPA_PIN}" rev-parse HEAD 2>/dev/null)" = "${GINIPA_COMMIT}" ] || _gp_die "pin ${GINIPA_PIN} HEAD != ${GINIPA_COMMIT}"
[ -z "$(git -C "${GINIPA_PIN}" status --porcelain 2>/dev/null)" ] || _gp_die "pin ${GINIPA_PIN} has local changes"
[ "$(md5sum < "${GINIPA_PIN}/auglab/configs/${GINIPA_CONFIG}" 2>/dev/null | cut -c1-32)" = "${GINIPA_CONFIG_MD5}" ] \
  || _gp_die "config ${GINIPA_CONFIG} md5 in ${GINIPA_PIN} != ${GINIPA_CONFIG_MD5}"
export GINIPA_PIN GINIPA_COMMIT GINIPA_CONFIG GINIPA_CONFIG_MD5
unset -f _gp_get _gp_die; unset _gp_dir _gp_conf
