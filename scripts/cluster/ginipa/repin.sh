#!/usr/bin/env bash
# GIN-IPA arm (ginipa): (re)pin the AugLab code on all three clusters in ONE command. Run on a KILLARNEY login node
# (the gin-ipa branch lives only in the Killarney clone /scratch/paulh/auglab_ginipa/AugLab; it is never pushed to GitHub).
#
#   bash scripts/cluster/ginipa/repin.sh [REF] [CONFIG]          # default REF gin-ipa, CONFIG transform_params_gpu_ginipa.json
#   GINIPA_SRC=<clone> bash scripts/cluster/ginipa/repin.sh ...   # other source clone
#   bash scripts/cluster/ginipa/repin.sh --check                  # verify the pins named by the committed PIN.conf, change nothing
#
# What it does (read-only on every existing AugLab checkout: the shared sub-workspaces trees and /scratch/paulh/flatauglab_pin
# are never touched):
#   1. resolve REF to a commit in the source clone (committed content only; uncommitted edits of the reviewers are not shipped);
#   2. `git bundle create --all` (whole history, ~2.5 MB) -> Vulcan /scratch/paulh/ginipa_prep/bundles/ -> TamIA (through the relay);
#   3. on each cluster: fresh clone of the bundle into <pin root>/<sha12>/AugLab, detached checkout of the commit, origin removed.
#      One directory PER COMMIT: re-pinning never changes the code a running / chained training imports (its job body checks the
#      commit it was recorded with). An existing directory is verified (HEAD, clean) and reused;
#   4. verify per cluster: HEAD == commit, clean, config present, md5 of the config, md5 over every tracked file (must agree);
#   5. write scripts/cluster/ginipa/PIN.conf in the Vulcan main repo (commit it with skill git-workflow) and print it.
# After a re-pin: regenerate the run table (`bash scripts/cluster/ginipa/ginipa.sh runtable`), re-record (`... record`), and only then submit;
# record/submit refuse while the run table's commit / config md5 differ from PIN.conf or from the pin on disk.
set -euo pipefail
VREPO=/project/aip-jcohen/paulh/mri_synthesis_project
SRC="${GINIPA_SRC:-/scratch/paulh/auglab_ginipa/AugLab}"
ROOT_V=/project/aip-jcohen/paulh/ginipa_pin        # Vulcan  (project space, backed up)
ROOT_K=/project/6102268/paulh/ginipa_pin           # Killarney (project space)
ROOT_T=/scratch/p/paulh/ginipa_pin                 # TamIA: /project is at 498K/500K files (group quota) -> scratch, see PREP_REPORT
BDIR_V=/scratch/paulh/ginipa_prep/bundles; BDIR_T=/scratch/p/paulh/ginipa_prep/bundles
vrun() { ssh -n -o BatchMode=yes klogin02 "ssh -o BatchMode=yes vulcan.alliancecan.ca $(printf '%q' "bash -lc $(printf '%q' "$1")")"; }
trun() { ssh -n -o BatchMode=yes klogin02 "ssh -o BatchMode=yes vulcan.alliancecan.ca $(printf '%q' "ssh -o BatchMode=yes tamia.alliancecan.ca $(printf '%q' "bash -lc $(printf '%q' "$1")")")"; }
[ "$(hostname -s)" = klogin02 ] && vrun() { ssh -n -o BatchMode=yes vulcan.alliancecan.ca "bash -lc $(printf '%q' "$1")"; }
[ "$(hostname -s)" = klogin02 ] && trun() { ssh -n -o BatchMode=yes vulcan.alliancecan.ca "ssh -o BatchMode=yes tamia.alliancecan.ca $(printf '%q' "bash -lc $(printf '%q' "$1")")"; }

# verification snippet, identical on every cluster: prints "<HEAD> <dirty count> <config md5> <tree md5>"
verify_snip() {  # $1 pin dir, $2 config name
  printf '%s' "cd '$1' 2>/dev/null || { echo MISSING; exit 0; }; h=\$(git rev-parse HEAD); d=\$(git status --porcelain | wc -l); "
  printf '%s' "c=\$(md5sum < 'auglab/configs/$2' 2>/dev/null | cut -c1-32); t=\$(git ls-files -z | xargs -0 md5sum | md5sum | cut -c1-32); echo \"\$h \$d \${c:-NOCONFIG} \$t\""
}

if [ "${1:-}" = "--check" ]; then
  PIN_CONF_TXT="$(vrun "cat ${VREPO}/scripts/cluster/ginipa/PIN.conf")"
  g() { printf '%s\n' "${PIN_CONF_TXT}" | awk -F'\t' -v k="$1" '$1==k{print $2}'; }
  sha="$(g auglab_commit)"; cfg="$(g config)"; s12="${sha:0:12}"
  echo "PIN.conf: commit ${sha} config ${cfg} md5 $(g config_md5) tree $(g tree_md5)"
  echo "killarney $(bash -c "$(verify_snip "${ROOT_K}/${s12}/AugLab" "${cfg}")")"
  echo "vulcan    $(vrun "$(verify_snip "${ROOT_V}/${s12}/AugLab" "${cfg}")")"
  echo "tamia     $(trun "$(verify_snip "${ROOT_T}/${s12}/AugLab" "${cfg}")")"
  exit 0
fi

REF="${1:-gin-ipa}"; CFG="${2:-transform_params_gpu_ginipa.json}"
sha="$(git -C "${SRC}" rev-parse --verify "${REF}^{commit}")"; s12="${sha:0:12}"
git -C "${SRC}" cat-file -e "${sha}:auglab/configs/${CFG}" || { echo "config ${CFG} not in ${sha}"; exit 1; }
git -C "${SRC}" cat-file -e "${sha}:auglab/trainers/nnUNetTrainerDAExtGPUGinIpa.py" || { echo "GinIpa trainer not in ${sha}"; exit 1; }
echo "[repin] source ${SRC} ref ${REF} -> ${sha}; config ${CFG}"
[ -z "$(git -C "${SRC}" status --porcelain)" ] || echo "[repin] NOTE: the source clone has uncommitted changes -- they are NOT shipped (bundle = commits only)"
B="ginipa_${s12}.bundle"; LB="/scratch/paulh/auglab_ginipa/bundles/${B}"; mkdir -p "$(dirname "${LB}")"
git -C "${SRC}" bundle create -q "${LB}" --all
git -C "${SRC}" bundle verify -q "${LB}" >/dev/null
echo "[repin] bundle ${LB} ($(stat -c %s "${LB}") bytes, md5 $(md5sum < "${LB}" | cut -c1-32))"
# ship: Killarney -> Vulcan scratch -> TamIA scratch
vrun "mkdir -p ${BDIR_V}"
ssh -o BatchMode=yes klogin02 "ssh -o BatchMode=yes vulcan.alliancecan.ca 'cat > ${BDIR_V}/${B}'" < "${LB}"
trun "mkdir -p ${BDIR_T}"
vrun "ssh -o BatchMode=yes tamia.alliancecan.ca 'cat > ${BDIR_T}/${B}' < ${BDIR_V}/${B}"
lm="$(md5sum < "${LB}" | cut -c1-32)"
[ "$(vrun "md5sum < ${BDIR_V}/${B} | cut -c1-32")" = "${lm}" ] && [ "$(trun "md5sum < ${BDIR_T}/${B} | cut -c1-32")" = "${lm}" ] \
  || { echo "[repin] bundle md5 differs after transfer"; exit 1; }

install_snip() {  # $1 bundle path, $2 pin root
  local d="$2/${s12}/AugLab"
  printf '%s' "set -e; d='${d}'; if [ -d \"\$d/.git\" ]; then echo \"exists \$d\"; else mkdir -p \"\$(dirname \"\$d\")\"; "
  printf '%s' "git clone -q --no-checkout '$1' \"\$d\"; git -C \"\$d\" checkout -q --detach ${sha}; git -C \"\$d\" remote remove origin; echo \"installed \$d\"; fi"
}
echo "killarney: $(bash -c "$(install_snip "${LB}" "${ROOT_K}")")"
echo "vulcan:    $(vrun "$(install_snip "${BDIR_V}/${B}" "${ROOT_V}")")"
echo "tamia:     $(trun "$(install_snip "${BDIR_T}/${B}" "${ROOT_T}")")"

K="$(bash -c "$(verify_snip "${ROOT_K}/${s12}/AugLab" "${CFG}")")"
V="$(vrun "$(verify_snip "${ROOT_V}/${s12}/AugLab" "${CFG}")")"
T="$(trun "$(verify_snip "${ROOT_T}/${s12}/AugLab" "${CFG}")")"
printf 'verify killarney %s\nverify vulcan    %s\nverify tamia     %s\n' "$K" "$V" "$T"
read -r kh kd kc kt <<<"$K"
for x in "$K" "$V" "$T"; do
  read -r h d c t <<<"$x"
  [ "$h" = "${sha}" ] && [ "$d" = 0 ] && [ "$c" = "$kc" ] && [ "$t" = "$kt" ] && [ "$c" != NOCONFIG ] || { echo "[repin] VERIFY FAILED ($x)"; exit 1; }
done
TSV="$(printf 'key\tvalue\nauglab_commit\t%s\nauglab_source\tKillarney %s (branch gin-ipa, never pushed), bundle %s md5 %s\nconfig\t%s\nconfig_md5\t%s\ntree_md5\t%s\npin_vulcan\t%s\npin_killarney\t%s\npin_tamia\t%s\npinned_at\t%s\n' \
  "${sha}" "${SRC}" "${B}" "${lm}" "${CFG}" "${kc}" "${kt}" "${ROOT_V}/${s12}/AugLab" "${ROOT_K}/${s12}/AugLab" "${ROOT_T}/${s12}/AugLab" "$(date -u '+%F %T UTC')")"
printf '%s\n' "${TSV}" | ssh -o BatchMode=yes klogin02 "ssh -o BatchMode=yes vulcan.alliancecan.ca 'cat > ${VREPO}/scripts/cluster/ginipa/PIN.conf'"
echo "[repin] VERIFY PASSED on all three clusters; wrote ${VREPO}/scripts/cluster/ginipa/PIN.conf:"; printf '%s\n' "${TSV}"
echo "[repin] next (Vulcan): commit PIN.conf (skill git-workflow), then 'bash scripts/cluster/ginipa/ginipa.sh runtable' and 'record'."
