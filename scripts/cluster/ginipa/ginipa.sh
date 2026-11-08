#!/usr/bin/env bash
# GIN-IPA competitor arm (ginipa, METHOD auglab_ginipa, category auglab): run table, record, submit (+ dry run), status.
# Same structure as the flatauglab arm (scripts/cluster/flatauglab/launch.sh + vulcan_move.sh + killarney_move.sh), with the AugLab
# code taken from a separate PIN (scripts/cluster/ginipa/PIN.conf, made by repin.sh) instead of the shared sub-workspaces checkout.
#
# Units: a unit = one Slurm job chain. TamIA unit = one whole h100:4 node, ONE fold per GPU (GIN-IPA needs ~2.7x the memory of the
# parent trainer: two folds do not fit one H100); Vulcan / Killarney unit = one fold on one L40S. Every unit is a chain of SEGMENTS
# (afterany) of SEG_TIME each; the recorded fold command resumes from checkpoint_latest at runtime (train_common.sh pack mode), so a
# chain survives the 24 h limit and a fold can move cluster (resume with the same RUN_ID, as vulcan_move.sh / killarney_move.sh did).
# The dispatch plan is scripts/cluster/ginipa/plan.conf (one line per unit); the probe plan is plan_probe.conf.
#
#   bash scripts/cluster/ginipa/ginipa.sh runtable              # Vulcan: new arm root + run_table.tsv from PIN.conf + plan.conf (no jobs)
#   bash scripts/cluster/ginipa/ginipa.sh record <cluster> [UNIT...]  # vulcan|killarney on Vulcan, tamia on TamIA: record + pin + verify
#   bash scripts/cluster/ginipa/ginipa.sh submit <cluster> [--dry-run] [UNIT...]   # vulcan|tamia here; killarney: killarney.sh submit
#   bash scripts/cluster/ginipa/ginipa.sh move <unit> [--from killarney] [--dry-run]   # Vulcan: TamIA stall -> resume that unit's folds on Vulcan L40S
#   bash scripts/cluster/ginipa/ginipa.sh mirror tamia            # Vulcan: copy the run table to TamIA's arm root
#   bash scripts/cluster/ginipa/ginipa.sh retarget <V_unit|K_unit> [--dry-run]   # Vulcan: never-started unit Vulcan <-> Killarney
#   bash scripts/cluster/ginipa/ginipa.sh status [cluster]       # epoch, median s/epoch, peak VRAM, final, per recorded fold
#   bash scripts/cluster/ginipa/ginipa.sh shims check|install    # the 9 additive <DS>GinIpaTrainers.py shims in this cluster's venv
# PROBE=1: probe root + plan_probe.conf, throwaway results base <scratch>/_ginipa_probe, every segment = `timeout PROBE_SECONDS`
#   (default 2400 s) inside a 50-min job; the fold commands keep the policy epochs (the timeout ends them mid-training, so nnU-Net's
#   end-of-training validation never runs). GINIPA_ROOT_FILE overrides the state file.
# Refusals: the run table's commit / config md5 must equal PIN.conf and the pin on disk (pin_env.sh), every recorded command must have
# the pin FIRST on PYTHONPATH, the pinned config path + md5, the run table's trainer / epochs / RUN_ID / fold / dataset loader default,
# no val config, seed offset 0, results under the setting's tree (probe: under _ginipa_probe), and a submit needs VERIFY_PASSED.
set -euo pipefail
GDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${GDIR}/../../.." && pwd)"; cd "${REPO}"
source "${GDIR}/pin_env.sh"            # GINIPA_CLUSTER / _PIN / _COMMIT / _CONFIG / _CONFIG_MD5 (refuses on a bad pin)
METHOD=auglab_ginipa; TAG=ginipa
case "${GINIPA_CLUSTER}" in tamia) RB=/scratch/p/paulh ;; *) RB=/scratch/paulh ;; esac
export SCRATCH="${RB}"
STATE="${GINIPA_ROOT_FILE:-${RB}/_ginipa_root${PROBE:+_probe}.txt}"
PLAN="${GINIPA_PLAN:-${GDIR}/plan${PROBE:+_probe}.conf}"
MODE="${1:?usage: runtable|mirror|record|submit|move|retarget|status|shims|body}"; shift
T=benchmark/02_tasks
# setting | scripts dir (under benchmark/02_tasks) | train wrapper | TamIA env (scripts/cluster/) | trainer | epochs | dataset name
B=brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma; C=abdomen_healthy/chaos/5_scripts_chaos; O=brain_healthy/on-harmony/5_scripts_on-harmony
M=brain_ms/open-ms/5_scripts_open-ms; TF=mandible_healthy/toothfairy2/5_scripts_toothfairy2; I=breast_cancer/ispy2/5_scripts_ispy2
P=pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic; A=liver_hcc/atlas-liver-hcc/5_scripts_atlas-liver-hcc; PS=pancreas_disease/pansegdata/5_scripts_pansegdata
SETTINGS=(
"brats_t1n|$B|04_56_train_t1n_${METHOD}.sh|tamia_env.sh|nnUNetTrainerBraTS2024GliomaGinIpa|2500|brats2024-glioma"
"brats_t2w|$B|04_59_train_t2w_${METHOD}.sh|tamia_env.sh|nnUNetTrainerBraTS2024GliomaT2wGinIpa|2500|brats2024-glioma"
"brats_t2f|$B|04_61_train_t2f_${METHOD}.sh|tamia_env.sh|nnUNetTrainerBraTS2024GliomaT2fGinIpa|2500|brats2024-glioma"
"brats_t1c|$B|04_79_train_t1c_${METHOD}.sh|tamia_env.sh|nnUNetTrainerBraTS2024GliomaT1cGinIpa|2500|brats2024-glioma"
"chaos_t1in|$C|04_91_train_t1in_${METHOD}.sh|tamia_env_chaos.sh|nnUNetTrainerCHAOSGinIpa|200|chaos"
"chaos_t2spir|$C|04_92_train_t2spir_${METHOD}.sh|tamia_env_chaos.sh|nnUNetTrainerCHAOSGinIpa|200|chaos"
"onh_T1w|$O|04_91_train_t1w_${METHOD}.sh|tamia_env_onharmony.sh|nnUNetTrainerOnHarmonyGinIpa|2000|on-harmony"
"onh_T2w|$O|04_92_train_t2w_${METHOD}.sh|tamia_env_onharmony.sh|nnUNetTrainerOnHarmonyGinIpa|2000|on-harmony"
"onh_dwi_ap|$O|04_93_train_dwi_ap_${METHOD}.sh|tamia_env_onharmony.sh|nnUNetTrainerOnHarmonyGinIpa|2000|on-harmony"
"openms_flair|$M|04_71_train_flair_${METHOD}.sh|tamia_env_openms.sh|nnUNetTrainerOpenMSGinIpa|2000|open-ms"
"openms_t1w|$M|04_72_train_t1w_${METHOD}.sh|tamia_env_openms.sh|nnUNetTrainerOpenMSGinIpa|2000|open-ms"
"tf2_cbct|$TF|04_27_train_cbct_${METHOD}.sh|tamia_env_toothfairy2.sh|nnUNetTrainerToothFairy2GinIpa|1000|toothfairy2"
"ispy2_t1wce|$I|04_46_train_t1wce_${METHOD}.sh|tamia_env_ispy2.sh|nnUNetTrainerISPY2GinIpa|1000|ispy2"
"ispy2_t2w|$I|04_47_train_t2w_${METHOD}.sh|tamia_env_ispy2.sh|nnUNetTrainerISPY2GinIpa|1000|ispy2"
"pelvic_ct|$P|04_43_train_ct_${METHOD}.sh|tamia_env_totalseg-pelvic.sh|nnUNetTrainerTotalsegPelvicGinIpa|200|totalseg-pelvic"
"pelvic_mri|$P|04_44_train_mri_${METHOD}.sh|tamia_env_totalseg-pelvic.sh|nnUNetTrainerTotalsegPelvicGinIpa|200|totalseg-pelvic"
"atlas_t1w|$A|04_23_train_t1w_${METHOD}.sh|tamia_env_atlas-liver-hcc.sh|nnUNetTrainerAtlasLiverHCCGinIpa|2000|atlas-liver-hcc"
"panseg_t1wce|$PS|04_43_train_t1wce_${METHOD}.sh|tamia_env_pansegdata.sh|nnUNetTrainerPANSEGDATAGinIpa|2000|pansegdata"
"panseg_t2w|$PS|04_44_train_t2w_${METHOD}.sh|tamia_env_pansegdata.sh|nnUNetTrainerPANSEGDATAGinIpa|2000|pansegdata"
)
# probe-only parent rows (same setting, the parent trainer = <DS>AugLabDefault with the auglab_default config, for the slowdown factor)
PARENT_ROWS=(
"brats_t1n_parent|$B|04_21_train_t1n_auglab_default.sh|tamia_env.sh|nnUNetTrainerBraTS2024GliomaAugLabDefault|2500|brats2024-glioma"
)
setting_row() { local s; for s in "${SETTINGS[@]}" "${PARENT_ROWS[@]}"; do [ "${s%%|*}" = "$1" ] && { echo "$s"; return 0; }; done; return 1; }
loader_default() { local c="${T}/$1/04_train/04_00_common.sh"
  echo "$(sed -n 's/^DA_WORKERS_DEFAULT="\{0,1\}\([0-9]*\).*/\1/p' "$c" | head -1)/$(sed -n 's/^NNUNET_COMPILE_DEFAULT="\{0,1\}\([0-9]*\).*/\1/p' "$c" | head -1)"; }
clean_cmd() { sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$1"; }
getv() { printf '%s\n' "$1" | sed -n "s/^ *export $2=//p" | head -1 | tr -d "'"; }

# ------------------------------------------------------------------------------------------------------------------- shims
if [ "${MODE}" = shims ]; then
  V="$(ls -d .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer)"; act="${1:-check}"; bad=0
  SH="$(find "${REPO}/${T}" -path '*/02_nnunet/*GinIpaTrainers.py' | sort)"
  [ "$(echo "${SH}" | grep -c .)" = 9 ] || { echo "expected 9 shims in the repo, found: ${SH}"; exit 1; }
  # 1. import test of the REPO copies (by file path, as nnU-Net would import them): a strict no-op without the pin, all 9 classes
  #    registered with it (login node: imports only, a few CPU-seconds)
  for withpin in 0 1; do
    pp="${REPO}"; [ "${withpin}" = 1 ] && pp="${GINIPA_PIN}:${REPO}"
    mkdir -p "${RB}/_ginipa_shimtest_cwd"   # empty cwd + python -P: never import stray modules from a shared /tmp (TamIA has one)
    out="$(cd "${RB}/_ginipa_shimtest_cwd" && NNUNET_PROJECT_ROOT="${REPO}" PYTHONPATH="${pp}" "${REPO}/.venv/bin/python" -P - ${SH} 2>&1 <<'PY'
import importlib.util, os, sys
n0 = len(sys.path)
for f in sys.argv[1:]:
    spec = importlib.util.spec_from_file_location("ginipa_shimtest_" + os.path.basename(f)[:-3], f)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    names = sorted(n for n in dir(m) if n.startswith("nnUNetTrainer") and n.endswith("GinIpa"))
    print("MOD", os.path.basename(f), "classes:", ",".join(names) or "-")
print("SYSPATH_ADDED", len(sys.path) - n0)
PY
)" || true
    echo "--- import test (repo copies), pin on PYTHONPATH: ${withpin}"; printf '%s\n' "${out}" | sed 's/^/    /'
    if [ "${withpin}" = 0 ]; then
      printf '%s\n' "${out}" | grep -q 'classes: [A-Za-z]' && { echo "    NOT A NO-OP without the pin"; bad=1; }
      printf '%s\n' "${out}" | grep -q '^SYSPATH_ADDED 0$' || { echo "    sys.path changed without the pin"; bad=1; }
      printf '%s\n' "${out}" | grep -qi 'error\|Traceback\|failed' && { echo "    ERROR without the pin"; bad=1; }
    else
      [ "$(printf '%s\n' "${out}" | grep -c '^MOD .* classes: nnUNetTrainer')" = 9 ] || { echo "    not all 9 shims register with the pin"; bad=1; }
      printf '%s\n' "${out}" | grep -q 'import failed\|Traceback' && { echo "    a shim failed to import with the pin"; bad=1; }
    fi
  done
  [ "${bad}" = 0 ] || { echo "SHIMS NOT OK (${GINIPA_CLUSTER}): import test failed -- nothing installed"; exit 1; }
  # 2. installed copies in this cluster's venv (additive: a new file name per dataset, never an existing shim)
  for f in ${SH}; do
    dst="${V}/$(basename "$f")"
    if [ -f "${dst}" ] && cmp -s "$f" "${dst}"; then echo "ok        ${dst}"
    elif [ "${act}" = install ]; then
      [ ! -e "${dst}" ] || { echo "REFUSING: ${dst} exists with other content"; exit 1; }
      cp "$f" "${dst}"; echo "installed ${dst}"
    else echo "MISSING   ${dst}"; bad=1; fi
  done
  [ "${bad}" = 0 ] && echo "SHIMS OK (${GINIPA_CLUSTER})" || { echo "SHIMS NOT INSTALLED (${GINIPA_CLUSTER}): ginipa.sh shims install"; exit 1; }
  exit 0
fi

# ---------------------------------------------------------------------------------------------------------------- runtable
if [ "${MODE}" = runtable ]; then
  [ "${GINIPA_CLUSTER}" = vulcan ] || { echo "runtable is made on Vulcan"; exit 1; }
  if [ -f "${STATE}" ] && [ -d "$(cat "${STATE}")" ]; then
    R="$(cat "${STATE}")"; ls "${R}"/*/*/JOBS.tsv >/dev/null 2>&1 && { echo "REFUSING: ${R} already has submitted units"; exit 1; }
    [ "${1:-}" = --force ] || { echo "REFUSING: ${R} exists (nothing submitted): pass --force to rebuild its run table (e.g. after repin.sh)"; exit 1; }
    TS="$(cat "${R}/TS")"; mv "${R}/run_table.tsv" "${R}/run_table.tsv.bak_$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
  else TS="$(date +%Y%m%d_%H%M%S)"; R="${RB}/${TAG}${PROBE:+_probe}_${TS}"; mkdir -p "${R}"; echo "${TS}" > "${R}/TS"; echo "${R}" > "${STATE}"; fi
  python3 - "${PLAN}" "${R}/run_table.tsv" "${TS}" "${METHOD}" "${GINIPA_COMMIT}" "${GINIPA_CONFIG}" "${GINIPA_CONFIG_MD5}" "${PROBE:-}" \
      "$(for s in "${SETTINGS[@]}" "${PARENT_ROWS[@]}"; do IFS='|' read -r n d tw te tr ep dn <<<"$s"; echo "$s|$(loader_default "$d")"; done)" <<'PY'
import sys
plan, out, ts, method, commit, cfg, md5, probe, rows = sys.argv[1:10]
S = {}
for r in rows.splitlines():
    n, d, tw, te, tr, ep, dn, ld = r.split("|"); S[n] = (d, tw, te, tr, ep, dn, ld)
units, seen = [], {}
for ln in open(plan):
    ln = ln.split("#", 1)[0].strip()
    if not ln: continue
    unit, cl, seg, st, mem = [x.strip() for x in ln.split("|")]
    assert cl in ("tamia", "vulcan", "killarney"), ln
    ms = mem.split()
    assert (cl == "tamia" and 1 <= len(ms) <= 4) or (cl != "tamia" and len(ms) == 1), f"unit {unit}: tamia 1-4 folds (one per GPU), L40S exactly 1"
    gpus = []
    for m in ms:
        sf, _, g = m.partition("@"); s, f = sf.split(":")
        g = int(g or 0); assert s in S, s; assert f in ("0", "1", "2"), m
        assert (s, f) not in seen, f"{s} fold {f} twice"; seen[(s, f)] = unit
        assert (cl == "tamia" and 0 <= g <= 3 and g not in gpus) or (cl != "tamia" and g == 0), f"unit {unit}: GPU {g} (one fold per GPU)"
        gpus.append(g); units.append((s, f, unit, cl, g, seg, st))
if not probe:
    miss = [(s, f) for s in S if not s.endswith("_parent") for f in "012" if (s, f) not in seen]
    assert not miss, f"plan misses {miss}"
hdr = "setting\tfold\tdataset\ttraining_contrast\tepochs\ttrainer\tloader_DA/compile\tcluster\tunit\tgpu\tsegments\tseg_time\tauglab_commit\tconfig\tconfig_md5\trun_id\ttrain_wrapper\tscripts_dir"
with open(out, "w") as fh:
    fh.write(hdr + "\n")
    for s, f, unit, cl, g, seg, st in units:
        d, tw, te, tr, ep, dn, ld = S[s]
        tc = s.split("_", 1)[1].replace("_parent", "")
        m = "auglab_default" if s.endswith("_parent") else method
        fh.write("\t".join([s, f, dn, tc, ep, tr, ld, cl, unit, str(g), seg, st, commit, cfg, md5, f"{dn}_{tc}_{m}_{ts}", tw, d]) + "\n")
print(f"run table: {len(units)} folds, {len(set(u[2] for u in units))} units -> {out}")
PY
  { echo "runtable $(date '+%F %T') repo ${REPO} HEAD $(git rev-parse HEAD) dirty=$(git status --porcelain --untracked-files=no | wc -l) plan ${PLAN} ($(md5sum < "${PLAN}" | cut -c1-32))"
    echo "pin ${GINIPA_COMMIT} config ${GINIPA_CONFIG} md5 ${GINIPA_CONFIG_MD5} (PIN.conf $(md5sum < "${GDIR}/PIN.conf" | cut -c1-32))"; } >> "${R}/LAUNCH_INFO"
  cp "${PLAN}" "${GDIR}/PIN.conf" "${R}/"; column -t -s$'\t' "${R}/run_table.tsv" | cut -c1-220
  exit 0
fi

[ -f "${STATE}" ] || { echo "no arm root yet (${STATE}): run 'ginipa.sh runtable' on Vulcan (and copy the root to TamIA/Killarney)"; exit 1; }
R="$(cat "${STATE}")"; RT="${R}/run_table.tsv"; [ -f "${RT}" ] || { echo "no ${RT}"; exit 1; }
# the run table must name the pin this cluster verified
while IFS=$'\t' read -r s f _ _ _ _ _ _ _ _ _ _ cm cfg md5 _; do
  [ "${cm}" = "${GINIPA_COMMIT}" ] && [ "${cfg}" = "${GINIPA_CONFIG}" ] && [ "${md5}" = "${GINIPA_CONFIG_MD5}" ] \
    || { echo "REFUSING: run table row ${s}:${f} pins ${cm:0:12}/${cfg}/${md5} but PIN.conf + disk say ${GINIPA_COMMIT:0:12}/${GINIPA_CONFIG}/${GINIPA_CONFIG_MD5}"; exit 1; }
done < <(tail -n +2 "${RT}")
TS="$(cat "${R}/TS")"
rows_of() { awk -F'\t' -v c="$1" -v u="$2" 'NR>1 && $8==c && (u=="" || $9==u)' "${RT}"; }
units_of() { awk -F'\t' -v c="$1" 'NR>1 && $8==c {print $9}' "${RT}" | awk '!s[$0]++'; }

# ------------------------------------------------------------------------------------------------------------------ mirror
if [ "${MODE}" = mirror ]; then   # Vulcan -> TamIA: the arm root's run table (+ TS, plan, PIN.conf) and TamIA's state file
  [ "${GINIPA_CLUSTER}" = vulcan ] && [ "${1:-}" = tamia ] || { echo "usage (Vulcan): ginipa.sh mirror tamia   (Killarney: killarney.sh prep)"; exit 1; }
  TR_="${R/#\/scratch\/paulh\//\/scratch\/p\/paulh\/}"; TST="/scratch/p/paulh/$(basename "${STATE}")"
  tar cf - -C "${R}" run_table.tsv TS "$(basename "${PLAN}")" PIN.conf LAUNCH_INFO | ssh -o BatchMode=yes tamia.alliancecan.ca \
    "mkdir -p '${TR_}' && tar xf - -C '${TR_}' && echo '${TR_}' > '${TST}' && md5sum '${TR_}/run_table.tsv'"
  echo "local  $(md5sum "${RT}")"; echo "[mirror] TamIA root ${TR_} (state ${TST})"; exit 0
fi

# ------------------------------------------------------------------------------------------------------------------ record
verify_cmd() {   # $1 cmd file, $2 run-table row, $3 cluster the job runs on, $4 pin dir on that cluster, $5 repo there
  local c res pre cfgp vcfg ep da cmp_ so pp tr fl ok
  IFS=$'\t' read -r s f dn tc ep_ tr_ ld cl unit g seg st cm cfg md5 rid tw sd <<<"$2"
  c="$(clean_cmd "$1")"
  res=$(getv "$c" nnUNet_results); pre=$(getv "$c" nnUNet_preprocessed); cfgp=$(getv "$c" AUGLAB_PARAMS_GPU_JSON); vcfg=$(getv "$c" AUGLAB_VAL_PARAMS_GPU_JSON)
  ep=$(getv "$c" NNUNET_NUM_EPOCHS); da=$(getv "$c" nnUNet_n_proc_DA); cmp_=$(getv "$c" nnUNet_compile); so=$(getv "$c" NNUNET_SEED_OFFSET); pp=$(getv "$c" PYTHONPATH)
  tr=$(printf '%s\n' "$c" | grep -o '\-tr [A-Za-z0-9_]*' | head -1 | cut -d' ' -f2)
  fl=$(printf '%s\n' "$c" | grep -oE 'nnUNetv2_train [0-9A-Za-z_]+ 3d_fullres [0-9]+' | head -1 | awk '{print $4}')
  ok=OK
  [[ "${pp}" == "$4:"* ]] || ok="${ok},PIN_NOT_FIRST_ON_PYTHONPATH(${pp})"
  [ "${cfgp}" = "$4/auglab/configs/${cfg}" ] || ok="${ok},CFG_PATH(${cfgp})"
  [ "$3" != "${GINIPA_CLUSTER}" ] || { [ -f "${cfgp}" ] && [ "$(md5sum < "${cfgp}" | cut -c1-32)" = "${md5}" ]; } || ok="${ok},CFG_MD5"
  [ -z "${vcfg}" ] || ok="${ok},VAL_CFG_SET(${vcfg})"
  [ "${tr}" = "${tr_}" ] || ok="${ok},TRAINER(${tr})"
  [ "${fl}" = "${f}" ] || ok="${ok},FOLD(${fl})"
  [ -n "${PROBE:-}" ] || [ "${ep}" = "${ep_}" ] || ok="${ok},EPOCHS(${ep})"
  [ "${da}/${cmp_}" = "${ld}" ] || ok="${ok},LOADER(${da}/${cmp_}!=${ld})"
  [ "${so:-0}" = 0 ] || ok="${ok},SEED_OFFSET(${so})"
  [[ "${res}" == */${tc}/auglab/${rid} ]] || ok="${ok},RESULTS(${res})"
  if [ -n "${PROBE:-}" ]; then [[ "${res}" == */_ginipa_probe/* ]] || ok="${ok},PROBE_NOT_THROWAWAY"
  else [[ "${res}" == */_ginipa_probe/* ]] && ok="${ok},PROBE_DIR_IN_REAL_RUN"; fi
  [ "$3" != "${GINIPA_CLUSTER}" ] || [ -d "${pre}" ] || ok="${ok},NO_PREPROC(${pre})"
  [ "$3" != "${GINIPA_CLUSTER}" ] || [ ! -e "${res}" ] || [ -n "${ALLOW_EXISTING:-}" ] || ok="${ok},RESULTS_DIR_EXISTS"
  grep -q 'sub-workspaces' "$1" && ok="${ok},SHARED_AUGLAB_PATH"
  [[ "$5" == /project/6102268/* ]] && grep -q '/project/aip-jcohen' "$1" && ok="${ok},VULCAN_PATH_LEFT"
  printf '   %-6s %-14s f%s GPU%s %s\n      trainer=%s ep=%s loader=%s/%s seed_offset=%s\n      cfg=%s\n      PYTHONPATH=%s\n      results=%s\n' \
    "${ok}" "${s}" "${f}" "${g}" "$(basename "$1")" "${tr}" "${ep}" "${da}" "${cmp_}" "${so:-0}" "${cfgp}" "${pp}" "${res}"
  [ "${ok}" = OK ]
}
insert_pin() {   # put $2 first on the recorded command's (literal, single-quoted) PYTHONPATH; exactly one export must match
  python3 - "$1" "$2" <<'PY'
import sys
p, pin = sys.argv[1], sys.argv[2]
s = open(p).read(); key = "export PYTHONPATH=\\'"
if s.count(key) != 1:
    sys.exit(f"{p}: {s.count(key)} 'export PYTHONPATH=' lines (expect 1)")
if (key + pin + ":") not in s:
    s = s.replace(key, key + pin + ":", 1); open(p, "w").write(s)
PY
}

if [ "${MODE}" = record ]; then
  CL="${1:?record <vulcan|killarney|tamia> [UNIT...]}"; shift; SEL="${*:-$(units_of "${CL}")}"
  case "${CL}" in tamia) [ "${GINIPA_CLUSTER}" = tamia ] || { echo "record tamia on TamIA"; exit 1; } ;;
    vulcan|killarney) [ "${GINIPA_CLUSTER}" = vulcan ] || { echo "record ${CL} on Vulcan (killarney.sh prep rewrites the Killarney units)"; exit 1; } ;; esac
  bad=0
  for u in ${SEL}; do
    UD="${R}/${CL}/${u}"; [ -f "${UD}/index.tsv" ] && { echo "[record] ${CL}/${u} already recorded -- skipped"; continue; }
    mkdir -p "${UD}"
    while IFS=$'\t' read -r s f dn tc ep tr ld cl unit g seg st cm cfg md5 rid tw sd; do
      w="${T}/${sd}/04_train/${tw}"; [ -f "${w}" ] || { echo "MISSING ${w}"; exit 1; }
      envf="$(grep -o 'source "$(dirname "$0")/../00_utils/env[a-z0-9_]*\.sh"' "${w}" | sed 's|.*/00_utils/||; s|"$||')"
      echo "[record] ${CL}/${u} <- ${s} fold ${f} ($(basename "${w}"), ${envf})"
      (
        source "${T}/${sd}/00_utils/${envf}"
        if [ "${CL}" = tamia ]; then tenv="$(setting_row "${s}" | cut -d'|' -f4)"; source "scripts/cluster/${tenv}"; fi
        [ -z "${PROBE:-}" ] || { export PREDICTIONS_ROOT="${RB}/_ginipa_probe/${TS}/${s}/pred"; mkdir -p "${PREDICTIONS_ROOT}"; }
        export GINIPA_EXPECT_COMMIT="${cm}" GINIPA_EXPECT_CONFIG_MD5="${md5}"
        if [[ "${s}" == *_parent ]]; then   # probe-only parent: the competitors' auglab_default wrapper, shared AugLab (reference speed)
          TRAIN_FOLDS="${f}" RUN_JOB_PACK_DIR="${UD}" bash "${w}" "${rid}" < /dev/null
        else TRAIN_FOLDS="${f}" RUN_JOB_PACK_DIR="${UD}" bash "${w}" "${rid}" < /dev/null; fi
      ) > "${UD}/record_${s}_f${f}.log" 2>&1 || { echo "   record FAILED (${UD}/record_${s}_f${f}.log)"; tail -5 "${UD}/record_${s}_f${f}.log"; exit 1; }
    done < <(rows_of "${CL}" "${u}")
  done
  echo "=== pin + verify (every recorded fold of ${CL}) ==="
  for u in ${SEL}; do
    UD="${R}/${CL}/${u}"; n=0
    while IFS=$'\t' read -r cmd log name done_; do
      row="$(awk -F'\t' -v r="${name#fold?_}" -v f="${name:4:1}" 'NR>1 && $16==r && $2==f' "${RT}")"
      [ -n "${row}" ] || { echo "   ${name}: not in the run table"; bad=1; continue; }
      if [[ "$(cut -f1 <<<"${row}")" == *_parent ]]; then echo "   PARENT ${name} (shared AugLab, reference only)"; n=$((n+1)); continue; fi
      insert_pin "${cmd}" "${GINIPA_PIN}"
      verify_cmd "${cmd}" "${row}" "${GINIPA_CLUSTER}" "${GINIPA_PIN}" "${REPO}" || bad=1; n=$((n+1))
    done < "${UD}/index.tsv"
    [ "${n}" = "$(rows_of "${CL}" "${u}" | grep -c .)" ] || { echo "   ${u}: ${n} recorded folds != run table"; bad=1; }
  done
  if [ "${bad}" = 0 ]; then for u in ${SEL}; do date '+%F %T' > "${R}/${CL}/${u}/VERIFY_PASSED"; done; echo "VERIFY PASSED (${CL}: ${SEL})"
  else echo "VERIFY FAILED"; exit 1; fi
  exit 0
fi

# ------------------------------------------------------------------------------------------------------------------ submit
write_body() {   # $1 unit dir, $2 cluster -> prints the body path. One body per unit, reused by every segment of the chain.
  local UD="$1" body="$1/_unit_body.sh" ps="${PROBE_SECONDS:-2400}"
  cat > "${body}" <<EOS
#!/bin/bash
# GIN-IPA arm (ginipa) unit $(basename "${UD}") on $2 -- GENERATED by scripts/cluster/ginipa/ginipa.sh submit. One fold per GPU.
set -uo pipefail
cd '${REPO}'
# CUDA JIT cache on node-local disk: on 2026-11-07 a TamIA node's Lustre eviction left \$HOME/.nv/ComputeCache/index locked and
# every new GPU process on other nodes hung on it (ptlrpc_set_wait, 0% GPU) -- never share that cache between jobs.
export CUDA_CACHE_PATH="\${SLURM_TMPDIR:-/tmp}/cuda_compute_cache_\${SLURM_JOB_ID:-x}"
INDEX='${UD}/index.tsv'
all_done=1
while IFS=\$'\t' read -r cmd log name donefile; do [ -n "\${donefile}" ] && [ -f "\${donefile}" ] || all_done=0; done < "\${INDEX}"
[ "\${all_done}" = 1 ] && { echo "[ginipa] every fold has checkpoint_final -- nothing to do"; exit 0; }
# preflight: the pin this run table names, on disk, clean; every fold command puts it first and reads the pinned config
export GINIPA_EXPECT_COMMIT='${GINIPA_COMMIT}' GINIPA_EXPECT_CONFIG_MD5='${GINIPA_CONFIG_MD5}'
( source scripts/cluster/ginipa/pin_env.sh ) || { echo "[ginipa] PIN PREFLIGHT FAILED -- refusing"; exit 4; }
PYTHONPATH='${GINIPA_PIN}' .venv/bin/python -P - '${GINIPA_PIN}' '${GINIPA_PIN}/auglab/configs/${GINIPA_CONFIG}' '${GINIPA_CONFIG_MD5}' "\${INDEX}" <<'PY' || { echo "[ginipa] PREFLIGHT FAILED -- refusing"; exit 4; }
import hashlib, importlib, re, sys
pin, cfg, md5, index = sys.argv[1:5]
bad = []
for spec in ["auglab.trainers.nnUNetTrainerDAExtGPUGinIpa", "auglab.transforms.gpu.gin_ipa", "auglab.transforms.gpu.gin_ipa_orig",
             "auglab.trainers.nnUNetTrainerDAExt", "auglab.transforms.gpu.transforms", "auglab.configs"]:
    m = importlib.import_module(spec); f = getattr(m, "__file__", None) or list(m.__path__)[0]
    print("[ginipa] preflight", spec, "->", f); f.startswith(pin + "/") or bad.append(spec)
got = hashlib.md5(open(cfg, "rb").read()).hexdigest(); print("[ginipa] config", cfg, got); got == md5 or bad.append("config md5")
for line in open(index):
    if "_auglab_default_" in line.split("\t")[2]:
        print("[ginipa] fold cmd", line.split("\t")[2], "= probe PARENT reference (shared AugLab, default01-23), not checked"); continue
    c = open(line.split("\t")[0]).read().replace("\\\\'", "'")
    pp = re.search(r"export PYTHONPATH='([^']*)'", c); aj = re.search(r"export AUGLAB_PARAMS_GPU_JSON='([^']*)'", c)
    ok = bool(pp and pp.group(1).startswith(pin + ":") and aj and aj.group(1) == cfg)
    print("[ginipa] fold cmd", line.split("\t")[2], "pin first + pinned config:", ok); ok or bad.append(line.split("\t")[2])
sys.exit(1 if bad else 0)
PY
echo "[ginipa] host=\$(hostname) job=\${SLURM_JOB_ID:-?} unit=$(basename "${UD}") pin=$(git -C "${GINIPA_PIN}" rev-parse --short=12 HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -l 15 > '${UD}'/vram_\${SLURM_JOB_ID:-x}.csv 2>/dev/null &
SMI=\$!
read -ra GMAP < '${UD}/GPU_MAP'
declare -a PIDS NAMES; i=0
while IFS=\$'\t' read -r cmd log name donefile; do
  g="\${GMAP[\$i]}"; plog="\${cmd%.sh}.log"
  echo "[ginipa] launch \${name} on GPU \${g} -> \${plog}"
  CUDA_VISIBLE_DEVICES=\${g} ${PROBE:+timeout ${ps}} bash "\${cmd}" >> "\${plog}" 2>&1 &
  PIDS[\$i]=\$!; NAMES[\$i]="\${name}"; i=\$((i + 1)); sleep 5
done < "\${INDEX}"
rc_all=0
for j in "\${!PIDS[@]}"; do if wait "\${PIDS[\$j]}"; then echo "[ginipa] OK   \${NAMES[\$j]}"; else rc=\$?; echo "[ginipa] EXIT \${rc} \${NAMES[\$j]}"; [ "\${rc}" = 124 ] || rc_all=1; fi; done
kill \${SMI} 2>/dev/null; echo "[ginipa] done rc_all=\${rc_all}"; exit \${rc_all}
EOS
  echo "${body}"
}
if [ "${MODE}" = body ]; then write_body "${1:?body <unit dir> <cluster>}" "${2:?cluster}"; exit 0; fi   # used by killarney.sh submit
if [ "${MODE}" = submit ]; then
  CL="${1:?submit <vulcan|tamia> [--dry-run] [UNIT...]}"; shift; DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
  [ "${CL}" = "${GINIPA_CLUSTER}" ] || { echo "submit ${CL} on ${CL} (killarney: scripts/cluster/ginipa/killarney.sh submit)"; exit 1; }
  SEL="${*:-$(units_of "${CL}")}"
  if [ "${CL}" = tamia ]; then   # launch rule: at most GINIPA_TAMIA_MAX_NODES (4) whole nodes with a live chain at once
    act=0; new=0
    for u in $(units_of tamia); do UD="${R}/tamia/${u}"; [ -f "${UD}/JOBS.tsv" ] || continue
      while IFS=$'\t' read -r _ _ _ d; do [ -n "$d" ] && [ -f "$d" ] || { act=$((act + 1)); break; }; done < "${UD}/index.tsv"; done
    for u in ${SEL}; do [ -f "${R}/tamia/${u}/JOBS.tsv" ] || new=$((new + 1)); done
    [ $((act + new)) -le "${GINIPA_TAMIA_MAX_NODES:-4}" ] || { echo "REFUSING: ${act} TamIA units still training + ${new} new > ${GINIPA_TAMIA_MAX_NODES:-4} nodes (launch rule)"; exit 1; }
    echo "[submit] TamIA nodes: ${act} active + ${new} new <= ${GINIPA_TAMIA_MAX_NODES:-4}"
  fi
  for u in ${SEL}; do
    UD="${R}/${CL}/${u}"
    [ -f "${UD}/VERIFY_PASSED" ] || { echo "REFUSING: ${CL}/${u} not recorded + verified"; exit 1; }
    [ "${DRY}" = 1 ] || [ ! -f "${UD}/JOBS.tsv" ] || { echo "[submit] ${u} already submitted ($(tr '\n' ' ' < "${UD}/JOBS.tsv")) -- skipped"; continue; }
    # GPU map in index.tsv order, from the run table
    : > "${UD}/GPU_MAP.new"; gm=""
    while IFS=$'\t' read -r cmd log name done_; do
      gm="${gm} $(awk -F'\t' -v r="${name#fold?_}" -v f="${name:4:1}" 'NR>1 && $16==r && $2==f {print $10}' "${RT}")"; done < "${UD}/index.tsv"
    echo "${gm# }" > "${UD}/GPU_MAP"; rm -f "${UD}/GPU_MAP.new"
    nseg="$(rows_of "${CL}" "${u}" | head -1 | cut -f11)"; st="$(rows_of "${CL}" "${u}" | head -1 | cut -f12)"
    [ -z "${PROBE:-}" ] || { nseg=1; st=00:50:00; }
    body="$(write_body "${UD}" "${CL}")"
    case "${CL}" in
      tamia)  RES=(--nodes=1 --gpus-per-node=h100:4 --cpus-per-task=48 --mem=0) ;;
      vulcan) RES=(--gres=gpu:l40s:1 --cpus-per-task=16 --mem=110G --exclude=rack02-06) ;;
    esac
    common=(--account=aip-jcohen "${RES[@]}" --time="${st}" --job-name="${TAG}${PROBE:+probe}_${u}" --output="${UD}/seg_%j.out")
    prev=""; jobs_=""
    for ((k=1; k<=nseg; k++)); do
      dep=(); [ -z "${prev}" ] || dep=(--dependency=afterany:"${prev}")
      if [ "${DRY}" = 1 ]; then echo "[dry] sbatch ${common[*]} ${dep[*]} ${body}"; prev="DRY${k}"
      else prev="$(sbatch --parsable "${common[@]}" "${dep[@]}" "${body}")"; sleep 2; fi
      jobs_="${jobs_:+${jobs_},}${prev}"
    done
    [ "${DRY}" = 1 ] || { printf '%s\t%s\t%s\t%s\n' "${u}" "${CL}" "${jobs_}" "$(date '+%F %T')" | tee -a "${UD}/JOBS.tsv" >> "${R}/JOBS_ALL.tsv"; }
    echo "[submit${DRY:+}] ${CL}/${u}: ${nseg} x ${st}: ${jobs_} (GPU map ${gm# })"
  done
  exit 0
fi

# -------------------------------------------------------------------------------------------------------------------- move
# Move a TamIA unit's folds to VULCAN L40S (TamIA fair-share stall: see memory reference_training_speed_by_cluster_20261030): the same
# RUN_IDs resume from TamIA's checkpoint_latest (<= 50 epochs repeated). Order (Vulcan login node, TamIA ssh master):
#   1. scancel the unit's TamIA chain (tail first);  2. ginipa.sh move <UNIT> [--dry-run]  (refuses while a ginipa_<UNIT> job is queued
#   on TamIA; rsync + md5 of checkpoint_latest / checkpoint_best per fold into the Vulcan run dir; run table rows -> cluster vulcan, unit
#   MV_<setting>_f<k>, backup kept);  3. ALLOW_EXISTING=1 ginipa.sh record vulcan MV_...;  4. ginipa.sh submit vulcan MV_...
# --from relay: Vulcan has no TamIA master (since 2026-11-07), so tamia_push.sh (Killarney login node, masters to both) copies the
#   unfinished folds through that node and writes .ginipa_moved_from_tamia; here only rows with that marker move (finished folds stay
#   TamIA rows: tamia_relay.sh brings them over for predict + eval).
if [ "${MODE}" = move ]; then
  [ "${GINIPA_CLUSTER}" = vulcan ] || { echo "move runs on Vulcan"; exit 1; }
  U="${1:?move <TamIA unit> [--from killarney] [--dry-run]}"; shift; DRY=0; SRC=tamia
  while [ $# -gt 0 ]; do case "$1" in --dry-run) DRY=1 ;; --from) SRC="$2"; shift ;; esac; shift; done
  if [ "${SRC}" = killarney ] || [ "${SRC}" = relay ]; then
    MK=.ginipa_moved_from_killarney; [ "${SRC}" = relay ] && MK=.ginipa_moved_from_tamia; SRCCL=killarney; [ "${SRC}" = relay ] && SRCCL=tamia
    # Killarney units are pushed by `killarney.sh push <unit>` (run on Killarney after scancel of the chain: rsync + md5, then a marker in each
    # Vulcan fold dir); here only check the markers + checkpoint_latest and switch the run table rows (same RUN_IDs) to Vulcan units.
    NEWU=""
    while IFS=$'\t' read -r s f dn tc ep tr ld cl unit g seg st cm cfg md5 rid tw sd; do
      envf="$(grep -o 'source "$(dirname "$0")/../00_utils/env[a-z0-9_]*\.sh"' "${T}/${sd}/04_train/${tw}" | sed 's|.*/00_utils/||; s|"$||')"
      VRD="$( source "${T}/${sd}/00_utils/${envf}" >/dev/null 2>&1; echo "${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab/${rid}" )"
      vfd="$(ls -d "${VRD}"/Dataset*/*__nnUNetPlans__3d_fullres/fold_${f} 2>/dev/null | head -1 || true)"
      if [ -z "${vfd}" ] || [ ! -f "${vfd}/${MK}" ] || [ ! -f "${vfd}/checkpoint_latest.pth" ]; then
        [ "${SRC}" = relay ] && { echo "[move] ${s} f${f}: no ${MK} -- stays a TamIA row (finished there; tamia_relay.sh copies it)"; continue; }
        echo "REFUSING: ${s} f${f}: no pushed fold (marker + checkpoint_latest) under ${VRD} -- run killarney.sh push ${U} on Killarney"; exit 1
      fi
      echo "[move] ${s} f${f}: $(head -1 "${vfd}/${MK}")"; NEWU="${NEWU} MV_${s}_f${f}"; MVF="${MVF:-} ${s}:${f}"
    done < <(rows_of "${SRCCL}" "${U}")
    [ -n "${NEWU}" ] || { echo "no pushed ${SRCCL} rows for unit ${U}"; exit 1; }
    if [ "${DRY}" = 0 ]; then
      mkdir -p "${R}/_launch"
      cp "${RT}" "${RT}.bak_move_${U}_$(date +%Y%m%d_%H%M%S)"
      awk -F'\t' -v OFS='\t' -v u="${U}" -v c="${SRCCL}" -v m="${MVF} " 'NR>1 && $8==c && $9==u && index(m, " "$1":"$2" ") {$8="vulcan"; $9="MV_"$1"_f"$2; $10=0} {print}' "${RT}" > "${RT}.new" && mv "${RT}.new" "${RT}"
      echo "$(date '+%F %T') moved ${SRCCL} unit ${U} (chain cancelled, pushed + md5-verified) -> Vulcan units${NEWU}" | tee -a "${R}/MOVES.log" "${R}/_launch/LAUNCH_LOG.txt" >> "${R}/LAUNCH_INFO"
    fi
    echo "next: ALLOW_EXISTING=1 bash scripts/cluster/ginipa/ginipa.sh record vulcan${NEWU} && bash scripts/cluster/ginipa/ginipa.sh submit vulcan${NEWU}"
    exit 0
  fi
  SSHT=(ssh -n -o BatchMode=yes -o ConnectTimeout=30 tamia.alliancecan.ca)
  q="$("${SSHT[@]}" "squeue -u ${USER} -h -o '%i %j'")" || { echo "TamIA unreachable"; exit 4; }
  if grep -q " ${TAG}_${U}\$" <<<"${q}"; then echo "REFUSING: TamIA still queues $(grep " ${TAG}_${U}\$" <<<"${q}" | awk '{print $1}' | tr '\n' ' ')-- scancel them (tail first)"; [ "${DRY}" = 1 ] || exit 1; fi
  declare -A TE=([brats]=tamia_env.sh [chaos]=tamia_env_chaos.sh [onh]=tamia_env_onharmony.sh [openms]=tamia_env_openms.sh [tf2]=tamia_env_toothfairy2.sh
    [ispy2]=tamia_env_ispy2.sh [pelvic]=tamia_env_totalseg-pelvic.sh [atlas]=tamia_env_atlas-liver-hcc.sh [panseg]=tamia_env_pansegdata.sh)
  Q2='for f in checkpoint_latest.pth checkpoint_best.pth; do [ -f $f ] && echo $f $(stat -c %s $f) $(md5sum < $f | cut -c1-32); done; true'
  NEWU=""
  while IFS=$'\t' read -r s f dn tc ep tr ld cl unit g seg st cm cfg md5 rid tw sd; do
    envf="$(grep -o 'source "$(dirname "$0")/../00_utils/env[a-z0-9_]*\.sh"' "${T}/${sd}/04_train/${tw}" | sed 's|.*/00_utils/||; s|"$||')"
    VRD="$( source "${T}/${sd}/00_utils/${envf}" >/dev/null 2>&1; echo "${PREDICTIONS_ROOT}/${MODEL_TYPE}/${TRAINING_CONTRAST}/auglab/${rid}" )"
    TRD="$("${SSHT[@]}" "bash -lc 'cd ${REPO} && export SCRATCH=/scratch/p/paulh && source ${T}/${sd}/00_utils/${envf} >/dev/null 2>&1 && source scripts/cluster/${TE[${s%%_*}]} >/dev/null 2>&1 && echo \${PREDICTIONS_ROOT}/\${MODEL_TYPE}/\${TRAINING_CONTRAST}/auglab/${rid}'")"
    tfd="$("${SSHT[@]}" "ls -d ${TRD}/Dataset*/*__nnUNetPlans__3d_fullres/fold_${f} 2>/dev/null | head -1")"
    [ -n "${tfd}" ] || { echo "REFUSING: no fold_${f} under TamIA ${TRD}"; exit 1; }
    vfd="${VRD}/${tfd#${TRD}/}"; echo "[move] ${s} f${f}: TamIA ${tfd} -> Vulcan ${vfd}"
    if [ "${DRY}" = 0 ]; then
      mkdir -p "${vfd}"
      rsync -a -e "ssh -o BatchMode=yes" --include='*.json' --exclude='*' "tamia.alliancecan.ca:$(dirname "${tfd}")/" "$(dirname "${vfd}")/"
      rsync -a -e "ssh -o BatchMode=yes" --exclude=wandb/ --exclude='validation/' "tamia.alliancecan.ca:${tfd}/" "${vfd}/"
      L="$(cd "${vfd}" && bash -c "${Q2}")"; Rm="$("${SSHT[@]}" "cd ${tfd} && ${Q2}")"
      [ "$L" = "$Rm" ] && [ -n "$L" ] || { echo "MISMATCH ${s} f${f}: Vulcan '$L' / TamIA '$Rm'"; exit 5; }
      echo "   md5 OK: $(echo "$L" | tr '\n' ' ')" | tee -a "${R}/MOVES.log"
    fi
    NEWU="${NEWU} MV_${s}_f${f}"
  done < <(rows_of tamia "${U}")
  [ -n "${NEWU}" ] || { echo "no TamIA rows for unit ${U}"; exit 1; }
  if [ "${DRY}" = 0 ]; then
    cp "${RT}" "${RT}.bak_move_${U}_$(date +%Y%m%d_%H%M%S)"
    awk -F'\t' -v OFS='\t' -v u="${U}" 'NR>1 && $8=="tamia" && $9==u {$8="vulcan"; $9="MV_"$1"_f"$2; $10=0} {print}' "${RT}" > "${RT}.new" && mv "${RT}.new" "${RT}"
    echo "$(date '+%F %T') moved TamIA unit ${U} -> Vulcan units${NEWU}" | tee -a "${R}/MOVES.log" >> "${R}/LAUNCH_INFO"
  fi
  echo "next: ALLOW_EXISTING=1 bash scripts/cluster/ginipa/ginipa.sh record vulcan${NEWU} && bash scripts/cluster/ginipa/ginipa.sh submit vulcan${NEWU}"
  echo "      then: bash scripts/cluster/ginipa/ginipa.sh mirror tamia   (post_poller.sh reads the updated run table: these folds become vulcan folds)"
  exit 0
fi

# ---------------------------------------------------------------------------------------------------------------- retarget
# retarget <V_unit> [--dry-run]: re-target a Vulcan unit that NEVER STARTED to Killarney (Vulcan queue too slow): refuses unless every job
# of its chain is CANCELLED (scancel them first) and no fold dir exists; the run table rows become cluster killarney, unit K_<rest> (same
# RUN_IDs, same folds); the old unit dir moves to <root>/_superseded_retarget/. Then: ginipa.sh record killarney K_...; on Killarney
# killarney.sh prep K_...; killarney.sh submit K_...  (the datasets must be staged on Killarney).
# retarget <K_unit> [--dry-run]: the INVERSE, for a retargeted unit that never started on Killarney (Killarney queue too slow). Order:
#   1. Killarney: scancel the unit's whole chain (one scancel call);  2. Killarney: killarney.sh release [--dry-run] K_...  (refuses unless
#   every job is CANCELLED, no seg_*.out, no fold dir; moves its Killarney unit dir aside and writes <root>/killarney/K_.../RELEASED_FROM_KILLARNEY
#   on Vulcan);  3. Vulcan: ginipa.sh retarget K_... [--dry-run]: needs that marker and the superseded, VERIFY_PASSED Vulcan unit V_<rest>
#   (every job CANCELLED), re-verifies its recorded fold commands (verify_cmd: pin, config md5, trainer, epochs, results dir, ...), refuses
#   if a fold dir exists on Vulcan; then the run table rows go back to cluster vulcan, unit V_<rest>, GPU 0 (backup kept), the V_ dir moves
#   back from _superseded_retarget/ (its cancelled JOBS.tsv kept as JOBS.tsv.cancelled_<ts>) and the Vulcan K_ dir moves to
#   _superseded_retarget/;  4. ginipa.sh submit vulcan [--dry-run] V_...   (same RUN_ID, same verified fold command; post_poller.sh reads the
#   run table, so the fold is a vulcan fold again).
if [ "${MODE}" = retarget ]; then
  [ "${GINIPA_CLUSTER}" = vulcan ] || { echo "retarget runs on Vulcan"; exit 1; }
  U="${1:?retarget <vulcan unit V_...|killarney unit K_...> [--dry-run]}"; DRY=0; [ "${2:-}" = --dry-run ] && DRY=1
  SUP="${R}/_superseded_retarget"; stamp="$(date +%Y%m%d_%H%M%S)"
  all_cancelled() {   # $1 JOBS.tsv: every job of every chain listed in it must be CANCELLED
    local j st; for j in $(cut -f3 "$1" | tr ',' ' '); do
      st="$(sacct -j "${j}" -X -n -o State | head -1 | tr -d ' ')"; [[ "${st}" == CANCELLED* ]] || { echo "REFUSING: job ${j} is ${st:-unknown} (scancel the whole chain first)"; return 1; }
    done; }
  if [[ "${U}" == K_* ]]; then   # ---- inverse: Killarney -> Vulcan
    KD="${R}/killarney/${U}"; VU="V_${U#K_}"; SD="${SUP}/${VU}"; UD="${R}/vulcan/${VU}"
    [ -d "${KD}" ] || { echo "no ${KD}"; exit 1; }
    [ -f "${KD}/RELEASED_FROM_KILLARNEY" ] || { echo "REFUSING: ${U} not released (Killarney: scancel its chain, then killarney.sh release ${U})"; exit 1; }
    [ -f "${SD}/VERIFY_PASSED" ] && [ -f "${SD}/index.tsv" ] && [ -f "${SD}/JOBS.tsv" ] || { echo "REFUSING: no verified superseded Vulcan unit ${SD}"; exit 1; }
    [ ! -e "${UD}" ] || { echo "REFUSING: ${UD} exists"; exit 1; }
    [ ! -e "${SUP}/${U}" ] || { echo "REFUSING: ${SUP}/${U} exists"; exit 1; }
    [ -n "$(rows_of killarney "${U}")" ] || { echo "REFUSING: no run table rows for killarney unit ${U}"; exit 1; }
    all_cancelled "${SD}/JOBS.tsv" || exit 1
    echo "[retarget] ${U}: released on Killarney: $(head -1 "${KD}/RELEASED_FROM_KILLARNEY")"
    bad=0; n=0
    while IFS=$'\t' read -r cmd log name done_; do
      c="${SD}/$(basename "${cmd}")"; [ "$(dirname "${cmd}")" = "${UD}" ] && [ -f "${c}" ] || { echo "REFUSING: ${name}: cmd ${cmd} not in ${UD} / ${c} missing"; exit 1; }
      row="$(awk -F'\t' -v r="${name#fold?_}" -v f="${name:4:1}" -v u="${U}" 'NR>1 && $16==r && $2==f && $8=="killarney" && $9==u' "${RT}")"
      [ -n "${row}" ] || { echo "REFUSING: ${name} is not a run table row of killarney unit ${U}"; exit 1; }
      res="$(getv "$(clean_cmd "${c}")" nnUNet_results)"
      if ls -d "${res}"/Dataset*/*/fold_"${name:4:1}" >/dev/null 2>&1; then echo "REFUSING: ${name} has a fold dir under ${res} on Vulcan"; exit 1; fi
      verify_cmd "${c}" "${row}" vulcan "${GINIPA_PIN}" "${REPO}" || bad=1; n=$((n+1))
    done < "${SD}/index.tsv"
    [ "${n}" = "$(rows_of killarney "${U}" | grep -c .)" ] || { echo "REFUSING: ${SD}: ${n} folds != run table rows of ${U}"; exit 1; }
    [ "${bad}" = 0 ] || { echo "REFUSING: re-verify of ${SD} failed (ALLOW_EXISTING=1 only if the run dir exists because of OTHER folds)"; exit 1; }
    if [ "${DRY}" = 1 ]; then echo "[dry] would: run table killarney/${U} -> vulcan/${VU}; mv ${SD} -> ${UD} (JOBS.tsv -> JOBS.tsv.cancelled_${stamp}); mv ${KD} -> ${SUP}/${U}"; exit 0; fi
    cp "${RT}" "${RT}.bak_retarget_${U}_${stamp}"
    awk -F'\t' -v OFS='\t' -v u="${U}" -v nu="${VU}" 'NR>1 && $8=="killarney" && $9==u {$8="vulcan"; $9=nu; $10=0} {print}' "${RT}" > "${RT}.new" && mv "${RT}.new" "${RT}"
    mv "${SD}" "${UD}"; mv "${UD}/JOBS.tsv" "${UD}/JOBS.tsv.cancelled_${stamp}"; mv "${KD}" "${SUP}/${U}"
    echo "$(date '+%F %T') retargeted never-started Killarney unit ${U} (chain cancelled, released) -> Vulcan unit ${VU} (restored, re-verified)" | tee -a "${R}/LAUNCH_INFO" "${R}/_launch/LAUNCH_LOG.txt"
    echo "next: bash scripts/cluster/ginipa/ginipa.sh submit vulcan --dry-run ${VU} && bash scripts/cluster/ginipa/ginipa.sh submit vulcan ${VU}"
    exit 0
  fi
  UD="${R}/vulcan/${U}"; [ -f "${UD}/JOBS.tsv" ] || { echo "no ${UD}/JOBS.tsv"; exit 1; }
  all_cancelled "${UD}/JOBS.tsv" || exit 1
  while IFS=$'\t' read -r cmd log name done_; do
    res="$(getv "$(clean_cmd "${cmd}")" nnUNet_results)"
    if ls -d "${res}"/Dataset*/*/fold_"${name:4:1}" >/dev/null 2>&1; then echo "REFUSING: ${name} has a fold dir under ${res} (it started)"; exit 1; fi
  done < "${UD}/index.tsv"
  NU="K_${U#V_}"; [ ! -e "${SUP}/${U}" ] || { echo "REFUSING: ${SUP}/${U} exists"; exit 1; }
  if [ "${DRY}" = 1 ]; then echo "[dry] would: run table vulcan/${U} -> killarney/${NU}; mv ${UD} -> ${SUP}/${U}"; exit 0; fi
  cp "${RT}" "${RT}.bak_retarget_${U}_${stamp}"
  awk -F'\t' -v OFS='\t' -v u="${U}" -v nu="${NU}" 'NR>1 && $8=="vulcan" && $9==u {$8="killarney"; $9=nu} {print}' "${RT}" > "${RT}.new" && mv "${RT}.new" "${RT}"
  mkdir -p "${SUP}"; mv "${UD}" "${SUP}/${U}"
  echo "$(date '+%F %T') retargeted never-started Vulcan unit ${U} (chain cancelled) -> Killarney unit ${NU}" | tee -a "${R}/LAUNCH_INFO" "${R}/_launch/LAUNCH_LOG.txt"
  echo "next: bash scripts/cluster/ginipa/ginipa.sh record killarney ${NU}; Killarney: killarney.sh prep ${NU} && killarney.sh submit ${NU}"
  exit 0
fi

# ------------------------------------------------------------------------------------------------------------------ status
if [ "${MODE}" = status ]; then
  for idx in "${R}"/${1:-*}/*/index.tsv; do   # skips _superseded_retarget/ (and any other _* dir) unless named explicitly
    [ -f "${idx}" ] || continue; UD="$(dirname "${idx}")"; [ -n "${1:-}" ] || [[ "${UD#${R}/}" != _* ]] || continue; echo "-- ${UD#${R}/} $(cut -f3 "${UD}/JOBS.tsv" 2>/dev/null | tr '\n' ' ')"
    i=0; read -ra G <<<"$(cat "${UD}/GPU_MAP" 2>/dev/null || echo)"
    while IFS=$'\t' read -r cmd log name done_; do
      f="${name:4:1}"; [ -f "${cmd}" ] || { printf '   %-70s cmd missing (%s)\n' "${name#fold?_}_f${f}" "${cmd}"; i=$((i+1)); continue; }
      res="$(getv "$(clean_cmd "${cmd}")" nnUNet_results)"
      L="$(ls -t "${res}"/Dataset*/*/fold_${f}/training_log_*.txt 2>/dev/null | head -1 || true)"
      ts=$(grep -o "Epoch time: [0-9.]* s" "${L}" 2>/dev/null | awk '{print $3}' | tail -n +3 | sort -n || true)
      med=$(echo "${ts}" | awk 'NF{a[++n]=$1} END{if(n) print a[int((n+1)/2)]; else print "NA"}'); nep=$(echo "${ts}" | grep -c . || true)
      last=$(grep -oE "Epoch [0-9]+" "${L}" 2>/dev/null | tail -1 | grep -oE '[0-9]+$' || true)
      g="${G[$i]:-?}"; vr=$(cat "${UD}"/vram_*.csv 2>/dev/null | awk -F', ' -v g="${g}" '$2+0==g+0 {v=$3; gsub(/ MiB/,"",v); if (v+0>m+0) m=v+0} END{print m+0}' || true)
      printf '   %-70s GPU%s epoch=%-5s med_s/ep=%-7s (n=%s) peakVRAM=%sMiB final=%s\n' "${name#fold?_}_f${f}" "${g}" "${last:-NA}" "${med}" "${nep}" "${vr}" \
        "$([ -f "${done_}" ] && echo yes || echo no)"
      i=$((i+1))
    done < "${idx}"
  done
  exit 0
fi
echo "unknown mode ${MODE}"; exit 1
