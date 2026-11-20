#!/bin/bash
# SLAug arm (slaug): run post_run.sh's predict + evaluate for the BraTS x4 and ON-Harmony x3 settings on TamIA instead of Vulcan
# (their Vulcan controllers sat 3-4 days on Priority and were cancelled before starting; user approval 2026-11-20), then land the
# outputs at the exact Vulcan paths post_run.sh writes and finish with post_run.sh's own audit. Same wrappers (record mode of the
# shared predict driver: checkpoint_best, --disable_tta, pin on PYTHONPATH, on-harmony RAS->native resample inside each predict
# command), same evaluators and ev bodies as post_run.sh; only the TamIA path overrides (scripts/cluster/tamia_env*.sh + brats
# METRICS_ROOT on scratch) are added. Only the 7 settings below (no external cohorts: post_run.sh reads none for them).
#   TamIA login : bash scripts/cluster/slaug/tamia_post.sh record          one pack dir per setting, cmd files grep-verified
#                 bash scripts/cluster/slaug/tamia_post.sh submit          2 whole h100:4 nodes (brats 12 rows, onh 9 rows) +
#                                                                          1 CPU eval job per setting (afterok its node)
#   (jobs)      : tamia_post.sh gpujob <brats|onh> / evaljob <setting>
#   Killarney   : bash scripts/cluster/slaug/tamia_post.sh push [setting...]   tar TamIA -> Vulcan through this node, md5 both ends
#   Vulcan      : bash scripts/cluster/slaug/tamia_post.sh finalize [setting...]   post_run.sh's audit -> POST_AUDIT.txt, POST.status
# Status: TamIA <root>/post_tamia/<setting>/{PREDICT,EVAL}.status; Vulcan <root>/post/<setting>/{LANDED,POST.status,POST_AUDIT.txt}.
set -uo pipefail
REPO=/project/aip-jcohen/paulh/mri_synthesis_project
TS_=20261114_073800; VROOT=/scratch/paulh/slaug_${TS_}; TROOT=/scratch/p/paulh/slaug_${TS_}
T=benchmark/02_tasks
B=${T}/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma; O=${T}/brain_healthy/on-harmony/5_scripts_on-harmony
# setting | group | scripts dir | env file | tamia env | training contrast | predict wrapper | eval dataset id
ROWS=(
"brats_t1n|brats|${B}|env.sh|tamia_env.sh|t1n|05_32_predict_t1n_auglab_slaug.sh|051"
"brats_t2w|brats|${B}|env_t2w.sh|tamia_env.sh|t2w|05_33_predict_t2w_auglab_slaug.sh|052"
"brats_t2f|brats|${B}|env_t2f.sh|tamia_env.sh|t2f|05_34_predict_t2f_auglab_slaug.sh|053"
"brats_t1c|brats|${B}|env_t1c.sh|tamia_env.sh|t1c|05_35_predict_t1c_auglab_slaug.sh|054"
"onh_T1w|onh|${O}|env.sh|tamia_env_onharmony.sh|T1w|05_30_predict_t1w_auglab_slaug.sh|"
"onh_T2w|onh|${O}|env_t2w.sh|tamia_env_onharmony.sh|T2w|05_31_predict_t2w_auglab_slaug.sh|"
"onh_dwi_ap|onh|${O}|env_dwi.sh|tamia_env_onharmony.sh|dwi_ap|05_32_predict_dwi_ap_auglab_slaug.sh|"
)
# expected predictions per fold (item:n) and evaluated cases per fold (group:n; on-harmony gre has 8 images, 7 GT)
EXP_PRED_brats="t1n:70 t1c:70 t2w:70 t2f:70"; EXP_EVAL_brats="t1n:70 t1c:70 t2w:70 t2f:70"
EXP_PRED_onh="T1w:8 T2w:8 bold:8 dwi_ap:8 epi_ap:4 gre_echo1_mag:8"; EXP_EVAL_onh="T1w:8 T2w:8 bold:8 dwi_ap:8 epi_ap:4 gre_echo1_mag:7"
# results bases relative to each cluster (TamIA = scratch layout of tamia_env*.sh; Vulcan = the repo tree post_run.sh writes)
tbase() { case "$1" in brats) echo "/scratch/p/paulh/brats2024-glioma/8_results" ;; onh) echo "/scratch/p/paulh/on-harmony/8_results" ;; esac; }
vbase() { case "$1" in brats) echo "${REPO}/${T}/brain_tumor/brats2024-glioma/8_results_brats2024-glioma" ;;
                       onh) echo "${REPO}/${T}/brain_healthy/on-harmony/8_results_on-harmony" ;; esac; }
model() { case "$1" in brats) echo brats2024_glioma_model ;; onh) echo on_harmony_model ;; esac; }
row() { local r; for r in "${ROWS[@]}"; do [ "${r%%|*}" = "$1" ] && { echo "$r"; return 0; }; done; echo "unknown setting $1" >&2; return 1; }
rid() { local s="$1" ds; case "$s" in brats_*) ds=brats2024-glioma ;; onh_*) ds=on-harmony ;; esac; echo "${ds}_${s#*_}_auglab_slaug_${TS_}"; }
settings_of() { local r; for r in "${ROWS[@]}"; do IFS='|' read -r s g _ <<<"$r"; [ -z "${1:-}" ] || [ "$g" = "$1" ] && echo "$s"; done; }
log() { echo "[$(date '+%F %T')] $*"; }
# envsetup <dir> <envfile> <tamia env> <tc> [group]: dataset env, then the TamIA overrides (same order as tamia_rung5_predict_job.sh)
envsetup() { export TRAINING_CONTRAST="$4"; source "$1/00_utils/$2"; source "scripts/cluster/$3"; export TRAINING_CONTRAST="$4"
  [ "${5:-}" != brats ] || export METRICS_ROOT="/scratch/p/paulh/brats2024-glioma/8_results/02_metrics"; }
count_preds() {   # <pred base> <item:n list> -> prints mismatches, returns 1 if any
  local bad=0 k it n; for k in 0 1 2; do for it in $2; do
    n=$(ls "$1/fold${k}/${it%%:*}" 2>/dev/null | grep -c '\.nii\.gz$'); [ "$n" = "${it##*:}" ] || { echo "fold${k} ${it%%:*}: ${n}/${it##*:}"; bad=1; }
  done; done; return $bad; }

MODE="${1:?usage: tamia_post.sh record|submit|gpujob <g>|evaljob <s>|push [s..]|finalize [s..]}"; shift
cd "${REPO}"
case "${MODE}" in
record)   # TamIA login node
  export SCRATCH=/scratch/p/paulh; PT="${TROOT}/post_tamia"; mkdir -p "${PT}"
  log "record at HEAD $(git rev-parse --short HEAD) ($(git status --porcelain -- scripts benchmark/00_commun_scripts ${B} ${O} | grep -vc '^??') tracked edits)"
  for r in "${ROWS[@]}"; do IFS='|' read -r s g dir envf tenv tc wr ds <<<"$r"; R_="$(rid "$s")"; PD="${PT}/${s}/pack"; mkdir -p "${PT}/${s}"
    [ ! -e "${PD}/index.tsv" ] || { log "REFUSING: ${PD}/index.tsv exists (move the pack dir aside to re-record)"; exit 1; }
    [ "$(awk -F'\t' -v s="$s" 'NR>1 && $1==s {print $16}' "${TROOT}/run_table.tsv" | sort -u)" = "${R_}" ] || { log "REFUSING: ${s} run id != ${R_} in the run table"; exit 1; }
    ( envsetup "${dir}" "${envf}" "${tenv}" "${tc}" "${g}"; P="${PREDICTIONS_ROOT}/${MODEL_TYPE}/${tc}/auglab/${R_}"
      [ "$P" = "$(tbase "$g")/01_predictions/$(model "$g")/${tc}/auglab/${R_}" ] || { echo "unexpected pred base $P"; exit 1; }
      [ -z "$(ls "$P"/fold[012]/*/*.nii.gz 2>/dev/null | head -1)" ] || { echo "predictions already exist under $P"; exit 1; }
      RUN_JOB_PACK_DIR="${PD}" bash "${dir}/05_predict/${wr}" "${R_}" ) > "${PT}/${s}/record.log" 2>&1 || { log "ERROR recording ${s} (see ${PT}/${s}/record.log)"; exit 1; }
    n=$(grep -c . "${PD}/index.tsv"); [ "$n" = 3 ] || { log "ERROR ${s}: ${n} rows recorded, expected 3"; exit 1; }
    case "$g" in brats) nitem=4; dsid="${ds}" ;; onh) nitem=6; dsid="0$(case "$tc" in T1w) echo 31 ;; T2w) echo 32 ;; dwi_ap) echo 33 ;; esac)" ;; esac
    while IFS=$'\t' read -r cf _l nm _d; do e=""
      grep -q -- "${R_}/fold" "$cf" || e="${e} run-id"
      [ "$(grep -o -- "-d ${dsid} -c 3d_fullres" "$cf" | wc -l)" = "${nitem}" ] || e="${e} -d${dsid}x${nitem}"
      [ "$(grep -o -- "--disable_tta -chk checkpoint_best.pth" "$cf" | wc -l)" = "${nitem}" ] || e="${e} best/no-tta"
      grep -q "slaug_pin/2a8cfde30fc9/AugLab" "$cf" || e="${e} pin"
      for io in -i -o; do   # recorded inside a $'...' string: quotes appear as \'
        [ "$(tr -d '\\' < "$cf" | grep -o -- "${io} '/scratch/p/paulh/" | wc -l)" = "${nitem}" ] || e="${e} scratch${io}"; done
      grep -q "CUDA_VISIBLE_DEVICES" "$cf" && e="${e} inner-CUDA"
      [ "$g" != onh ] || [ "$(grep -o "05_02_resample_predictions_to_native.py" "$cf" | wc -l)" = 6 ] || e="${e} resample"
      [ -z "$e" ] || { log "ERROR ${s} ${cf}: check failed:${e}"; exit 1; }
    done < "${PD}/index.tsv"
    log "${s}: 3 cmd files recorded + verified (${R_}, -d ${dsid}, best, no TTA, pin; onh: resample x6)"
  done ;;
submit)   # TamIA login node
  PT="${TROOT}/post_tamia"; J="${PT}/JOBS.tsv"; [ ! -e "$J" ] || { log "REFUSING: $J exists"; exit 1; }
  for g in brats onh; do for s in $(settings_of "$g"); do [ -f "${PT}/${s}/pack/index.tsv" ] || { log "not recorded: ${s}"; exit 1; }; done; done
  for g in brats onh; do
    jid=$(sbatch --parsable --account=aip-jcohen --nodes=1 --gpus-per-node=h100:4 --cpus-per-task=48 --mem=0 --time="${GPU_TIME:-03:00:00}" \
          --exclude=tg10807 --job-name="slpost_pred_${g}" --output="${PT}/pred_${g}_%j.log" \
          --wrap "bash ${REPO}/scripts/cluster/slaug/tamia_post.sh gpujob ${g}") || { log "sbatch gpu ${g} failed"; exit 1; }
    printf 'pred_%s\t%s\n' "$g" "$jid" >> "$J"; log "predict node ${g}: ${jid}"; sleep 3
    for s in $(settings_of "$g"); do
      args=(--parsable --account=aip-jcohen --cpus-per-task=16 --mem=96G --time="${EVAL_TIME:-05:00:00}" --dependency="afterok:${jid}"
            --job-name="slpost_eval_${s}" --output="${PT}/${s}/eval_%j.log" --wrap "bash ${REPO}/scripts/cluster/slaug/tamia_post.sh evaljob ${s}")
      e=$(sbatch "${args[@]}" 2>&1) || e=$(sbatch --partition=cpubase_bycore_b2 "${args[@]}" 2>&1) || { log "sbatch eval ${s} failed: $e"; exit 1; }
      printf 'eval_%s\t%s\n' "$s" "$e" >> "$J"; log "eval ${s}: ${e} (afterok ${jid})"; sleep 3
    done
  done ;;
gpujob)   # whole h100:4 node: every recorded predict command of the group, pinned round-robin, then file-count gate
  g="${1:?group}"; export SCRATCH=/scratch/p/paulh; PT="${TROOT}/post_tamia"
  export CUDA_CACHE_PATH="${SLURM_TMPDIR:-/tmp}/cuda_compute_cache_${SLURM_JOB_ID:-x}"   # TamIA 2026-11-07 hang on $HOME/.nv
  source scripts/cluster/slaug/pin_env.sh   # refuses unless the TamIA pin is at the pinned commit
  log "gpujob ${g} host=$(hostname) job=${SLURM_JOB_ID:-?} HEAD=$(git rev-parse --short HEAD) pin=${SLAUG_PIN}"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
  declare -a PIDS NAMES; i=0
  for s in $(settings_of "$g"); do
    P="$(tbase "$g")/01_predictions/$(model "$g")/$(row "$s" | cut -d'|' -f6)/auglab/$(rid "$s")"
    [ -z "$(ls "$P"/fold[012]/*/*.nii.gz 2>/dev/null | head -1)" ] || { log "REFUSING ${s}: predictions already under $P"; exit 1; }
    while IFS=$'\t' read -r cf _l nm _d; do gpu=$(( i % 4 ))
      log "launch ${nm} (${s}) on GPU ${gpu}"; CUDA_VISIBLE_DEVICES=${gpu} bash "$cf" > "${cf%.sh}.log" 2>&1 &
      PIDS[$i]=$!; NAMES[$i]="${s} ${nm}"; i=$((i + 1)); sleep 5
    done < "${PT}/${s}/pack/index.tsv"
  done
  for j in "${!PIDS[@]}"; do wait "${PIDS[$j]}" && log "exit 0  ${NAMES[$j]}" || log "exit !0 ${NAMES[$j]}"; done
  rc=0; ev="EXP_PRED_${g}"
  for s in $(settings_of "$g"); do
    P="$(tbase "$g")/01_predictions/$(model "$g")/$(row "$s" | cut -d'|' -f6)/auglab/$(rid "$s")"
    if m="$(count_preds "$P" "${!ev}")"; then echo "[$(date '+%F %T')] OK predictions ${!ev} x 3 folds (job ${SLURM_JOB_ID:-?})" > "${PT}/${s}/PREDICT.status"; log "${s}: predictions OK"
    else printf '[%s] FAILED predictions (job %s): %s\n' "$(date '+%F %T')" "${SLURM_JOB_ID:-?}" "$(echo $m)" > "${PT}/${s}/PREDICT.status"; log "${s}: FAILED $(echo $m)"; rc=1; fi
  done
  exit $rc ;;
evaljob)  # CPU job: post_run.sh's ev body for one setting (+ TamIA overrides), then case-count check of every eval_all.csv
  s="${1:?setting}"; IFS='|' read -r _ g dir envf tenv tc wr ds <<<"$(row "$s")"; R_="$(rid "$s")"; PT="${TROOT}/post_tamia"; ST="${PT}/${s}/EVAL.status"
  export SCRATCH=/scratch/p/paulh RUN_JOB_INLINE=1; unset RUN_JOB_PACK_DIR RUN_JOB_DEPENDENCY
  grep -q '] OK ' "${PT}/${s}/PREDICT.status" 2>/dev/null || { log "REFUSING: predictions of ${s} not OK" | tee "$ST"; exit 1; }
  M="$(tbase "$g")/02_metrics/$(model "$g")/${tc}/auglab_${R_}"
  [ ! -e "$M" ] || { log "REFUSING: $M exists (stale-skip risk: move it aside first)" | tee "$ST"; exit 1; }
  log "evaljob ${s} job=${SLURM_JOB_ID:-?} HEAD=$(git rev-parse --short HEAD) -> ${M}" | tee "$ST"
  ( envsetup "${dir}" "${envf}" "${tenv}" "${tc}" "${g}"
    echo "nnUNet_raw=${nnUNet_raw} PREDICTIONS_ROOT=${PREDICTIONS_ROOT} METRICS_ROOT=${METRICS_ROOT}"
    case "$g" in
      brats) export TRAINING_CONTRAST="${tc}" DATASET_ID="${ds}" EVAL_INLINE=1 CATEGORY=auglab METRICS_SUBDIR=
             for F in 0 1 2; do bash "${dir}/06_evaluate/06_01_evaluate_run.sh" "${R_}" "${F}" || exit 1; done ;;
      onh)   export METRICS_SUBDIR=; for F in 0 1 2; do bash "${dir}/06_evaluate/06_01_evaluate_testset.sh" "${R_}" "${F}" || exit 1; done ;;
    esac ) || { log "FAILED: evaluator exit != 0" | tee -a "$ST"; exit 1; }
  ev="EXP_EVAL_${g}"
  .venv/bin/python - "$M" ${!ev} >> "$ST" 2>&1 <<'PY' || { log "FAILED: case counts" | tee -a "$ST"; exit 1; }
import csv, sys
from pathlib import Path
m = Path(sys.argv[1]); exp = {x.split(":")[0]: int(x.split(":")[1]) for x in sys.argv[2:]}; bad = []; rows = 0
for k in range(3):
    f = m / f"fold{k}" / "eval_all.csv"
    if not f.exists():
        bad.append(f"fold{k}: no eval_all.csv"); continue
    cases = {}
    for r in csv.DictReader(f.open()):
        cases.setdefault(r["group"], set()).add(r["case"]); rows += 1
    for c, n in exp.items():
        if len(cases.get(c, ())) != n:
            bad.append(f"fold{k}/{c}: {len(cases.get(c, ()))} cases, expected {n}")
    if set(cases) - set(exp):
        bad.append(f"fold{k}: unexpected groups {sorted(set(cases) - set(exp))}")
print("verify:", f"{rows} rows over 3 folds;", "OK" if not bad else "; ".join(bad)); sys.exit(1 if bad else 0)
PY
  log "OK" | tee -a "$ST" ;;
push)     # Killarney login node (holds both ssh masters): TamIA -> Vulcan, md5 of every file both ends
  SSHT=(ssh -n -o BatchMode=yes tamia.alliancecan.ca); SSHV=(ssh -o BatchMode=yes vulcan.alliancecan.ca); SSHVN=(ssh -n -o BatchMode=yes vulcan.alliancecan.ca)
  for s in ${@:-$(settings_of)}; do IFS='|' read -r _ g _ _ _ tc _ _ <<<"$(row "$s")"; R_="$(rid "$s")"
    TP="$(tbase "$g")/01_predictions/$(model "$g")/${tc}/auglab/${R_}"; VP="$(vbase "$g")/01_predictions/$(model "$g")/${tc}/auglab/${R_}"
    TM="$(tbase "$g")/02_metrics/$(model "$g")/${tc}"; VM="$(vbase "$g")/02_metrics/$(model "$g")/${tc}"; MD="auglab_${R_}"
    "${SSHT[@]}" "grep -q '] OK$' ${TROOT}/post_tamia/${s}/EVAL.status" || { log "${s}: TamIA eval not OK -- skipped"; continue; }
    "${SSHVN[@]}" "test ! -e ${VROOT}/post/${s}/LANDED && test -d ${VP} && test ! -e ${VM}/${MD} && test -z \"\$(ls ${VP}/fold[012]/*/*.nii.gz 2>/dev/null | head -1)\"" \
      || { log "${s}: Vulcan target not fresh (LANDED, predictions or ${MD} exist) -- skipped"; continue; }
    log "${s}: predictions + metrics TamIA -> Vulcan"
    "${SSHT[@]}" "cd ${TP} && tar cf - fold0 fold1 fold2" | "${SSHV[@]}" "cd ${VP} && tar xf -" || { log "${s}: prediction copy FAILED"; continue; }
    "${SSHT[@]}" "cd ${TM} && tar cf - ${MD}" | "${SSHV[@]}" "cd ${VM} && tar xf -" || { log "${s}: metrics copy FAILED"; continue; }
    "${SSHT[@]}" "cd ${TROOT}/post_tamia/${s} && tar cf - --exclude='*.nii.gz' ." | "${SSHV[@]}" "mkdir -p ${VROOT}/post/${s}/post_logs/tamia && cd ${VROOT}/post/${s}/post_logs/tamia && tar xf -" || log "${s}: log copy failed (non-fatal)"
    PQ="find fold0 fold1 fold2 -type f | LC_ALL=C sort | xargs md5sum"; CQ="find ${MD} -type f -name '*.csv' | LC_ALL=C sort | xargs md5sum"
    a="$("${SSHT[@]}" "cd ${TP} && nice -n 10 bash -c \"${PQ}\"; cd ${TM} && bash -c \"${CQ}\"")"
    b="$("${SSHVN[@]}" "cd ${VP} && nice -n 10 bash -c \"${PQ}\"; cd ${VM} && bash -c \"${CQ}\"")"
    na=$(echo "$a" | grep -c .); nb=$(echo "$b" | grep -c .)
    if [ "$a" = "$b" ] && [ "$na" -gt 0 ]; then
      printf '%s TamIA -> Vulcan, %s files md5-identical (predictions fold0-2 + metrics csv)\n' "$(date '+%F %T')" "$na" | "${SSHV[@]}" "cat > ${VROOT}/post/${s}/LANDED"
      log "${s}: LANDED (${na} files md5-identical)"
    else log "${s}: md5 MISMATCH (TamIA ${na} files, Vulcan ${nb}) -- not marked LANDED"; fi
  done ;;
finalize) # Vulcan login node: post_run.sh's audit, POST_AUDIT.txt + POST.status in its log format
  for s in ${@:-$(settings_of)}; do SD="${VROOT}/post/${s}"; ST="${SD}/POST.status"; R_="$(rid "$s")"
    [ -f "${SD}/LANDED" ] || { log "${s}: not LANDED -- skipped"; continue; }
    case "$s" in brats_*) ONLY=brats2024-glioma ;; onh_*) ONLY=on-harmony ;; esac
    { log "START ${s} host=$(hostname) job=none HEAD=$(git rev-parse --short HEAD) -- predict + evaluate ran on TamIA (scripts/cluster/slaug/tamia_post.sh: $(cut -d' ' -f3- "${SD}/LANDED"))"
      log "run ${R_}"
      log "audit vs the headline PA+Auglab run (sources: ${ONLY})"; } >> "$ST"
    FAIL=0; .venv/bin/python scripts/cluster/flatauglab/audit_vs_headline.py "${s}" --only "${ONLY}" --run-table "${VROOT}/run_table.tsv" > "${SD}/POST_AUDIT.txt" 2>&1 || FAIL=1
    grep -v '^ok ' "${SD}/POST_AUDIT.txt" >> "$ST"
    if [ "$FAIL" = 0 ]; then log "DONE ${s} (predict + evaluate on TamIA, audit on Vulcan)" >> "$ST"; else log "FAILED ${s} (audit, see POST_AUDIT.txt)" >> "$ST"; fi
    log "${s}: $(tail -1 "$ST")"
  done ;;
*) echo "unknown mode ${MODE}" >&2; exit 2 ;;
esac
