#!/bin/bash
#SBATCH --job-name=lldmmrihcc_pack_predict_eval_extra
#SBATCH --account=aip-jcohen
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=/scratch/p/paulh/lld-mmri-hcc/_packruns/predict_extra_slurm_%j.out
# lld-mmri-hcc pack PREDICT + EVALUATE for the 6 NEW phases (ce-pre/ce-art/ce-pv/
# ce-del/inphase/outphase -- added 2026-09-01, see 00_utils/00_01_bidsify.py) against
# the 7 HEADLINE methods (6-method suite + DualVal val100 mirror). Companion to
# 05_23_tamia_pack_predict_evaluate_ladder_extra_phases.sh (the 4 non-headline
# ablation-rung checkpoints) -- split into two jobs to keep each one's wall-clock
# bounded and independently resumable/retryable.
#
# 157 HCC patients, 6 NEW items, 7 runs x 3 folds = 21 fold-tasks (each task predicts
# all 6 new items sequentially on its GPU). Mirrors 05_20_tamia_pack_predict_evaluate.sh
# EXACTLY (same TamIA whole-node lesson: predict AND evaluate run as direct bash/python
# calls inside ONE whole-node job, never nesting another sbatch submission) except:
# items=6 (not 2), and eval_all.csv gets REGENERATED per fold to merge old+new items
# (summarize_fold.py is passed ALL 8 items' present metrics files, not just the 6 new
# ones -- otherwise the merge would silently drop the original t2wi/dwi rows).
#
# Time budget: 05_20's 21-fold-task/2-item predict phase took ~12min. This is the same
# 21 fold-tasks but 3x the items (6 vs 2) -- ~36min extrapolated; 6:00:00 is a generous,
# not measured, margin on an untested scale-up (same caveat 05_20/05_21 both flag).
set -uo pipefail
HERE="/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/02_tasks/liver_hcc/lld-mmri-hcc/5_scripts_lld-mmri-hcc"
ATLAS_HERE="/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/02_tasks/liver_hcc/atlas-liver-hcc/5_scripts_atlas-liver-hcc"
PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
PACK_DIR="/scratch/p/paulh/lld-mmri-hcc/_packruns/predict_extra_pack_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PACK_DIR}"

echo "[pack] host=$(hostname) job=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

source "${HERE}/00_utils/env.sh"
source "${PROJECT_ROOT}/scripts/cluster/tamia_env_lld-mmri-hcc_atlascross.sh"
echo "[pack] nnUNet_raw=${nnUNet_raw}"
echo "[pack] PREDICTIONS_ROOT=${PREDICTIONS_ROOT}"

NEW_ITEMS="ce-pre ce-art ce-pv ce-del inphase outphase"
ALL_ITEMS="t2wi dwi ${NEW_ITEMS}"

TS=20260804_062605
T0=$(date +%s)
echo "[pack] ==== PHASE 1: record all 7 runs x 3 folds (=21), items=${NEW_ITEMS} ===="
cd "${HERE}/05_predict"
for job in \
    "05_02_predict_atlas_baseline.sh atlas-liver-hcc_t1w_baseline_${TS}" \
    "05_03_predict_atlas_auglab_default.sh atlas-liver-hcc_t1w_auglab_default_${TS}" \
    "05_04_predict_atlas_synthseg_noEM.sh atlas-liver-hcc_t1w_synthseg_noEM_${TS}" \
    "05_05_predict_atlas_synthseg_EM.sh atlas-liver-hcc_t1w_synthseg_EM_${TS}" \
    "05_06_predict_atlas_srcsm.sh atlas-liver-hcc_t1w_srcsm_${TS}" \
    "05_07_predict_atlas_auglabAug_v26_6_2_train050_val000.sh atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val000_${TS}" \
    "05_08_predict_atlas_auglabAug_v26_6_2_train050_val100.sh atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val100_${TS}" ; do
    wrapper="${job%% *}"; run_id="${job#* }"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${wrapper}" "${run_id}" all ${NEW_ITEMS}
done

N_RECORDED=$(grep -c . "${PACK_DIR}/index.tsv" 2>/dev/null || echo 0)
echo "[pack] recorded ${N_RECORDED} fold-predict commands -> ${PACK_DIR}/index.tsv"
[ "${N_RECORDED}" -gt 0 ] || { echo "[pack] ERROR: nothing recorded" >&2; exit 1; }

echo "[pack] ==== PHASE 2: bounded worker pool, 2 concurrent tasks PER GPU (8-way total) ===="
mapfile -t LINES < "${PACK_DIR}/index.tsv"
N=${#LINES[@]}
SLOTS=8
G=4
worker() {
    local slot="$1" idx="$1"
    local gpu=$(( slot % G ))
    while [ "${idx}" -lt "${N}" ]; do
        IFS=$'\t' read -r cmdfile log name donefile <<< "${LINES[$idx]}"
        local plog="${cmdfile%.sh}.log"
        local t0=$(date +%s)
        echo "[pack] slot${slot}(GPU${gpu}): launch '${name}' -> ${plog}"
        CUDA_VISIBLE_DEVICES=${gpu} bash "${cmdfile}" > "${plog}" 2>&1
        echo "[pack] slot${slot}(GPU${gpu}): '${name}' exit=$? ($(( $(date +%s) - t0 ))s)"
        idx=$(( idx + SLOTS ))
    done
}
pids=()
for slot in 0 1 2 3 4 5 6 7; do worker "${slot}" & pids+=($!); done
for p in "${pids[@]}"; do wait "${p}"; done
echo "[pack] predict phase done ($(( $(date +%s) - T0 ))s elapsed since job start)"

echo "[pack] ==== PHASE 3: verify EXACT file counts (157 per fold, per NEW item) ===="
verify_fail=0
for item in ${NEW_ITEMS}; do
    for d in "${PREDICTIONS_ROOT}"/atlas_liver_hcc_model/*/*/*/fold*/${item}; do
        [ -d "${d}" ] || continue
        n=$(find "${d}" -name '*.nii.gz' 2>/dev/null | wc -l)
        if [ "${n}" -ne 157 ]; then
            echo "[pack] BAD COUNT (${n}/157): ${d}" >&2
            verify_fail=1
        fi
    done
done
echo "[pack] predict verification done (verify_fail=${verify_fail})"

echo "[pack] ==== PHASE 4: evaluate (CPU-only, direct python calls, throttled) ===="
EVALUATE_PY="${ATLAS_HERE}/06_evaluate/06_00_evaluate.py"
LABEL_MAP='{"tumour": [2, 1]}'

run_eval() {   # $1=category $2=run_id $3=fold $4=item
    local category="$1" run_id="$2" fold="$3" item="$4"
    local pred_dir="${PREDICTIONS_ROOT}/atlas_liver_hcc_model/t1w/${category}/${run_id}/fold${fold}/${item}"
    local gt_dir="${nnUNet_raw}/labelsTs_${item}"
    local eval_dir="${METRICS_ROOT}/atlas_liver_hcc_model/t1w/${category}_${run_id}/fold${fold}"
    local n
    n=$(find "${pred_dir}" -name '*.nii.gz' 2>/dev/null | wc -l)
    if [ "${n}" -ne 157 ]; then
        echo "  ! ${run_id} fold${fold} ${item}: ${n}/157 predictions -- refusing to evaluate" >&2
        return 1
    fi
    mkdir -p "${eval_dir}"
    "${PROJECT_ROOT}/.venv/bin/python" "${EVALUATE_PY}" \
        --pred_dir "${pred_dir}" --gt_dir "${gt_dir}" --label_map "${LABEL_MAP}" \
        --name "${item}" --out_csv "${eval_dir}/${item}_metrics.csv" --workers 4 \
        > "${eval_dir}/${item}_eval.log" 2>&1
    local rows
    rows=$(( $(wc -l < "${eval_dir}/${item}_metrics.csv") - 1 ))
    echo "[pack] eval done: ${run_id} fold${fold} ${item} (${rows}/157 rows)"
    [ "${rows}" -eq 157 ] || { echo "[pack] WARNING: ${run_id} fold${fold} ${item} wrote only ${rows}/157 rows" >&2; return 1; }
}
export -f run_eval
export EVALUATE_PY LABEL_MAP nnUNet_raw PREDICTIONS_ROOT METRICS_ROOT PROJECT_ROOT

cat > "${PACK_DIR}/eval_tasks.txt" <<TASKS
nnUNet atlas-liver-hcc_t1w_baseline_${TS}
auglab atlas-liver-hcc_t1w_auglab_default_${TS}
auglab atlas-liver-hcc_t1w_synthseg_noEM_${TS}
auglab atlas-liver-hcc_t1w_synthseg_EM_${TS}
auglab atlas-liver-hcc_t1w_srcsm_${TS}
auglab atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val000_${TS}
auglab atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val100_${TS}
TASKS

: > "${PACK_DIR}/eval_joblist.txt"
while read -r category run_id; do
    [ -n "${category}" ] || continue
    for fold in 0 1 2; do
        for item in ${NEW_ITEMS}; do
            echo "${category} ${run_id} ${fold} ${item}" >> "${PACK_DIR}/eval_joblist.txt"
        done
    done
done < "${PACK_DIR}/eval_tasks.txt"

cat "${PACK_DIR}/eval_joblist.txt" | xargs -P 4 -L 1 bash -c 'run_eval "$@"' _
rc_eval=$?
echo "[pack] evaluate phase done rc=${rc_eval}"

echo "[pack] ==== PHASE 5: re-merge each fold's ALL 8 items into eval_all.csv (summarize_fold.py) ===="
# Unlike 05_20 (which only ever saw t2wi/dwi), this MUST check for all 8 possible
# item metrics.csv files, not just the 6 new ones -- the merge is a full rebuild of
# eval_all.csv from whatever *_metrics.csv files are present, so omitting the
# already-existing t2wi/dwi files here would silently drop them from the merged
# output even though their raw metrics.csv files are untouched on disk.
find "${METRICS_ROOT}" -mindepth 3 -maxdepth 3 -type d | while read -r run_dir; do
    run_name="$(basename "${run_dir}")"
    run_id="${run_name#nnUNet_}"; run_id="${run_id#auglab_}"
    for fold_dir in "${run_dir}"/fold*/; do
        [ -d "${fold_dir}" ] || continue
        fold="$(basename "${fold_dir}" | sed 's/fold//')"
        present=()
        for item in ${ALL_ITEMS}; do
            [ -f "${fold_dir}${item}_metrics.csv" ] && present+=("${item}")
        done
        if [ ${#present[@]} -gt 0 ]; then
            "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate/summarize_fold.py" \
                "${fold_dir}" "${run_id}" "${fold}" --group-col contrast --groups-word Contrasts \
                --title-suffix " | LLD-MMRI-HCC MRI | tumour only" --groups "${present[@]}" > /dev/null 2>&1
        fi
    done
done
echo "[pack] eval_all.csv merge done"

echo "[pack] ==== FINAL COMPREHENSIVE CHECK: all metrics files, exact row counts ===="
for item in ${ALL_ITEMS}; do
    find "${METRICS_ROOT}" -name "${item}_metrics.csv" 2>/dev/null | sort | while read -r f; do
        n=$(( $(wc -l < "$f") - 1 ))
        [ "${n}" -eq 157 ] || echo "[pack] BAD ROW COUNT (${n}/157): ${f}"
    done
done
total_new=$(find "${METRICS_ROOT}" \( -name "ce-pre_metrics.csv" -o -name "ce-art_metrics.csv" -o -name "ce-pv_metrics.csv" -o -name "ce-del_metrics.csv" -o -name "inphase_metrics.csv" -o -name "outphase_metrics.csv" \) 2>/dev/null | wc -l)
echo "[pack] total NEW metrics files: ${total_new} (expect 42 = 7 runs x 3 folds x 6 new items)"
total_all=$(find "${METRICS_ROOT}" -name "*_metrics.csv" 2>/dev/null | grep -vE "^${METRICS_ROOT}/atlas_liver_hcc_model/t1w/ablations" | wc -l)
echo "[pack] total ALL headline metrics files (incl. pre-existing t2wi/dwi): ${total_all} (expect 56 = 7 runs x 3 folds x 8 items)"
echo "[pack] ==== DONE (verify_fail=${verify_fail}, eval rc=${rc_eval}) ===="
