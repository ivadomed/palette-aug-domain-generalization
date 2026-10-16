#!/bin/bash
#SBATCH --job-name=lldmmrihcc_pack_predict_eval_ladder_extra
#SBATCH --account=aip-jcohen
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=/scratch/p/paulh/lld-mmri-hcc/_packruns/predict_ladder_extra_slurm_%j.out
# lld-mmri-hcc pack PREDICT + EVALUATE for the 6 NEW phases (ce-pre/ce-art/ce-pv/
# ce-del/inphase/outphase -- see 00_utils/00_01_bidsify.py) against the atlas-liver-hcc
# causal-ablation ladder's 4 non-headline rungs (kmeans/label_remap/voronoi/
# v26_6_2-alone). Companion to 05_22 (the 7 headline methods) -- split into two jobs
# to keep each one's wall-clock bounded and independently resumable/retryable.
# Mirrors 05_21_tamia_pack_predict_evaluate_ladder.sh exactly (own metrics land in
# ablations/, not the flat headline layout -- see CLAUDE.md's 02_metrics subdir
# convention) except items=6 new phases (not 2), writing INTO the SAME ablations/
# run dirs 05_21 already created, and the eval_all.csv re-merge covers all 8 items.
#
# Time budget: 05_21's 12-fold-task/2-item predict phase took a fraction of 05_20's
# ~12min (fewer runs: 4 vs 7). This is the same 12 fold-tasks but 3x the items (6 vs
# 2) -- roughly proportionally scaled; 4:00:00 is a generous, not measured, margin.
set -uo pipefail
HERE="/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/02_tasks/liver_hcc/lld-mmri-hcc/5_scripts_lld-mmri-hcc"
ATLAS_HERE="/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/02_tasks/liver_hcc/atlas-liver-hcc/5_scripts_atlas-liver-hcc"
PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
PACK_DIR="/scratch/p/paulh/lld-mmri-hcc/_packruns/predict_ladder_extra_pack_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PACK_DIR}"

echo "[pack] host=$(hostname) job=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

source "${HERE}/00_utils/env.sh"
source "${PROJECT_ROOT}/scripts/cluster/tamia_env_lld-mmri-hcc_atlascross.sh"
echo "[pack] nnUNet_raw=${nnUNet_raw}"
echo "[pack] PREDICTIONS_ROOT=${PREDICTIONS_ROOT}"
echo "[pack] ATLAS_NNUNET_RAW=${ATLAS_NNUNET_RAW}"

NEW_ITEMS="ce-pre ce-art ce-pv ce-del inphase outphase"
ALL_ITEMS="t2wi dwi ${NEW_ITEMS}"

TS=20260829_165445
T0=$(date +%s)
echo "[pack] ==== PHASE 1: record all 4 ladder runs x 3 folds (=12), items=${NEW_ITEMS} ===="
cd "${HERE}/05_predict"
for job in \
    "05_09_predict_atlas_baseline_kmeans.sh atlas-liver-hcc_t1w_baseline_kmeans_${TS}" \
    "05_10_predict_atlas_baseline_kmeans_label_remap.sh atlas-liver-hcc_t1w_baseline_kmeans_label_remap_${TS}" \
    "05_11_predict_atlas_baseline_kmeans_label_remap_voronoi.sh atlas-liver-hcc_t1w_baseline_kmeans_label_remap_voronoi_${TS}" \
    "05_12_predict_atlas_v26_6_2_train050_val100.sh atlas-liver-hcc_t1w_v26_6_2_train050_val100_${TS}" ; do
    wrapper="${job%% *}"; run_id="${job#* }"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${wrapper}" "${run_id}" all ${NEW_ITEMS}
done

N_RECORDED=$(grep -c . "${PACK_DIR}/index.tsv" 2>/dev/null || echo 0)
echo "[pack] recorded ${N_RECORDED} fold-predict commands -> ${PACK_DIR}/index.tsv"
[ "${N_RECORDED}" -gt 0 ] || { echo "[pack] ERROR: nothing recorded" >&2; exit 1; }

echo "[pack] ==== PHASE 2: bounded worker pool, 3 concurrent tasks PER GPU (12-way total) ===="
mapfile -t LINES < "${PACK_DIR}/index.tsv"
N=${#LINES[@]}
SLOTS=12
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
for slot in $(seq 0 $((SLOTS - 1))); do worker "${slot}" & pids+=($!); done
for p in "${pids[@]}"; do wait "${p}"; done
echo "[pack] predict phase done ($(( $(date +%s) - T0 ))s elapsed since job start)"

echo "[pack] ==== PHASE 3: verify EXACT file counts (157 per fold, per NEW item) ===="
verify_fail=0
for item in ${NEW_ITEMS}; do
    for d in "${PREDICTIONS_ROOT}"/atlas_liver_hcc_model/*/*/*_${TS}/fold*/${item}; do
        [ -d "${d}" ] || continue
        n=$(find "${d}" -name '*.nii.gz' 2>/dev/null | wc -l)
        if [ "${n}" -ne 157 ]; then
            echo "[pack] BAD COUNT (${n}/157): ${d}" >&2
            verify_fail=1
        fi
    done
done
echo "[pack] predict verification done (verify_fail=${verify_fail})"

echo "[pack] ==== PHASE 4: evaluate (CPU-only, direct python calls, throttled) — into ablations/ ===="
EVALUATE_PY="${ATLAS_HERE}/06_evaluate/06_00_evaluate.py"
LABEL_MAP='{"tumour": [2, 1]}'

run_eval() {   # $1=category $2=run_id $3=fold $4=item
    local category="$1" run_id="$2" fold="$3" item="$4"
    local pred_dir="${PREDICTIONS_ROOT}/atlas_liver_hcc_model/t1w/${category}/${run_id}/fold${fold}/${item}"
    local gt_dir="${nnUNet_raw}/labelsTs_${item}"
    local eval_dir="${METRICS_ROOT}/atlas_liver_hcc_model/t1w/ablations/${category}_${run_id}/fold${fold}"
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
auglab atlas-liver-hcc_t1w_baseline_kmeans_${TS}
auglab atlas-liver-hcc_t1w_baseline_kmeans_label_remap_${TS}
auglab atlas-liver-hcc_t1w_baseline_kmeans_label_remap_voronoi_${TS}
nnUNet atlas-liver-hcc_t1w_v26_6_2_train050_val100_${TS}
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
find "${METRICS_ROOT}/atlas_liver_hcc_model/t1w/ablations" -mindepth 1 -maxdepth 1 -type d | while read -r run_dir; do
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
                --title-suffix " | LLD-MMRI-HCC MRI | tumour only (causal-ablation ladder rung)" --groups "${present[@]}" > /dev/null 2>&1
        fi
    done
done
echo "[pack] eval_all.csv merge done"

echo "[pack] ==== FINAL COMPREHENSIVE CHECK: all metrics files, exact row counts ===="
for item in ${ALL_ITEMS}; do
    find "${METRICS_ROOT}/atlas_liver_hcc_model/t1w/ablations" -name "${item}_metrics.csv" 2>/dev/null | sort | while read -r f; do
        n=$(( $(wc -l < "$f") - 1 ))
        [ "${n}" -eq 157 ] || echo "[pack] BAD ROW COUNT (${n}/157): ${f}"
    done
done
total_new=$(find "${METRICS_ROOT}/atlas_liver_hcc_model/t1w/ablations" \( -name "ce-pre_metrics.csv" -o -name "ce-art_metrics.csv" -o -name "ce-pv_metrics.csv" -o -name "ce-del_metrics.csv" -o -name "inphase_metrics.csv" -o -name "outphase_metrics.csv" \) 2>/dev/null | wc -l)
echo "[pack] total NEW metrics files: ${total_new} (expect 24 = 4 runs x 3 folds x 6 new items)"
total_all=$(find "${METRICS_ROOT}/atlas_liver_hcc_model/t1w/ablations" -name "*_metrics.csv" 2>/dev/null | wc -l)
echo "[pack] total ALL ladder metrics files (incl. pre-existing t2wi/dwi): ${total_all} (expect 32 = 4 runs x 3 folds x 8 items)"
echo "[pack] ==== DONE (verify_fail=${verify_fail}, eval rc=${rc_eval}) ===="
