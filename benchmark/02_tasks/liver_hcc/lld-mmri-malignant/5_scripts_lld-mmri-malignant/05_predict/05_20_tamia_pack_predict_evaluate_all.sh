#!/bin/bash
#SBATCH --job-name=lldmal_pack_predict_eval_all
#SBATCH --account=aip-jcohen
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --output=/scratch/p/paulh/lld-mmri-malignant/_packruns/predict_eval_all_slurm_%j.out
# lld-mmri-malignant pack PREDICT + EVALUATE — the FULL grid in one job:
#   11 atlas-liver-hcc runs (7 headline + 4 causal-ablation ladder rungs)
#   x 3 folds x 8 items x 109 patients.
# Unlike lld-mmri-hcc (which split headline/ladder and old/new phases across four
# jobs, 05_20..05_23), this dataset is onboarded in one shot, so one job covers
# everything — TamIA allocates whole nodes, so a job that only exercises part of the
# node wastes the rest for its whole wall-clock (see CLAUDE.md's TamIA section).
#
# Metrics routing differs per run and MUST NOT be flattened: the 7 headline runs go
# to the flat 02_metrics/<model>/t1w/<category>_<run_id>/ layout, the 4 ladder rungs
# go to t1w/ablations/ (CLAUDE.md's 02_metrics subdir convention). The eval task list
# below carries the subdir per run for exactly this reason.
#
# Every verification here checks EXACT ROW COUNTS, not file existence: a previous
# TamIA pack job silently truncated 14/48 fold-metrics files and an existence check
# missed it entirely (see project_msd_spleen_onboarding memory).
set -uo pipefail
HERE="/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/02_tasks/liver_hcc/lld-mmri-malignant/5_scripts_lld-mmri-malignant"
ATLAS_HERE="/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/02_tasks/liver_hcc/atlas-liver-hcc/5_scripts_atlas-liver-hcc"
PROJECT_ROOT="/project/aip-jcohen/paulh/mri_synthesis_project"
PACK_DIR="/scratch/p/paulh/lld-mmri-malignant/_packruns/pack_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${PACK_DIR}"

N_CASES=109

echo "[pack] host=$(hostname) job=${SLURM_JOB_ID:-?}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

source "${HERE}/00_utils/env.sh"
source "${PROJECT_ROOT}/scripts/cluster/tamia_env_lld-mmri-malignant_atlascross.sh"
echo "[pack] nnUNet_raw=${nnUNet_raw}"
echo "[pack] PREDICTIONS_ROOT=${PREDICTIONS_ROOT}"
echo "[pack] METRICS_ROOT=${METRICS_ROOT}"

ITEMS="t2wi dwi ce-pre ce-art ce-pv ce-del inphase outphase"
TS_HEAD=20260804_062605
TS_LAD=20260829_165445
T0=$(date +%s)

echo "[pack] ==== PHASE 1: record 11 runs x 3 folds (=33 fold-tasks), items=${ITEMS} ===="
cd "${HERE}/05_predict"
for job in \
    "05_02_predict_atlas_baseline.sh atlas-liver-hcc_t1w_baseline_${TS_HEAD}" \
    "05_03_predict_atlas_auglab_default.sh atlas-liver-hcc_t1w_auglab_default_${TS_HEAD}" \
    "05_04_predict_atlas_synthseg_noEM.sh atlas-liver-hcc_t1w_synthseg_noEM_${TS_HEAD}" \
    "05_05_predict_atlas_synthseg_EM.sh atlas-liver-hcc_t1w_synthseg_EM_${TS_HEAD}" \
    "05_06_predict_atlas_srcsm.sh atlas-liver-hcc_t1w_srcsm_${TS_HEAD}" \
    "05_07_predict_atlas_auglabAug_v26_6_2_train050_val000.sh atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val000_${TS_HEAD}" \
    "05_08_predict_atlas_auglabAug_v26_6_2_train050_val100.sh atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val100_${TS_HEAD}" \
    "05_09_predict_atlas_baseline_kmeans.sh atlas-liver-hcc_t1w_baseline_kmeans_${TS_LAD}" \
    "05_10_predict_atlas_baseline_kmeans_label_remap.sh atlas-liver-hcc_t1w_baseline_kmeans_label_remap_${TS_LAD}" \
    "05_11_predict_atlas_baseline_kmeans_label_remap_voronoi.sh atlas-liver-hcc_t1w_baseline_kmeans_label_remap_voronoi_${TS_LAD}" \
    "05_12_predict_atlas_v26_6_2_train050_val100.sh atlas-liver-hcc_t1w_v26_6_2_train050_val100_${TS_LAD}" ; do
    wrapper="${job%% *}"; run_id="${job#* }"
    RUN_JOB_PACK_DIR="${PACK_DIR}" bash "${wrapper}" "${run_id}" all ${ITEMS}
done

N_RECORDED=$(grep -c . "${PACK_DIR}/index.tsv" 2>/dev/null || echo 0)
echo "[pack] recorded ${N_RECORDED} fold-predict commands (expect 33) -> ${PACK_DIR}/index.tsv"
[ "${N_RECORDED}" -gt 0 ] || { echo "[pack] ERROR: nothing recorded" >&2; exit 1; }

echo "[pack] ==== PHASE 2: bounded worker pool, 3 concurrent tasks per GPU (12-way) ===="
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
echo "[pack] predict phase done ($(( $(date +%s) - T0 ))s elapsed)"

echo "[pack] ==== PHASE 3: verify EXACT prediction counts (${N_CASES} per fold per item) ===="
verify_fail=0
for item in ${ITEMS}; do
    for d in "${PREDICTIONS_ROOT}"/atlas_liver_hcc_model/*/*/*/fold*/${item}; do
        [ -d "${d}" ] || continue
        n=$(find "${d}" -name '*.nii.gz' 2>/dev/null | wc -l)
        if [ "${n}" -ne "${N_CASES}" ]; then
            echo "[pack] BAD COUNT (${n}/${N_CASES}): ${d}" >&2
            verify_fail=1
        fi
    done
done
echo "[pack] predict verification done (verify_fail=${verify_fail})"

echo "[pack] ==== PHASE 4: evaluate (CPU, direct python, throttled) ===="
EVALUATE_PY="${ATLAS_HERE}/06_evaluate/06_00_evaluate.py"
LABEL_MAP='{"tumour": [2, 1]}'

run_eval() {   # $1=category $2=run_id $3=subdir("-"=flat) $4=fold $5=item
    local category="$1" run_id="$2" subdir="$3" fold="$4" item="$5"
    local pred_dir="${PREDICTIONS_ROOT}/atlas_liver_hcc_model/t1w/${category}/${run_id}/fold${fold}/${item}"
    local gt_dir="${nnUNet_raw}/labelsTs_${item}"
    local base="${METRICS_ROOT}/atlas_liver_hcc_model/t1w"
    [ "${subdir}" = "-" ] || base="${base}/${subdir}"
    local eval_dir="${base}/${category}_${run_id}/fold${fold}"
    local n
    n=$(find "${pred_dir}" -name '*.nii.gz' 2>/dev/null | wc -l)
    if [ "${n}" -ne "${N_CASES}" ]; then
        echo "  ! ${run_id} fold${fold} ${item}: ${n}/${N_CASES} predictions -- refusing to evaluate" >&2
        return 1
    fi
    mkdir -p "${eval_dir}"
    "${PROJECT_ROOT}/.venv/bin/python" "${EVALUATE_PY}" \
        --pred_dir "${pred_dir}" --gt_dir "${gt_dir}" --label_map "${LABEL_MAP}" \
        --name "${item}" --out_csv "${eval_dir}/${item}_metrics.csv" --workers 4 \
        > "${eval_dir}/${item}_eval.log" 2>&1
    local rows
    rows=$(( $(wc -l < "${eval_dir}/${item}_metrics.csv") - 1 ))
    echo "[pack] eval done: ${run_id} fold${fold} ${item} (${rows}/${N_CASES} rows)"
    [ "${rows}" -eq "${N_CASES}" ] || { echo "[pack] WARNING: ${run_id} fold${fold} ${item} wrote ${rows}/${N_CASES} rows" >&2; return 1; }
}
export -f run_eval
export EVALUATE_PY LABEL_MAP nnUNet_raw PREDICTIONS_ROOT METRICS_ROOT PROJECT_ROOT N_CASES

# category  run_id  metrics_subdir ("-" = flat headline layout, "ablations" = ladder)
cat > "${PACK_DIR}/eval_tasks.txt" <<TASKS
nnUNet atlas-liver-hcc_t1w_baseline_${TS_HEAD} -
auglab atlas-liver-hcc_t1w_auglab_default_${TS_HEAD} -
auglab atlas-liver-hcc_t1w_synthseg_noEM_${TS_HEAD} -
auglab atlas-liver-hcc_t1w_synthseg_EM_${TS_HEAD} -
auglab atlas-liver-hcc_t1w_srcsm_${TS_HEAD} -
auglab atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val000_${TS_HEAD} -
auglab atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val100_${TS_HEAD} -
auglab atlas-liver-hcc_t1w_baseline_kmeans_${TS_LAD} ablations
auglab atlas-liver-hcc_t1w_baseline_kmeans_label_remap_${TS_LAD} ablations
auglab atlas-liver-hcc_t1w_baseline_kmeans_label_remap_voronoi_${TS_LAD} ablations
nnUNet atlas-liver-hcc_t1w_v26_6_2_train050_val100_${TS_LAD} ablations
TASKS

: > "${PACK_DIR}/eval_joblist.txt"
while read -r category run_id subdir; do
    [ -n "${category}" ] || continue
    for fold in 0 1 2; do
        for item in ${ITEMS}; do
            echo "${category} ${run_id} ${subdir} ${fold} ${item}" >> "${PACK_DIR}/eval_joblist.txt"
        done
    done
done < "${PACK_DIR}/eval_tasks.txt"
echo "[pack] eval jobs: $(wc -l < "${PACK_DIR}/eval_joblist.txt") (expect 264 = 11 runs x 3 folds x 8 items)"

cat "${PACK_DIR}/eval_joblist.txt" | xargs -P 8 -L 1 bash -c 'run_eval "$@"' _
rc_eval=$?
echo "[pack] evaluate phase done rc=${rc_eval}"

echo "[pack] ==== PHASE 5: merge each fold's 8 items into eval_all.csv ===="
for base in "${METRICS_ROOT}/atlas_liver_hcc_model/t1w" "${METRICS_ROOT}/atlas_liver_hcc_model/t1w/ablations"; do
    [ -d "${base}" ] || continue
    find "${base}" -mindepth 1 -maxdepth 1 -type d -not -name ablations | while read -r run_dir; do
        run_name="$(basename "${run_dir}")"
        run_id="${run_name#nnUNet_}"; run_id="${run_id#auglab_}"
        for fold_dir in "${run_dir}"/fold*/; do
            [ -d "${fold_dir}" ] || continue
            fold="$(basename "${fold_dir}" | sed 's/fold//')"
            present=()
            for item in ${ITEMS}; do
                [ -f "${fold_dir}${item}_metrics.csv" ] && present+=("${item}")
            done
            if [ ${#present[@]} -gt 0 ]; then
                "${PROJECT_ROOT}/.venv/bin/python" "${PROJECT_ROOT}/benchmark/00_commun_scripts/00_03_evaluate/summarize_fold.py" \
                    "${fold_dir}" "${run_id}" "${fold}" --group-col contrast --groups-word Contrasts \
                    --title-suffix " | LLD-MMRI-MALIGNANT (ICC+metastasis) MRI | lesion only" \
                    --groups "${present[@]}" > /dev/null 2>&1
            fi
        done
    done
done
echo "[pack] eval_all.csv merge done"

echo "[pack] ==== FINAL AUDIT: exact row counts on every metrics CSV ===="
bad=0
while read -r f; do
    n=$(( $(wc -l < "$f") - 1 ))
    if [ "${n}" -ne "${N_CASES}" ]; then echo "[pack] BAD ROW COUNT (${n}/${N_CASES}): ${f}"; bad=$((bad+1)); fi
done < <(find "${METRICS_ROOT}/atlas_liver_hcc_model/t1w" -name "*_metrics.csv" 2>/dev/null | sort)
total=$(find "${METRICS_ROOT}/atlas_liver_hcc_model/t1w" -name "*_metrics.csv" 2>/dev/null | wc -l)
echo "[pack] total metrics files: ${total} (expect 264 = 11 runs x 3 folds x 8 items); bad row counts: ${bad}"
echo "[pack] ==== DONE (verify_fail=${verify_fail}, eval rc=${rc_eval}, bad_rows=${bad}) ===="
