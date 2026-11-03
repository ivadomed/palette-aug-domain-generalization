#!/usr/bin/env bash
# LOW-DATA texture step: the arm -> config / METHOD / RUN_ID block shared by the dataset wrappers
# (chaos 04_93, pansegdata 04_45). Sourced, never run. The two arms are the ladder's rung 4.5 and rung 5:
#   flat    = flat fill        transform_params_gpu_baseline_kmeans_label_remap_voronoi_flatfill_spatialDA_train050.json
#   texture = PALETTE alone    transform_params_gpu_v26_6_2_synth_spatialDA_train050.json
# Caller sets (after sourcing its env.sh): AUGLAB_CONFIGS_DIR, and passes "<flat|texture> <N> [RUN_ID]".
# Sets: ARM, N, NN, CFG, METHOD (lowdata_n<NN>_<arm>), RUN_ID (default <dataset>_<contrast>_<METHOD>_<timestamp>), and exports
# AUGLAB_PARAMS_GPU_JSON. RUN_IDs avoid the headline method substrings (flatfill_, v26_6_2_train050, _val000_) so no
# headline glob or roster resolves a low-data run; the RUN_ID must carry _<contrast>_lowdata_n<NN>_<arm>_ (LowDataMixin reads
# N from it, lowdata_texture_step.py the level and arm). The Pelvis wrapper (totalseg-pelvic 04_35) predates this file and
# keeps its own copy of the same block.
ARM="${1:?arm required (flat|texture)}"
N="${2:?N required}"
NN="$(printf '%02d' "${N}")"
case "${ARM}" in
  flat)    CFG="transform_params_gpu_baseline_kmeans_label_remap_voronoi_flatfill_spatialDA_train050.json" ;;
  texture) CFG="transform_params_gpu_v26_6_2_synth_spatialDA_train050.json" ;;
  *) echo "ERROR: unknown arm '${ARM}' (flat|texture)" >&2; exit 1 ;;
esac
: "${AUGLAB_CONFIGS_DIR:?AUGLAB_CONFIGS_DIR not set}" "${DATASET_NAME:?}" "${TRAINING_CONTRAST:?}"
[ -f "${AUGLAB_CONFIGS_DIR}/${CFG}" ] || { echo "ERROR: missing ${AUGLAB_CONFIGS_DIR}/${CFG}" >&2; exit 1; }
export AUGLAB_PARAMS_GPU_JSON="${AUGLAB_CONFIGS_DIR}/${CFG}"
METHOD="lowdata_n${NN}_${ARM}"
RUN_ID="${3:-${DATASET_NAME}_${TRAINING_CONTRAST}_${METHOD}_$(date +%Y%m%d_%H%M%S)}"
case "${RUN_ID}" in *"_${TRAINING_CONTRAST}_lowdata_n${NN}_${ARM}_"*) ;;
  *) echo "ERROR: RUN_ID ${RUN_ID} lacks _${TRAINING_CONTRAST}_lowdata_n${NN}_${ARM}_" >&2; exit 1 ;; esac
case "${RUN_ID}" in *_val000_*|*flatfill_*|*v26_6_2_train050*) echo "ERROR: RUN_ID ${RUN_ID} carries a headline method substring" >&2; exit 1 ;; esac
