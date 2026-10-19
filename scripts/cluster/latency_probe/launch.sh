#!/usr/bin/env bash
# TRAINING-LATENCY PROBE: per-epoch training time of PALETTE Augmentation (PA) next to its competitors, per task, on the
# same GPU type at the same time on the same node class. Modelled on the sizing probes (pansegdata 04_24/04_26 and the
# texlite PROBE mode of scripts/cluster/tamia_pack_texlite.sh): each run is RECORDED through the dataset's own headline
# train wrapper (so trainer, AugLab config, DA workers, compile flag, iterations/epoch and env are exactly the paper's),
# then the recorded fold-0 command is patched for the probe only and packed onto ONE whole 4-GPU node (one run per GPU).
#
# Per task, two packs of 4 runs (4 runs per node in both, so CPU / data-loading contention is the same for every run):
#   P1: auglab_default | synthseg_EM | srcsm | pa
#   P2: baseline | auglab_default (replicate: within-node anchor + run-to-run noise) | pa_matched | dualval (headline
#       DualVal trainer as-is, context only; a PA replicate where the headline OURS run was not DualVal)
#   pa         = the headline OURS config (..._train050.json) with the PLAIN single-validation val000 trainer. Where the
#                headline run used the DualVal trainer, the recorded command's -tr <X>AugLabDualVal becomes
#                <X>AugLabDefault and AUGLAB_VAL_PARAMS_GPU_JSON is cleared (what DualVal adds is a 2nd, synthetic,
#                validation pass per epoch, not part of the method). DA workers / compile flag as in the headline wrapper.
#   pa_matched = pa with nnUNet_n_proc_DA and nnUNet_compile set to auglab_default's values (several headline OURS
#                wrappers run DA_WORKERS=0 / compile=0 while the competitors run 8 / 1: that gap times the data loader
#                and torch.compile, not the augmentation). THE fair ratio for the augmentation cost.
# Probe-only patches (every run, grep-verified before submit): NNUNET_NUM_EPOCHS=25, nnUNet_wandb_enabled=0,
#   --disable_checkpointing, post-training validation skipped (train_noval.py), results in a THROWAWAY base
#   ${LAT}/results/<cluster>/<task>/ (never the real results tree). Fold 0, real training data.
# P3 (added 2026-10-19, `record3`): baseline | pa_noauglab | ad_rep2 | pa_matched, one whole node per task.
#   pa_noauglab = PA without the AugLab pipeline: the ablation ladder's real-fill rung (rung 5, "PALETTE alone",
#                 transform_params_gpu_v26_6_2_synth_spatialDA_train050.json = PALETTE p=0.5 + flip + nnU-Net rotation/
#                 scaling), recorded through the dataset's rung-5 val000 wrapper (last ROWS field). A DualVal trainer in
#                 the recorded command (chaos t1in 04_48) becomes <X>AugLabDefault + val json cleared, as for pa.
#                 nnUNet_n_proc_DA / nnUNet_compile = the baseline's of the same pack (ratio pa_noauglab / baseline).
#   pa_matched  = as in P2, matched to ad_rep2 (ratio pa_matched / ad_rep2 re-measured on the P3 node).
# healthy-spine-tum is not measurable (externally computed results, no data / train pipeline in this repo).
# pansegdata is not staged on TamIA (no data, no trainer shim in TamIA's venv) -> its packs run on Vulcan L40S.
#
#   bash scripts/cluster/latency_probe/launch.sh record [TASK...]    # record + patch + verify P1/P2 (no submit)
#   bash scripts/cluster/latency_probe/launch.sh record3 [TASK...]   # same for P3 (PA-No-Auglab pack)
#   bash scripts/cluster/latency_probe/launch.sh submit [TASK...]    # submit the recorded packs (1 whole-node job each)
#   bash scripts/cluster/latency_probe/launch.sh cleanup             # delete the throwaway results once report.sh collected
# Report: bash scripts/cluster/latency_probe/report.sh  (see its header).
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
REPO="$(pwd)"
if [ -d "/scratch/${USER:0:1}/${USER}" ]; then CLUSTER=tamia; export SCRATCH="/scratch/${USER:0:1}/${USER}"
else CLUSTER=vulcan; export SCRATCH="${SCRATCH:-/scratch/${USER}}"; fi
LAT="${LAT_DIR:-${SCRATCH}/review_20261015/latency}"
PACKS="${LAT}/packs"; RES="${LAT}/results/${CLUSTER}"; MAN="${LAT}/manifest_${CLUSTER}.tsv"
NOVAL="${REPO}/scripts/cluster/latency_probe/train_noval.py"
CFGDIR="${REPO}/sub-workspaces/auglab_workspace/AugLab/auglab/configs"
EPOCHS="${PROBE_EPOCHS:-25}"
MODE="${1:?usage: record|submit|cleanup [TASK...]}"; shift || true
T=benchmark/02_tasks
if [ -d "$T/liver_hcc/atlas-liver-hcc" ]; then AD_DIR="$T/liver_hcc/atlas-liver-hcc/5_scripts_atlas-liver-hcc"
else AD_DIR="benchmark/03_archive/atlas-liver-hcc/5_scripts_atlas-liver-hcc"; fi   # TamIA checkout (pre-2026-10-14 layout)

CFG_AD=transform_params_gpu_default01-23.json
CFG_SEM=transform_params_gpu_default01-23_Synthseg_EM.json
CFG_SRC=transform_params_gpu_srcsm_semrandconv.json
CFG_PA=transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json
CFG_VAL=transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json
CFG_NOAUG=transform_params_gpu_v26_6_2_synth_spatialDA_train050.json

# task|cluster|scripts dir|tamia env|contrast|baseline|auglab_default|synthseg_EM|srcsm|PA source|PA source is DualVal (1/0)|DualVal wrapper (- = none)|trainer: baseline|trainer: auglab|trainer: PA|trainer: DualVal|PA-No-Auglab wrapper (ladder rung 5, real fill)
ROWS=(
"brats|tamia|$T/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma|tamia_env.sh|t1n|04_19_train_t1n_baseline.sh|04_21_train_t1n_auglab_default.sh|04_22_train_t1n_synthseg_EM.sh|04_26_train_t1n_srcsm.sh|04_28_train_t1n_auglabAug_v26_6_2_train050_val000.sh|0|-|nnUNetTrainerBraTS2024GliomaT1nBaseline|nnUNetTrainerBraTS2024GliomaAugLabDefault|nnUNetTrainerBraTS2024GliomaAugLabDefault|-|04_89_train_t1n_v26_6_2_train050_val000.sh"
"openms|tamia|$T/brain_ms/open-ms/5_scripts_open-ms|tamia_env_openms.sh|flair|04_01_train_baseline.sh|04_04_train_auglab_default.sh|04_02_train_synthseg_EM.sh|04_20_train_srcsm.sh|04_22_train_auglabAug_v26_6_2_train050_val000.sh|0|-|nnUNetTrainerOpenMSBaseline|nnUNetTrainerOpenMSAugLabDefault|nnUNetTrainerOpenMSAugLabDefault|-|04_26_train_v26_6_2_train050_val000.sh"
"chaos|tamia|$T/abdomen_healthy/chaos/5_scripts_chaos|tamia_env_chaos.sh|t1in|04_33_train_t1in_baseline.sh|04_35_train_t1in_auglab_default.sh|04_36_train_t1in_synthseg_EM.sh|04_42_train_t1in_srcsm.sh|04_09_train_auglabAug_v26_6_2_train050_val000.sh|0|-|nnUNetTrainerCHAOSBaseline|nnUNetTrainerCHAOSAugLabDefault|nnUNetTrainerCHAOSAugLabV26_6_2|-|04_48_train_t1in_v26_6_2.sh"
"onharmony|tamia|$T/brain_healthy/on-harmony/5_scripts_on-harmony|tamia_env_onharmony.sh|T1w|04_07_train_t1w_baseline.sh|04_11_train_t1w_auglab_default.sh|04_09_train_t1w_synthseg_EM.sh|04_20_train_t1w_srcsm.sh|04_22_train_t1w_auglabAug_v26_6_2_train050_val000.sh|0|04_26_train_t1w_auglabAug_v26_6_2_dualval.sh|nnUNetTrainerOnHarmonyBaseline|nnUNetTrainerOnHarmonyAugLabDefault|nnUNetTrainerOnHarmonyAugLabDefault|nnUNetTrainerOnHarmonyAugLabDualVal|04_53_train_t1w_v26_6_2_train050_val000.sh"
"ispy2|tamia|$T/breast_cancer/ispy2/5_scripts_ispy2|tamia_env_ispy2.sh|t1wce|04_01_train_t1wce_baseline.sh|04_02_train_t1wce_auglab_default.sh|04_04_train_t1wce_synthseg_EM.sh|04_05_train_t1wce_srcsm.sh|04_06_train_t1wce_auglabAug_v26_6_2_dualval.sh|1|04_06_train_t1wce_auglabAug_v26_6_2_dualval.sh|nnUNetTrainerISPY2Baseline|nnUNetTrainerISPY2AugLabDefault|nnUNetTrainerISPY2AugLabDefault|nnUNetTrainerISPY2AugLabDualVal|04_30_train_t1wce_v26_6_2_train050_val000.sh"
"toothfairy2|tamia|$T/mandible_healthy/toothfairy2/5_scripts_toothfairy2|tamia_env_toothfairy2.sh|cbct|04_01_train_baseline.sh|04_02_train_auglab_default.sh|04_04_train_synthseg_EM.sh|04_05_train_srcsm.sh|04_06_train_auglabAug_v26_6_2_dualval.sh|1|04_06_train_auglabAug_v26_6_2_dualval.sh|nnUNetTrainerToothFairy2Baseline|nnUNetTrainerToothFairy2AugLabDefault|nnUNetTrainerToothFairy2AugLabDefault|nnUNetTrainerToothFairy2AugLabDualVal|04_17_train_cbct_v26_6_2_train050_val000.sh"
"pelvic|tamia|$T/pelvis_healthy/totalseg-pelvic/5_scripts_totalseg-pelvic|tamia_env_totalseg-pelvic.sh|ct|04_01_train_ct_baseline.sh|04_02_train_ct_auglab_default.sh|04_04_train_ct_synthseg_EM.sh|04_05_train_ct_srcsm.sh|04_06_train_ct_auglabAug_v26_6_2_dualval.sh|1|04_06_train_ct_auglabAug_v26_6_2_dualval.sh|nnUNetTrainerTotalsegPelvicBaseline|nnUNetTrainerTotalsegPelvicAugLabDefault|nnUNetTrainerTotalsegPelvicAugLabDefault|nnUNetTrainerTotalsegPelvicAugLabDualVal|04_27_train_ct_v26_6_2_train050_val000.sh"
"atlas|tamia|${AD_DIR}|tamia_env_atlas-liver-hcc.sh|t1w|04_01_train_baseline.sh|04_02_train_auglab_default.sh|04_04_train_synthseg_EM.sh|04_05_train_srcsm.sh|04_06_train_auglabAug_v26_6_2_dualval.sh|1|04_06_train_auglabAug_v26_6_2_dualval.sh|nnUNetTrainerAtlasLiverHCCBaseline|nnUNetTrainerAtlasLiverHCCAugLabDefault|nnUNetTrainerAtlasLiverHCCAugLabDefault|nnUNetTrainerAtlasLiverHCCAugLabDualVal|04_14_train_t1w_v26_6_2_train050_val000.sh"
"pansegdata|vulcan|$T/pancreas_disease/pansegdata/5_scripts_pansegdata|-|t1wce|04_01_train_t1wce_baseline.sh|04_02_train_t1wce_auglab_default.sh|04_04_train_t1wce_synthseg_EM.sh|04_05_train_t1wce_srcsm.sh|04_06_train_t1wce_auglabAug_v26_6_2_dualval.sh|1|04_06_train_t1wce_auglabAug_v26_6_2_dualval.sh|nnUNetTrainerPANSEGDATABaseline|nnUNetTrainerPANSEGDATAAugLabDefault|nnUNetTrainerPANSEGDATAAugLabDefault|nnUNetTrainerPANSEGDATAAugLabDualVal|04_18_train_t1wce_v26_6_2_train050_val000.sh"
)
MINE=(); for r in "${ROWS[@]}"; do IFS='|' read -r t c _ <<<"$r"; [ "$c" = "${CLUSTER}" ] && MINE+=("$t"); done
SEL="${*:-${MINE[*]}}"
row_of() { local r t; for r in "${ROWS[@]}"; do t="${r%%|*}"; [ "$t" = "$1" ] && { echo "$r"; return 0; }; done; return 1; }

# --- recorded command helpers: decode the %q-quoted `bash -c <body>` line run_job wrote, patch the plain body -------------
decode_body() { local line; line="$(sed -n '4p' "$1")"; eval "set -- ${line}"; [ "$1" = bash ] && [ "$2" = -c ] || return 1; printf '%s\n' "$3"; }
get() { sed -n "s/^ *export $1=//p" "$2" | head -1 | tr -d "'"; }
sub_exact() {  # file, sed ERE, replacement, label: apply and require exactly one changed line
    local f="$1" n; n="$(grep -cE -- "$2" "$f" || true)"
    [ "$n" = 1 ] || { echo "PATCH_FAIL($4: ${n} matches of /$2/)"; return 1; }
    sed -i -E "s#$2#$3#" "$f"
}

record_run() {  # task pack gpu slot wrapper runid cfg_for_ad_slot(or "")
    local task="$1" pack="$2" gpu="$3" slot="$4" w="$5" rid="$6" adcfg="$7"
    local row dir tenv tc envf pdir="${PACKS}/${task}_${pack}"
    row="$(row_of "$task")"; IFS='|' read -r _ _ dir tenv tc _ <<<"$row"
    envf="$(grep -o 'source "$(dirname "$0")/../00_utils/env[a-z0-9_]*\.sh"' "${dir}/04_train/${w}" | head -1 | sed 's|.*/00_utils/||; s|"$||')"
    local before=0; [ -f "${pdir}/index.tsv" ] && before="$(grep -c . "${pdir}/index.tsv")"
    (
        export nnUNet_wandb_enabled=0
        source "${dir}/00_utils/${envf}"
        [ "${tenv}" = "-" ] || source "scripts/cluster/${tenv}"
        # throwaway bases AFTER the cluster env (tamia_env_* set the real dirs unconditionally); env.sh keeps set values
        export PREDICTIONS_ROOT="${RES}/${task}/01_predictions" RESULTS_DIR="${RES}/${task}/results" METRICS_ROOT="${RES}/${task}/02_metrics"
        export nnUNet_results="${RES}/${task}/01_predictions/nnUNet" NNUNET_RESULTS_BASE="${RES}/${task}/01_predictions/nnUNet"
        mkdir -p "${PREDICTIONS_ROOT}" "${RESULTS_DIR}" "${METRICS_ROOT}"
        # on-harmony 04_11 sets no AugLab config (train_common would export it EMPTY, see memory "AugLab JSON empty-env")
        if [ -n "${adcfg}" ]; then export AUGLAB_PARAMS_GPU_JSON="${REPO}/sub-workspaces/auglab_workspace/AugLab/auglab/configs/${adcfg}"; else unset AUGLAB_PARAMS_GPU_JSON; fi
        unset AUGLAB_VAL_PARAMS_GPU_JSON
        TRAIN_FOLDS="0" NNUNET_NUM_EPOCHS="${EPOCHS}" RUN_JOB_PACK_DIR="${pdir}" bash "${dir}/04_train/${w}" "${rid}" < /dev/null
    ) > "${pdir}/record_${slot}.log" 2>&1 || { echo "RECORD_FAILED ${task}/${slot} (see ${pdir}/record_${slot}.log)"; return 1; }
    local after; after="$(grep -c . "${pdir}/index.tsv")"
    [ "$after" = $((before + 1)) ] || { echo "RECORD_COUNT ${task}/${slot}: index ${before} -> ${after}"; return 1; }
    local cmd; cmd="$(tail -1 "${pdir}/index.tsv" | cut -f1)"
    echo "${slot}" >> "${pdir}/SLOTS"; echo "${gpu}" >> "${pdir}/GPUS"
    echo "${cmd}"
}

patch_run() {  # cmdfile slot pa_from_dv ad_da ad_compile
    local cmd="$1" slot="$2" fromdv="$3" ad_da="$4" ad_c="$5" body
    cp "${cmd}" "${cmd%.sh}.orig"
    body="${cmd%.sh}.body"; decode_body "${cmd}" > "${body}" || { echo "DECODE_FAIL ${cmd}"; return 1; }
    sub_exact "$body" "^( *export NNUNET_NUM_EPOCHS=).*$" "\\1'${EPOCHS}'" epochs || return 1
    sub_exact "$body" "^( *export nnUNet_wandb_enabled=).*$" "\\1'0'" wandb || return 1
    sub_exact "$body" "\\.venv/bin/nnUNetv2_train " ".venv/bin/python ${NOVAL} " noval || return 1
    sub_exact "$body" "(-num_gpus [0-9]+)" "\\1 --disable_checkpointing" nockpt || return 1
    # AugLab config paths -> this checkout's configs dir (an archived wrapper's stale hop count leaves "/<name>.json")
    local v val; for v in AUGLAB_PARAMS_GPU_JSON AUGLAB_VAL_PARAMS_GPU_JSON; do
        val="$(get "$v" "$body")"; [ -n "$val" ] || continue
        [ -f "${CFGDIR}/${val##*/}" ] || { echo "CFG_NOT_IN_CHECKOUT(${val##*/})"; return 1; }
        [ "$val" = "${CFGDIR}/${val##*/}" ] || sub_exact "$body" "^( *export ${v}=).*$" "\\1'${CFGDIR}/${val##*/}'" "cfgpath_${v}" || return 1
    done
    case "$slot" in pa|pa_matched|pa_rep)
        if [ "$fromdv" = 1 ]; then
            sub_exact "$body" "-tr (nnUNetTrainer[A-Za-z0-9_]*)AugLabDualVal " "-tr \\1AugLabDefault " trainer || return 1
            sub_exact "$body" "^( *export AUGLAB_VAL_PARAMS_GPU_JSON=).*$" "\\1''" valjson || return 1
        fi;;
    esac
    if [ "$slot" = pa_noauglab ]; then   # rung-5 wrapper; DualVal detected from the recorded trainer, loader = baseline's
        if grep -qE -- "-tr nnUNetTrainer[A-Za-z0-9_]*AugLabDualVal " "$body"; then
            sub_exact "$body" "-tr (nnUNetTrainer[A-Za-z0-9_]*)AugLabDualVal " "-tr \\1AugLabDefault " trainer || return 1
            sub_exact "$body" "^( *export AUGLAB_VAL_PARAMS_GPU_JSON=).*$" "\\1''" valjson || return 1
        fi
        sub_exact "$body" "^( *export nnUNet_n_proc_DA=).*$" "\\1${ad_da}" da || return 1
        sub_exact "$body" "^( *export nnUNet_compile=).*$" "\\1'${ad_c}'" compile || return 1
    fi
    if [ "$slot" = pa_matched ]; then
        sub_exact "$body" "^( *export nnUNet_n_proc_DA=).*$" "\\1${ad_da}" da || return 1
        sub_exact "$body" "^( *export nnUNet_compile=).*$" "\\1'${ad_c}'" compile || return 1
    fi
    { sed -n '1,3p' "${cmd%.sh}.orig"
      echo "# latency probe: body decoded from the recorded 'bash -c' command (${cmd##*/}.orig) and patched by launch.sh"
      cat "$body"; } > "${cmd}"
    rm -f "$body"
}

verify_run() {  # cmdfile slot task gpu -> prints one line, returns 1 on problem
    local cmd="$1" slot="$2" task="$3" gpu="$4" row ebl eau epa edv b ok=OK
    row="$(row_of "$task")"; IFS='|' read -r _ _ _ _ _ _ _ _ _ _ _ _ ebl eau epa edv _ <<<"$row"
    b="$(mktemp -p "${LAT}")"; tail -n +5 "$cmd" > "$b"
    local res tr cfgp cfg vcfg ep wb da cp pre it
    res="$(get nnUNet_results "$b")"; cfgp="$(get AUGLAB_PARAMS_GPU_JSON "$b")"; cfg="${cfgp##*/}"; vcfg="$(get AUGLAB_VAL_PARAMS_GPU_JSON "$b")"
    ep="$(get NNUNET_NUM_EPOCHS "$b")"; wb="$(get nnUNet_wandb_enabled "$b")"; da="$(get nnUNet_n_proc_DA "$b")"; cp="$(get nnUNet_compile "$b")"
    pre="$(get nnUNet_preprocessed "$b")"; it="$(get NNUNET_ITERS_PER_EPOCH "$b")"
    tr="$(grep -o -- '-tr [A-Za-z0-9_]*' "$b" | head -1 | cut -d' ' -f2)"
    case "$res" in "${LAT}/results/"*) ;; *) ok="${ok},RESULTS_NOT_THROWAWAY";; esac
    [ -e "$res" ] && ok="${ok},RESULTS_EXIST"
    [ -d "$pre" ] || ok="${ok},NO_PREPROC"
    [ "$ep" = "${EPOCHS}" ] || ok="${ok},EPOCHS($ep)"; [ "$wb" = 0 ] || ok="${ok},WANDB($wb)"
    grep -q "python ${NOVAL} " "$b" || ok="${ok},NO_NOVAL"; grep -q -- '--disable_checkpointing' "$b" || ok="${ok},NO_DISABLE_CKPT"
    local etr ecfg evcfg=""
    case "$slot" in
        baseline) etr="$ebl"; ecfg="";;
        ad|ad_rep|ad_rep2) etr="$eau"; ecfg="${CFG_AD}";;
        pa_noauglab) etr="$eau"; ecfg="${CFG_NOAUG}";;
        sem) etr="$eau"; ecfg="${CFG_SEM}";;
        srcsm) etr="$eau"; ecfg="${CFG_SRC}";;
        pa|pa_matched|pa_rep) etr="$epa"; ecfg="${CFG_PA}";;
        dualval) etr="$edv"; ecfg="${CFG_PA}"; evcfg="${CFG_VAL}";;
    esac
    [ "$tr" = "$etr" ] || ok="${ok},TRAINER($tr!=$etr)"
    [ "$cfg" = "$ecfg" ] || ok="${ok},CFG(${cfg:-none}!=${ecfg:-none})"
    [ -z "$cfgp" ] || [ -f "$cfgp" ] || ok="${ok},CFG_MISSING"
    [ "${vcfg##*/}" = "$evcfg" ] || ok="${ok},VALCFG(${vcfg##*/}!=${evcfg:-none})"
    rm -f "$b"
    printf '   %-4s %-11s GPU%s trainer=%s cfg=%s DA=%s compile=%s iters=%s\n' "${ok%%,*}" "$slot" "$gpu" "$tr" "${cfg:-none}" "$da" "${cp:-0}" "${it:-default}"
    [ "$ok" = OK ] || { echo "        PROBLEMS: ${ok#OK,}"; return 1; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${CLUSTER}" "$task" "$(row_of "$task" | cut -d'|' -f5)" "$(basename "${cmd%/*}")" "$gpu" \
        "$slot" "$tr" "${cfg:-none}" "$da" "${cp:-0}" "$res" "$cmd" >> "${MAN}"
}

if [ "${MODE}" = record ]; then
  mkdir -p "${PACKS}" "${RES}"
  [ -f "${MAN}" ] || printf 'cluster\ttask\tcontrast\tpack\tgpu\tslot\ttrainer\tcfg\tda\tcompile\tresults\tcmd\n' > "${MAN}"
  TS="$(date +%Y%m%d_%H%M%S)"; bad=0
  for task in ${SEL}; do
    row="$(row_of "$task")" || { echo "unknown task $task"; exit 2; }
    IFS='|' read -r _ cl dir tenv tc wbl wad wsem wsrc wpa fromdv wdv _ <<<"$row"
    [ "$cl" = "${CLUSTER}" ] || { echo "[record] ${task} belongs to ${cl}, this is ${CLUSTER} -- skipped"; continue; }
    if [ -e "${PACKS}/${task}_P1" ] || [ -e "${PACKS}/${task}_P2" ]; then echo "[record] ${task} already recorded -- skipped"; continue; fi
    mkdir -p "${PACKS}/${task}_P1" "${PACKS}/${task}_P2"
    echo "== ${task} (${tc}) ${dir}"
    pre="lat_${task}_${tc}"
    # P1
    c_ad="$(record_run "$task" P1 0 ad "$wad" "${pre}_ad_${TS}" "${CFG_AD}")" || { echo "$c_ad"; bad=1; continue; }
    patch_run "$c_ad" ad "$fromdv" - - || { bad=1; continue; }
    AD_DA="$(get nnUNet_n_proc_DA <(tail -n +5 "$c_ad"))"; AD_C="$(get nnUNet_compile <(tail -n +5 "$c_ad"))"; AD_C="${AD_C:-0}"
    for s in "sem|${wsem}|1" "srcsm|${wsrc}|2" "pa|${wpa}|3"; do IFS='|' read -r slot w g <<<"$s"
        c="$(record_run "$task" P1 "$g" "$slot" "$w" "${pre}_${slot}_${TS}" "")" || { echo "$c"; bad=1; continue; }
        patch_run "$c" "$slot" "$fromdv" "$AD_DA" "$AD_C" || bad=1
    done
    # P2
    if [ "$wdv" = "-" ]; then s4="pa_rep|${wpa}"; else s4="dualval|${wdv}"; fi
    for s in "baseline|${wbl}|0" "ad_rep|${wad}|1" "pa_matched|${wpa}|2" "${s4}|3"; do IFS='|' read -r slot w g <<<"$s"
        cfg=""; [ "$slot" = ad_rep ] && cfg="${CFG_AD}"
        rid="${pre}_${slot}_${TS}"; [ "$slot" = dualval ] && rid="${pre}_dualval_val000_${TS}"   # DualVal needs the _val000_ marker
        c="$(record_run "$task" P2 "$g" "$slot" "$w" "$rid" "$cfg")" || { echo "$c"; bad=1; continue; }
        patch_run "$c" "$slot" "$fromdv" "$AD_DA" "$AD_C" || bad=1
    done
    for p in P1 P2; do
      echo " -- ${task}_${p}"; mapfile -t SL < "${PACKS}/${task}_${p}/SLOTS"; mapfile -t GP < "${PACKS}/${task}_${p}/GPUS"; i=0
      while IFS=$'\t' read -r cmd _ _ _; do
        verify_run "$cmd" "${SL[$i]}" "$task" "${GP[$i]}" || bad=1; i=$((i+1))
      done < "${PACKS}/${task}_${p}/index.tsv"
      [ "$i" = 4 ] || { echo "   PACK HAS ${i} RUNS (expect 4)"; bad=1; }
    done
  done
  [ "$bad" = 0 ] && echo "VERIFY PASSED" || echo "VERIFY FAILED"
  exit 0
fi

if [ "${MODE}" = record3 ]; then   # P3: baseline | pa_noauglab | ad_rep2 | pa_matched (see header)
  mkdir -p "${PACKS}" "${RES}"
  [ -f "${MAN}" ] || printf 'cluster\ttask\tcontrast\tpack\tgpu\tslot\ttrainer\tcfg\tda\tcompile\tresults\tcmd\n' > "${MAN}"
  TS="$(date +%Y%m%d_%H%M%S)"; bad=0
  for task in ${SEL}; do
    row="$(row_of "$task")" || { echo "unknown task $task"; exit 2; }
    IFS='|' read -r _ cl dir tenv tc wbl wad _ _ wpa fromdv _ _ _ _ _ wnoaug <<<"$row"
    [ "$cl" = "${CLUSTER}" ] || { echo "[record3] ${task} belongs to ${cl}, this is ${CLUSTER} -- skipped"; continue; }
    [ -e "${PACKS}/${task}_P3" ] && { echo "[record3] ${task} already recorded -- skipped"; continue; }
    mkdir -p "${PACKS}/${task}_P3"; echo "== ${task} (${tc}) ${dir}"; pre="lat_${task}_${tc}"
    c_bl="$(record_run "$task" P3 0 baseline "$wbl" "${pre}_baseline_${TS}" "")" || { echo "$c_bl"; bad=1; continue; }
    patch_run "$c_bl" baseline "$fromdv" - - || { bad=1; continue; }
    BL_DA="$(get nnUNet_n_proc_DA <(tail -n +5 "$c_bl"))"; BL_C="$(get nnUNet_compile <(tail -n +5 "$c_bl"))"; BL_C="${BL_C:-0}"
    c="$(record_run "$task" P3 1 pa_noauglab "$wnoaug" "${pre}_pa_noauglab_val000_${TS}" "")" || { echo "$c"; bad=1; continue; }
    patch_run "$c" pa_noauglab "$fromdv" "$BL_DA" "$BL_C" || bad=1
    c_ad="$(record_run "$task" P3 2 ad_rep2 "$wad" "${pre}_ad_rep2_${TS}" "${CFG_AD}")" || { echo "$c_ad"; bad=1; continue; }
    patch_run "$c_ad" ad_rep2 "$fromdv" - - || { bad=1; continue; }
    AD_DA="$(get nnUNet_n_proc_DA <(tail -n +5 "$c_ad"))"; AD_C="$(get nnUNet_compile <(tail -n +5 "$c_ad"))"; AD_C="${AD_C:-0}"
    c="$(record_run "$task" P3 3 pa_matched "$wpa" "${pre}_pa_matched_${TS}" "")" || { echo "$c"; bad=1; continue; }
    patch_run "$c" pa_matched "$fromdv" "$AD_DA" "$AD_C" || bad=1
    echo " -- ${task}_P3"; mapfile -t SL < "${PACKS}/${task}_P3/SLOTS"; mapfile -t GP < "${PACKS}/${task}_P3/GPUS"; i=0
    while IFS=$'\t' read -r cmd _ _ _; do verify_run "$cmd" "${SL[$i]}" "$task" "${GP[$i]}" || bad=1; i=$((i+1)); done < "${PACKS}/${task}_P3/index.tsv"
    [ "$i" = 4 ] || { echo "   PACK HAS ${i} RUNS (expect 4)"; bad=1; }
  done
  [ "$bad" = 0 ] && echo "VERIFY PASSED" || echo "VERIFY FAILED"
  exit 0
fi

if [ "${MODE}" = submit ]; then
  for task in ${SEL}; do for p in ${PACKS_SEL:-P1 P2 P3}; do
    d="${PACKS}/${task}_${p}"; [ -f "$d/index.tsv" ] || { echo "[submit] ${task}_${p}: not recorded"; continue; }
    ls "$d"/pack_*.out >/dev/null 2>&1 && { echo "[submit] ${task}_${p}: already ran/submitted -- skipped"; continue; }
    [ -f "$d/JOBID" ] && { echo "[submit] ${task}_${p}: already submitted ($(cat "$d/JOBID")) -- skipped"; continue; }
    if [ "${CLUSTER}" = tamia ]; then gt=h100; cpus=48; mem=0; else gt=l40s; cpus=48; mem=460G; fi
    out="$(PACK_GPU_TYPE="$gt" PACK_NODE_GPUS=4 PACK_CPUS="$cpus" PACK_MEM="$mem" PACK_TIME="${PACK_TIME:-02:30:00}" PACK_CHAIN=1 \
           PACK_GPU_MAP="0 1 2 3" PACK_USE_MPS=0 PACK_JOB_NAME="latency_${task}_${p}" bash scripts/job_runner/run_job_pack_submit.sh "$d")"
    echo "$out"; jid="$(echo "$out" | grep -oE 'last job id=[0-9]+' | grep -oE '[0-9]+')"; echo "${jid}" > "$d/JOBID"
    echo "[submit] ${task}_${p}: job ${jid}"; sleep 5
  done; done
  exit 0
fi

if [ "${MODE}" = cleanup ]; then   # throwaway training outputs; only after report.sh collect copied every log
  miss=0
  while IFS=$'\t' read -r cl task _ pack _ slot _ _ _ _ res _; do
    [ "$cl" = cluster ] && continue
    [ -s "${LAT}/collected_${CLUSTER}/logs/${task}_${pack}_${slot}.txt" ] || { echo "not collected yet: ${task}_${pack}_${slot}"; miss=1; }
  done < "${MAN}"
  [ "$miss" = 0 ] || { echo "refusing to delete ${RES}"; exit 1; }
  [ -n "${RES}" ] && [ -d "${RES}" ] && case "${RES}" in */review_20261015/latency/results/*) rm -rf -- "${RES}"; echo "deleted ${RES}";; esac
  exit 0
fi
echo "unknown mode ${MODE}"; exit 2
