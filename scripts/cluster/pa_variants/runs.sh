#!/usr/bin/env bash
# Run table of the PA variants round 2 (scripts/cluster/pa_variants/launch.sh), for the shared rung-6/7 predict/eval job bodies
# (scripts/cluster/tamia_rung6_pv_{predict,eval}_job.sh, ROSTER=this file). Same row format as tamia_texlite_runs.sh:
# category `auglab`, FLAT metrics next to the headline PA runs, each arm's own predict wrapper (TRAINER = the plain val000 trainer
# it was trained with), RUN_IDs resolved from the pack recording (no timestamps here). 2nd field = POST GROUP (= the setting).
# Row: name|group|dir|env|tamia env|tc|predict wrapper|args|RUN prefix|expected predictions|eval kind|expected scored cases
# PA_ARMS (optional, default "ctrl p075 edge2x alphawide"): restrict the rows to these arms (e.g. the fold 1-2 extension).
# Folds: the job bodies read RUNG_FOLDS (fold-0 screen: RUNG_FOLDS=0).
# PA_ROUND (default 2): 3 = round 3 arms (redistoff structoff structoffredistoff structoffbluroff), RUN_IDs from the round-3 pack root.
source scripts/cluster/tamia_rung6_pv_runs.sh     # shared helpers (rung6_eval_pre_env, rung6_predict_extra_env) + dir vars/counts
declare -A PA_METHOD=([ctrl]=auglabAugCtrl_v26_6_2_train050_val000 [p075]=auglabAugP075_v26_6_2_train075_val000
                      [edge2x]=auglabAugEdge2x_v26_6_2_train050_val000 [alphawide]=auglabAugAlphaWide_v26_6_2_train050_val000
                      [unsharpoff]=auglabAugUnsharpOff_v26_6_2_train050_val000 [redistoff]=auglabAugRedistOff_v26_6_2_train050_val000
                      [edgehalf]=auglabAugEdgeHalf_v26_6_2_train050_val000 [texhalf]=auglabAugTexHalf_v26_6_2_train050_val000
                      [structoff]=auglabAugStructOff_v26_6_2_train050_val000
                      [structoffredistoff]=auglabAugStructOffRedistOff_v26_6_2_train050_val000 [structoffbluroff]=auglabAugStructOffBlurOff_v26_6_2_train050_val000)
declare -A PA_NUM_O=([ctrl]=05_43 [p075]=05_44 [edge2x]=05_45 [alphawide]=05_46     # on-harmony / open-ms predict numbers
                     [redistoff]=05_48 [structoff]=05_51 [structoffredistoff]=05_52 [structoffbluroff]=05_53)
declare -A PA_NUM_C=([ctrl]=05_58 [p075]=05_59 [edge2x]=05_60 [alphawide]=05_61     # chaos predict numbers
                     [redistoff]=05_63 [structoff]=05_66 [structoffredistoff]=05_67 [structoffbluroff]=05_68)
if [ "${PA_ROUND:-2}" = 3 ]; then _PA_DEFAULT_ARMS="redistoff structoff structoffredistoff structoffbluroff"; _PA_TAG=pa_variants_r3
else _PA_DEFAULT_ARMS="ctrl p075 edge2x alphawide"; _PA_TAG=pa_variants; fi
RUNG_ROWS=()
for _a in ${PA_ARMS:-${_PA_DEFAULT_ARMS}}; do _m="${PA_METHOD[${_a}]:?unknown arm ${_a}}"
RUNG_ROWS+=(
"${_a}_onh_t1w|onh_t1w|$O|env.sh|tamia_env_onharmony.sh|T1w|${PA_NUM_O[${_a}]}_predict_t1w_${_m}.sh||on-harmony_T1w_${_m}|$ONP|onh_best|$ONE"
"${_a}_openms_flair|openms_flair|$M|env.sh|tamia_env_openms.sh|flair|${PA_NUM_O[${_a}]}_predict_flair_${_m}.sh||open-ms_flair_${_m}|$OM|openms_flair|$OM"
"${_a}_chaos_t2spir|chaos_t2spir|$C|env_t2spir.sh|tamia_env_chaos.sh|t2spir|${PA_NUM_C[${_a}]}_predict_t2spir_${_m}.sh||chaos_t2spir_${_m}|$CH|chaos:61|$CH"
)
done
RUNG_CAT=auglab; RUNG_MSUB=""
# run id from the pack recording: <PACKS_ROOT>/<any pack>/fold<k>_<prefix>_<timestamp>.sh -> exactly one distinct id
rung_resolve_run() {
    local prefix="$2" root ids; root="$(cat "${SCRATCH:-/scratch/${USER:0:1}/${USER}}/_packruns_${_PA_TAG}_root.txt")"
    ids="$(ls "${root}"/*/fold[0-9]_"${prefix}"_[0-9]*.sh 2>/dev/null | xargs -r -n1 basename | sed 's/^fold[0-9]_//; s/\.sh$//' | sort -u)"
    [ "$(echo "${ids}" | grep -c .)" = 1 ] || { echo "RESOLVE_FAILED(${prefix})"; return 1; }
    echo "${ids}"
}
