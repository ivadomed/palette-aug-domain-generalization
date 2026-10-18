#!/usr/bin/env bash
# Run table of the ssem50 arm ("SynthSeg-EM (+Auglab; 50%)", METHOD auglabAug_synthseg_EM_train050_val000) for the shared TamIA
# predict / eval job bodies (scripts/cluster/tamia_rung6_pv_{predict,eval}_job.sh, ROSTER=this file), queued by
# `scripts/cluster/ssem50/launch.sh queue-post`. Same row format as tamia_rung6_pv_runs.sh / pa_variants/runs.sh; category `auglab`,
# FLAT metrics next to the headline runs (RUNG_MSUB=""), each setting's generated predict wrapper, RUN_IDs resolved from the pack
# recording (no timestamps here). 2nd field = post group (= the training setting; hanseg / pddca ride on tf2_cbct). Expected counts =
# the rung-6 table's (same test sets). NOT in this table: atlas_t1w (no eval kind in the shared eval job: own test set via atlas 05_35 with TEXLITE_ARM=ssem50 ATLAS_OWN_ONLY=1),
# chaos -> amos / sliver07 fov_crop, breast and liver companions (see VERIFICATION.md).
# Row: name|group|dir|env|tamia env|tc|predict wrapper|args|RUN prefix|expected predictions|eval kind|expected scored cases
source scripts/cluster/tamia_rung6_pv_runs.sh     # shared helpers (rung6_eval_pre_env, rung6_predict_extra_env) + dir vars/counts
_M=auglabAug_synthseg_EM_train050_val000
PSG=pancreas_disease/pansegdata/5_scripts_pansegdata   # pansegdata (TamIA since 2026-10-18; eval kind "panseg")
RUNG_ROWS=(
"brats_t1n|brats_t1n|$B|env.sh|tamia_env.sh|t1n|05_83_predict_t1n_${_M}.sh||brats2024-glioma_t1n_${_M}|$BR|brats:051|$BR"
"brats_t2w|brats_t2w|$B|env_t2w.sh|tamia_env.sh|t2w|05_84_predict_t2w_${_M}.sh||brats2024-glioma_t2w_${_M}|$BR|brats:052|$BR"
"brats_t2f|brats_t2f|$B|env_t2f.sh|tamia_env.sh|t2f|05_85_predict_t2f_${_M}.sh||brats2024-glioma_t2f_${_M}|$BR|brats:053|$BR"
"brats_t1c|brats_t1c|$B|env_t1c.sh|tamia_env.sh|t1c|05_86_predict_t1c_${_M}.sh||brats2024-glioma_t1c_${_M}|$BR|brats:054|$BR"
"onh_T1w|onh_T1w|$O|env.sh|tamia_env_onharmony.sh|T1w|05_91_predict_t1w_${_M}.sh||on-harmony_T1w_${_M}|$ONP|onh_best|$ONE"
"onh_T2w|onh_T2w|$O|env_t2w.sh|tamia_env_onharmony.sh|T2w|05_92_predict_t2w_${_M}.sh||on-harmony_T2w_${_M}|$ONP|onh_best|$ONE"
"onh_dwi_ap|onh_dwi_ap|$O|env_dwi.sh|tamia_env_onharmony.sh|dwi_ap|05_93_predict_dwi_ap_${_M}.sh||on-harmony_dwi_ap_${_M}|$ONP|onh_best|$ONE"
"openms_flair|openms_flair|$M|env.sh|tamia_env_openms.sh|flair|05_54_predict_flair_${_M}.sh||open-ms_flair_${_M}|$OM|openms_flair|$OM"
"openms_t1w|openms_t1w|$M|env_t1w.sh|tamia_env_openms.sh|t1w|05_55_predict_t1w_${_M}.sh||open-ms_t1w_${_M}|$OM|openms_t1w|$OM"
"tf2_cbct|tf2_cbct|$TF|env.sh|tamia_env_toothfairy2.sh|cbct|05_32_predict_cbct_${_M}.sh||toothfairy2_cbct_${_M}|cbct:71|tf2|cbct:71"
"hanseg_sif|tf2_cbct|$HS|env.sh|tamia_env_hanseg.sh|cbct|05_31_predict_${_M}.sh|all ct mrt1|toothfairy2_cbct_${_M}|sif/ct:42 sif/mrt1:41|hanseg|ct:42 mrt1:41"
"pddca_sif|tf2_cbct|$PD|env.sh|tamia_env_pddca.sh|cbct|05_31_predict_${_M}.sh|all ct|toothfairy2_cbct_${_M}|sif/ct:40|pddca|ct:40"
"ispy2_t1wce|ispy2_t1wce|$I|env.sh|tamia_env_ispy2.sh|t1wce|05_50_predict_t1wce_${_M}.sh||ispy2_t1wce_${_M}|$IS|ispy2:100|$IS"
"ispy2_t2w|ispy2_t2w|$I|env_t2w.sh|tamia_env_ispy2.sh|t2w|05_51_predict_t2w_${_M}.sh||ispy2_t2w_${_M}|$IS|ispy2:101|$IS"
"chaos_t1in|chaos_t1in|$C|env.sh|tamia_env_chaos.sh|t1in|05_69_predict_t1in_${_M}.sh||chaos_t1in_${_M}|$CH|chaos:60|$CH"
"chaos_t2spir|chaos_t2spir|$C|env_t2spir.sh|tamia_env_chaos.sh|t2spir|05_70_predict_t2spir_${_M}.sh||chaos_t2spir_${_M}|$CH|chaos:61|$CH"
"pelvic_ct|pelvic_ct|$P|env.sh|tamia_env_totalseg-pelvic.sh|ct|05_24_predict_ct_${_M}.sh||totalseg-pelvic_ct_${_M}|$PV|pelvic|$PV"
"pelvic_mri|pelvic_mri|$P|env_mri.sh|tamia_env_totalseg-pelvic.sh|mri|05_25_predict_mri_${_M}.sh||totalseg-pelvic_mri_${_M}|$PV|pelvic|$PV"
"panseg_t1wce|panseg_t1wce|$PSG|env.sh|tamia_env_pansegdata.sh|t1wce|05_32_predict_t1wce_${_M}.sh||pansegdata_t1wce_${_M}|t1wce:42 t2w:42|panseg|t1wce:42 t2w:42"
"panseg_t2w|panseg_t2w|$PSG|env_t2w.sh|tamia_env_pansegdata.sh|t2w|05_33_predict_t2w_${_M}.sh||pansegdata_t2w_${_M}|t1wce:42 t2w:42|panseg|t1wce:42 t2w:42"
)
# Companion rows (hanseg_sif / pddca_sif) are dropped unless SSEM50_COMPANIONS=1: 2026-10-18 decision, companions run on Vulcan after the
# checkpoint sync. The predict / eval job bodies source this file with the queue-post environment, so the drop applies to them too.
# SSEM50_COMPANIONS=only (2026-10-18, `launch.sh queue-companions`): keep ONLY the companion rows, so a companion predict job behind the
# setting's own predict job never re-predicts the own test set.
case "${SSEM50_COMPANIONS:-0}" in
  1) ;;
  only) _k=(); for _r in "${RUNG_ROWS[@]}"; do case "${_r%%|*}" in *_sif) _k+=("$_r");; esac; done; RUNG_ROWS=("${_k[@]}") ;;
  *) _k=(); for _r in "${RUNG_ROWS[@]}"; do case "${_r%%|*}" in *_sif) ;; *) _k+=("$_r");; esac; done; RUNG_ROWS=("${_k[@]}") ;;
esac
RUNG_CAT=auglab; RUNG_MSUB=""
# run id from the pack recording: <PACKS_ROOT>/<any pack>/fold<k>_<prefix>_<timestamp>.sh -> exactly one distinct id
rung_resolve_run() {
    local prefix="$2" root ids; root="$(cat "${SSEM50_STATE:-${SCRATCH:-/scratch/${USER:0:1}/${USER}}/_packruns_ssem50_root.txt}")"
    ids="$(ls "${root}"/*/fold[0-9]_"${prefix}"_[0-9]*.sh 2>/dev/null | xargs -r -n1 basename | sed 's/^fold[0-9]_//; s/\.sh$//' | sort -u)"
    [ "$(echo "${ids}" | grep -c .)" = 1 ] || { echo "RESOLVE_FAILED(${prefix})"; return 1; }
    echo "${ids}"
}
