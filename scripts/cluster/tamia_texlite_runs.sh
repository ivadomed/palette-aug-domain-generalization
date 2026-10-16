#!/usr/bin/env bash
# Run table for PALETTE + REDUCED Auglab ("texlite"; scripts/cluster/tamia_pack_texlite.sh). Same row format and the same
# shared predict/eval job bodies as rungs 6/7 (tamia_rung6_pv_{predict,eval}_job.sh, ROSTER=this file): category `auglab`,
# FLAT metrics next to the headline OURS runs, the texlite predict wrappers (TRAINER = the plain val000 trainer the run was
# trained with), and RUN_IDs resolved from the texlite pack recording (no timestamps here). The 2nd field is the POST GROUP
# (= the training setting): one predict job per group. The liver setting (atlas_t1w) is predicted + evaluated by its own job
# (atlas-liver-hcc 05_predict/05_35_tamia_texlite_predict_eval.sh), not through these rows.
# Row: name|group|dir|env|tamia env|tc|predict wrapper|args|RUN prefix|expected predictions|eval kind|expected scored cases
# Arm: TEXLITE_ARM unset/empty = "texlite" (the 2026-10-15 launch), TEXLITE_ARM=keepres = "texlite_keepres"
# (GaussianBlur + SimulateLowRes kept; METHOD auglabAugTexliteKeepres_..., wrappers one number up, own pack root),
# TEXLITE_ARM=edgeonly (headline config with only Scharr + RandomConv off; METHOD auglabAugEdgeonly_..., two numbers up).
# Rows = the DEVELOPMENT settings only (on-harmony T1w, open-ms flair, chaos t2spir); glioma t1n, mandible cbct (+ hanseg/pddca)
# and HCC t1w are held out for the paper and their texlite/keepres runs were cancelled 2026-10-15 (rows removed).
# on-harmony: eval kind onh_best = checkpoint_best predictions in the flat fold{k}/<item>/ (since 2026-10-06; the RAS->native
# resample runs inside each predict job), so the predict job's legacy fold{k}/final/ resample replay (kind "onh") is skipped.
source scripts/cluster/tamia_rung6_pv_runs.sh     # shared helpers (rung6_eval_pre_env, rung6_predict_extra_env) + dir vars/counts
case "${TEXLITE_ARM:-}" in
  "") PT=auglabAugTexlite_v26_6_2_train050_val000; _NM=05_40; _NC=05_55; _NO=05_40 ;;
  keepres) PT=auglabAugTexliteKeepres_v26_6_2_train050_val000; _NM=05_41; _NC=05_56; _NO=05_41 ;;
  edgeonly) PT=auglabAugEdgeonly_v26_6_2_train050_val000; _NM=05_42; _NC=05_57; _NO=05_42 ;;
  *) echo "tamia_texlite_runs.sh: unknown TEXLITE_ARM='${TEXLITE_ARM}'" >&2; return 2 2>/dev/null || exit 2 ;;
esac
RUNG_ROWS=(
"onh_t1w|onh_t1w|$O|env.sh|tamia_env_onharmony.sh|T1w|${_NO}_predict_t1w_${PT}.sh||on-harmony_T1w_${PT}|$ONP|onh_best|$ONE"
"openms_flair|openms_flair|$M|env.sh|tamia_env_openms.sh|flair|${_NM}_predict_flair_${PT}.sh||open-ms_flair_${PT}|$OM|openms_flair|$OM"
"chaos_t2spir|chaos_t2spir|$C|env_t2spir.sh|tamia_env_chaos.sh|t2spir|${_NC}_predict_t2spir_${PT}.sh||chaos_t2spir_${PT}|$CH|chaos:61|$CH"
)
RUNG_CAT=auglab; RUNG_MSUB=""
# run id from the pack recording: <PACKS_ROOT>/<any pack>/fold<k>_<prefix>_<timestamp>.sh -> exactly one distinct id
rung_resolve_run() {
    local prefix="$2" root ids; root="$(cat "${SCRATCH:-/scratch/${USER:0:1}/${USER}}/_packruns_texlite${TEXLITE_ARM:+_${TEXLITE_ARM}}_root.txt")"
    ids="$(ls "${root}"/*/fold[0-9]_"${prefix}"_[0-9]*.sh 2>/dev/null | xargs -r -n1 basename | sed 's/^fold[0-9]_//; s/\.sh$//' | sort -u)"
    [ "$(echo "${ids}" | grep -c .)" = 1 ] || { echo "RESOLVE_FAILED(${prefix})"; return 1; }
    echo "${ids}"
}
