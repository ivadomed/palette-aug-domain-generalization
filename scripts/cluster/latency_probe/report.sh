#!/usr/bin/env bash
# Report of the training-latency probe (scripts/cluster/latency_probe/launch.sh).
#   bash scripts/cluster/latency_probe/report.sh            # on VULCAN: collect on TamIA (ssh) + copy back, collect Vulcan, render REPORT.md
#   bash scripts/cluster/latency_probe/report.sh collect    # on the cluster that ran the packs: logs -> collected_<cluster>/ + stats
#   bash scripts/cluster/latency_probe/report.sh render     # REPORT.md from every collected_*/stats.tsv present
# Per run: nnU-Net's own per-epoch "Epoch time: X s" lines from fold_0/training_log_*.txt; epochs 0-4 (warm-up, torch.compile,
# data-loader start) are skipped and the median + IQR over epochs 5..(PROBE_EPOCHS-1) is reported. Ratios are taken WITHIN a
# node: x / auglab_default of the same pack; vs baseline directly in P2, and for P1 runs chained through the auglab_default
# replicate: (x / AD_P1) * (AD_P2 / baseline_P2). Every number is awk/sort only (no Python on login nodes).
# Context (not part of the comparison): median epoch time of the production DualVal headline run, fold 0, when its log is on
# this cluster's scratch (production packs ran 2-3 folds per GPU, so its absolute time is not comparable).
set -uo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
REPO="$(pwd)"
if [ -d "/scratch/${USER:0:1}/${USER}" ]; then CLUSTER=tamia; export SCRATCH="/scratch/${USER:0:1}/${USER}"
else CLUSTER=vulcan; export SCRATCH="${SCRATCH:-/scratch/${USER}}"; fi
LAT="${LAT_DIR:-${SCRATCH}/review_20261015/latency}"
EPOCHS="${PROBE_EPOCHS:-25}"; SKIP="${SKIP_EPOCHS:-5}"
MODE="${1:-all}"

# quantiles of a list of numbers on stdin: "n median q1 q3" (linear interpolation, R type 7)
quant() { sort -g | awk '{a[NR]=$1} END {n=NR; if (n==0) {print 0, "NA", "NA", "NA"; exit}
    q(0.5); m=r; q(0.25); l=r; q(0.75); h=r; printf "%d %.2f %.2f %.2f\n", n, m, l, h}
    function q(p,  x, i, f) { x=1+(n-1)*p; i=int(x); f=x-i; r=(i<n)? a[i]+f*(a[i+1]-a[i]) : a[n] }'; }
epoch_times() { awk '/Epoch time:/ {for (i=1;i<=NF;i++) if ($i=="time:") print $(i+1)}' "$@"; }
# "<timestamp> <epoch time>" per epoch (timestamp = when nnU-Net logged the epoch end; sortable as a string)
epoch_ts() { awk '/Epoch time:/ {t=$2; sub(/:$/,"",t); for (i=1;i<=NF;i++) if ($i=="time:") print $1 "_" t, $(i+1)}' "$@"; }

collect() {
    local MAN="${LAT}/manifest_${CLUSTER}.tsv" OUT="${LAT}/collected_${CLUSTER}"
    [ -f "${MAN}" ] || { echo "[collect] no ${MAN} on ${CLUSTER}"; return 0; }
    mkdir -p "${OUT}/logs" "${OUT}/packs"
    printf 'cluster\ttask\tcontrast\tpack\tgpu\tslot\ttrainer\tcfg\tda\tcompile\titers\tval_iters\tgpu_name\thost\tjob\tepochs_logged\tn_timed\tmedian_s\tq1_s\tq3_s\tstatus\tn_full\tmedian_full_s\n' > "${OUT}/stats.tsv"
    # contention window: the 4 runs of a pack share the node's CPUs only until the FIRST one finishes; per pack, that
    # finisher's last epoch timestamp is the cutoff for the "all 4 running" statistics (n_full, median_full_s)
    declare -A CUT=()
    local _c _t _tc _p _g _s _x _res _l _e
    while IFS=$'\t' read -r _c _t _tc _p _g _s _x _x _x _x _res _x; do
        [ "$_c" = cluster ] && continue
        _l="$(ls -t "${_res}"/Dataset*/*/fold_0/training_log_*.txt 2>/dev/null | head -1)"; [ -n "$_l" ] || _l="${OUT}/logs/${_t}_${_p}_${_s}.txt"
        [ -f "$_l" ] || continue
        [ "$(epoch_times "$_l" | grep -c .)" -ge "${EPOCHS}" ] || continue
        _e="$(epoch_ts "$_l" | tail -1 | cut -d' ' -f1)"
        if [ -z "${CUT[${_t}_${_p}]:-}" ] || [[ "$_e" < "${CUT[${_t}_${_p}]}" ]]; then CUT[${_t}_${_p}]="$_e"; fi
    done < "${MAN}"
    while IFS=$'\t' read -r cl task tc pack gpu slot tr cfg da cp res cmd <&3; do
        [ "$cl" = cluster ] && continue
        local key="${task}_${pack}_${slot}" pdir="${cmd%/*}" log dbg it vit gname host job st pout
        log="$(ls -t "${res}"/Dataset*/*/fold_0/training_log_*.txt 2>/dev/null | head -1)"
        [ -n "$log" ] && cp "$log" "${OUT}/logs/${key}.txt"
        [ -s "${OUT}/logs/${key}.txt" ] && log="${OUT}/logs/${key}.txt"   # survives cleanup of the throwaway results
        [ -n "$log" ] || log=/dev/null   # (an empty file argument would make awk read the manifest loop's stdin)
        dbg="$(ls "${res}"/Dataset*/*/fold_0/debug.json 2>/dev/null | head -1)"
        [ -n "$dbg" ] && cp "$dbg" "${OUT}/logs/${key}.debug.json"; dbg="${OUT}/logs/${key}.debug.json"
        it="$(grep -o '"num_iterations_per_epoch": *"*[0-9]*' "$dbg" 2>/dev/null | grep -o '[0-9]*$')"
        vit="$(grep -o '"num_val_iterations_per_epoch": *"*[0-9]*' "$dbg" 2>/dev/null | grep -o '[0-9]*$')"
        pout="$(ls -t "${pdir}"/pack_*.out 2>/dev/null | head -1)"
        if [ -n "$pout" ]; then
            cp "$pout" "${OUT}/packs/${task}_${pack}_${pout##*/}"
            gname="$(grep -E "^${gpu}, " "$pout" | head -1 | cut -d, -f2 | sed 's/^ *//')"
            host="$(grep -o 'host=[^ ]*' "$pout" | head -1 | cut -d= -f2)"; job="$(grep -o 'job=[0-9]*' "$pout" | head -1 | cut -d= -f2)"
            st="$(grep -E "\] (OK|FAIL) +'fold0_[^']*_${slot}_" "$pout" | grep -oE 'OK|FAIL.*' | head -1)"
            [ -z "$st" ] && [ "$slot" = dualval ] && st="$(grep -E "\] (OK|FAIL) +'fold0_[^']*_dualval_val000_" "$pout" | grep -oE 'OK|FAIL.*' | head -1)"
            [ -z "$st" ] && st="running/unfinished"
        else gname=""; host=""; job="$(cat "${pdir}/JOBID" 2>/dev/null)"; st="not started"; fi
        local ne q
        ne="$(epoch_times "$log" 2>/dev/null | grep -c .)"
        q="$(epoch_times "$log" 2>/dev/null | awk -v s="${SKIP}" -v e="${EPOCHS}" 'NR>s && NR<=e' | quant)"
        local cut="${CUT[${task}_${pack}]:-9999}" qf
        qf="$(epoch_ts "$log" 2>/dev/null | awk -v s="${SKIP}" -v e="${EPOCHS}" -v c="$cut" 'NR>s && NR<=e && $1<=c {print $2}' | quant | cut -d' ' -f1,2)"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cl" "$task" "$tc" "${pack##*_}" "$gpu" "$slot" "$tr" "$cfg" "$da" "$cp" \
            "${it:-?}" "${vit:-?}" "${gname:-?}" "${host:-?}" "${job:-?}" "${ne:-0}" $q "${st// /_}" $qf >> "${OUT}/stats.tsv"
    done 3< "${MAN}"
    # context: the production DualVal headline run (fold 0), when on this cluster
    printf 'task\trun\tlogs\tepochs\tmedian_s_ep_ge5\n' > "${OUT}/context_dualval.tsv"
    local ctx
    if [ "${CLUSTER}" = tamia ]; then ctx=(
        "onharmony|${SCRATCH}/on-harmony/8_results/01_predictions/on_harmony_model/T1w/auglab|nnUNetTrainerOnHarmonyAugLabDualVal"
        "ispy2|${SCRATCH}/ispy2/8_results/01_predictions/ispy2_model/t1wce/auglab|nnUNetTrainerISPY2AugLabDualVal"
        "toothfairy2|${SCRATCH}/toothfairy2/8_results/01_predictions/toothfairy2_model/cbct/auglab|nnUNetTrainerToothFairy2AugLabDualVal"
        "pelvic|${SCRATCH}/totalseg-pelvic/8_results_totalseg-pelvic/01_predictions/totalseg_pelvic_model/ct/auglab|nnUNetTrainerTotalsegPelvicAugLabDualVal"
        "atlas|${SCRATCH}/atlas-liver-hcc/8_results/01_predictions/atlas_liver_hcc_model/t1w/auglab|nnUNetTrainerAtlasLiverHCCAugLabDualVal" )
    else ctx=( "pansegdata|${REPO}/benchmark/02_tasks/pancreas_disease/pansegdata/8_results_pansegdata/01_predictions/pansegdata_model/t1wce/auglab|nnUNetTrainerPANSEGDATAAugLabDualVal" ); fi
    local c t base trn d logs
    for c in "${ctx[@]}"; do IFS='|' read -r t base trn <<<"$c"
        for d in "${base}"/*_auglabAug_v26_6_2_train050_val000_2*; do
            [ -d "$d" ] || continue
            logs=( "$d"/Dataset*/"${trn}"__nnUNetPlans__3d_fullres/fold_0/training_log_*.txt ); [ -f "${logs[0]}" ] || continue
            printf '%s\t%s\t%s\t%s\t%s\n' "$t" "${d##*/}" "${#logs[@]}" "$(epoch_times "${logs[@]}" | grep -c .)" \
                "$(epoch_times "${logs[@]}" | awk 'NR>5' | quant | cut -d' ' -f2)" >> "${OUT}/context_dualval.tsv"
        done
    done
    echo "[collect] ${CLUSTER}: $(($(grep -c . "${OUT}/stats.tsv")-1)) runs -> ${OUT}/stats.tsv"
}

render() {
    local F=( "${LAT}"/collected_*/stats.tsv ); [ -f "${F[0]}" ] || { echo "[render] nothing collected"; return 1; }
    local R="${LAT}/REPORT.md"
    {
    echo "# Training latency probe (PALETTE Augmentation vs competitors)"
    echo
    echo "Generated by \`bash scripts/cluster/latency_probe/report.sh\` on $(date '+%Y-%m-%d %H:%M'). Method: fold 0 of each task's primary"
    echo "training contrast, the dataset's own headline train wrappers (configs, trainers, iterations/epoch exactly the paper's), ${EPOCHS} epochs,"
    echo "4 runs per whole 4-GPU node, one run per GPU (CPU data loading shared by the 4 runs: 48 CPUs per node). Median and IQR of nnU-Net's"
    echo "per-epoch wall time over epochs ${SKIP}-$((EPOCHS-1)) (n = epochs timed). Ratios only within a node: x / auglab_default of the same"
    echo "pack; x / baseline directly in P2 and chained through the P2 auglab_default replicate for P1 runs. Probe-only changes:"
    echo "checkpointing off, final validation pass skipped, WandB off (none of them runs inside an epoch)."
    echo
    echo "- **PA val000** = headline OURS config (\`..._train050.json\`) with the plain single-validation trainer (DualVal's second,"
    echo "  synthetic validation pass removed), DA workers / compile flag as in the headline wrapper."
    echo "- **PA val000, matched** = the same with \`nnUNet_n_proc_DA\` and \`nnUNet_compile\` set to auglab_default's values: the fair"
    echo "  ratio for the augmentation cost where the headline OURS wrapper ran DA_WORKERS=0 / compile=0 and the competitors 8 / 1."
    echo "- **DualVal (context)** = the headline DualVal trainer as run for the paper (validates twice per epoch)."
    echo
    echo "| task | contrast | cluster / GPU | method | DA | compile | median s/epoch | IQR | x auglab_default | x baseline | n | median, all 4 running (n) | status |"
    echo "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    cat "${F[@]}" | awk -F'\t' '
      $1=="cluster" {next}
      { k=$2; key=k SUBSEP $4 SUBSEP $6; med[key]=$18; row[++n]=$0; if (!(k in seen)) {seen[k]=1; order[++nt]=k} }
      END {
        lab["ad"]="auglab_default"; lab["sem"]="synthseg_EM"; lab["srcsm"]="SRCSM"; lab["pa"]="**PA val000**";
        lab["baseline"]="baseline (nnU-Net DA)"; lab["ad_rep"]="auglab_default (replicate)"; lab["pa_matched"]="**PA val000, matched DA/compile**";
        lab["pa_rep"]="PA val000 (replicate)"; lab["dualval"]="PA DualVal trainer (context)";
        so["P1","ad"]=1; so["P1","sem"]=2; so["P1","srcsm"]=3; so["P1","pa"]=4; so["P2","baseline"]=5; so["P2","ad_rep"]=6; so["P2","pa_matched"]=7; so["P2","pa_rep"]=8; so["P2","dualval"]=8;
        for (t=1;t<=nt;t++) { k=order[t]
          for (s=1;s<=8;s++) for (i=1;i<=n;i++) { split(row[i], f, "\t"); if (f[2]!=k || so[f[4],f[6]]!=s) continue
            m=f[18]; ad1=med[k,"P1","ad"]; ad2=med[k,"P2","ad_rep"]; bl=med[k,"P2","baseline"]
            ra="NA"; rb="NA"
            if (m!="NA") {
              if (f[4]=="P1") { if (ad1!="NA" && ad1>0) ra=sprintf("%.2f", m/ad1); if (ad1!="NA" && ad1>0 && ad2!="NA" && bl!="NA" && bl>0) rb=sprintf("%.2f*", m/ad1*ad2/bl) }
              else { if (ad2!="NA" && ad2>0) ra=sprintf("%.2f", m/ad2); if (bl!="NA" && bl>0) rb=sprintf("%.2f", m/bl) } }
            gn=f[13]; sub(/NVIDIA /,"",gn); sub(/ 80GB HBM3/,"",gn)
            iqr=(f[19]=="NA")? "NA" : f[19] "-" f[20]
            printf "| %s | %s | %s %s (%s) | %s | %s | %s | %s | %s | %s | %s | %s | %s (%s) | %s |\n", k, f[3], f[1], gn, f[4], lab[f[6]], f[9], f[10], m, iqr, ra, rb, f[17], f[23], f[22], f[21] } } }'
    echo
    echo "\\* chained through the auglab_default replicate of P2 (P1 runs and the baseline sit on different nodes)."
    echo "\"all 4 running\" = the same statistic restricted to epochs that ended before the first run of the pack finished (after that,"
    echo "the slower runs share the node's CPUs with fewer neighbours); NA = no timed epoch inside that window."
    echo
    echo "## Summary per task (ratios of median s/epoch; PA = PA val000)"
    echo
    echo "PA matched = DA workers / compile flag of auglab_default (the augmentation cost alone). Competitor ratios for the P1 methods"
    echo "(SRCSM, synthseg_EM) are taken through auglab_default: (PA_matched / AD_P2) / (X / AD_P1). Last row: geometric mean over tasks."
    echo
    echo "| task | cluster / GPU | PA matched / auglab_default | PA matched / baseline | PA matched / SRCSM | PA matched / synthseg_EM | PA as run (headline DA/compile) / auglab_default | DualVal / PA matched | auglab_default P2 / P1 |"
    echo "|---|---|---|---|---|---|---|---|---|"
    cat "${F[@]}" | awk -F'\t' '$1!="cluster" { k=$2; m[k,$4,$6]=$18; g[k]=$1 " " $13; if (!(k in s)) {s[k]=1; o[++n]=k} }
      function ok(x) { return (x!="" && x!="NA" && x>0) }
      function r(a,b) { return (ok(a) && ok(b)) ? a/b : "NA" }
      function fmt(x) { return (x=="NA") ? "NA" : sprintf("%.2f", x) }
      function acc(c, x) { if (x!="NA") { L[c]+=log(x); N[c]++ } }
      END { for (i=1;i<=n;i++) { k=o[i]; gn=g[k]; sub(/NVIDIA /,"",gn); sub(/ 80GB HBM3/,"",gn)
          pm=r(m[k,"P2","pa_matched"], m[k,"P2","ad_rep"])
          v[1]=pm; v[2]=r(m[k,"P2","pa_matched"], m[k,"P2","baseline"])
          a=r(m[k,"P1","srcsm"], m[k,"P1","ad"]); v[3]=(pm!="NA" && a!="NA") ? pm/a : "NA"
          a=r(m[k,"P1","sem"], m[k,"P1","ad"]); v[4]=(pm!="NA" && a!="NA") ? pm/a : "NA"
          v[5]=r(m[k,"P1","pa"], m[k,"P1","ad"]); v[6]=r(m[k,"P2","dualval"], m[k,"P2","pa_matched"]); v[7]=r(m[k,"P2","ad_rep"], m[k,"P1","ad"])
          line="| " k " | " gn; for (c=1;c<=7;c++) { line=line " | " fmt(v[c]); if (c<=5) acc(c, v[c]) }; print line " |" }
        line="| **geometric mean** | (" n " tasks) "; for (c=1;c<=7;c++) line=line " | " ((c<=5 && N[c]) ? sprintf("%.2f (n=%d)", exp(L[c]/N[c]), N[c]) : ""); print line " |" }'
    echo
    echo "## Context: production DualVal headline runs, fold 0 (packed 2-3 folds per GPU; absolute times not comparable)"
    echo
    echo "| task | cluster | run | log files | epochs logged | median s/epoch (epochs >= 5) |"
    echo "|---|---|---|---|---|---|"
    local c; for c in "${LAT}"/collected_*/context_dualval.tsv; do [ -f "$c" ] || continue; local cl="${c%/context_dualval.tsv}"; cl="${cl##*collected_}"
        awk -F'\t' -v cl="$cl" 'NR>1 {printf "| %s | %s | %s | %s | %s | %s |\n", $1, cl, $2, $3, $4, $5}' "$c"; done
    if [ -f "${LAT}/INTERPRETATION.md" ]; then echo; cat "${LAT}/INTERPRETATION.md"; fi
    } > "${R}"
    echo "[render] wrote ${R}"
}

case "${MODE}" in
    collect) collect;;
    render) render;;
    all)
        [ "${CLUSTER}" = vulcan ] || { echo "'all' runs on Vulcan (it reaches TamIA by ssh); here use collect"; exit 2; }
        T="/scratch/p/${USER}/review_20261015/latency"
        tar cf - -C "${REPO}" scripts/cluster/latency_probe | ssh tamia.alliancecan.ca \
            "cd /project/aip-jcohen/paulh/mri_synthesis_project && tar xf - && bash -lc 'cd /project/aip-jcohen/paulh/mri_synthesis_project && bash scripts/cluster/latency_probe/report.sh collect'" \
            || echo "[all] TamIA collect failed (unreachable?) -- rendering with what is local"
        ssh tamia.alliancecan.ca "cd ${T} && tar cf - collected_tamia manifest_tamia.tsv packs/*/record_*.log packs/*/*.orig 2>/dev/null" | tar xf - -C "${LAT}" \
            || echo "[all] copy-back from TamIA failed"
        collect; render;;
    *) echo "usage: report.sh [all|collect|render]"; exit 2;;
esac
