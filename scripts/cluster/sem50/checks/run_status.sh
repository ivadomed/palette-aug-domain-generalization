# Per recorded fold of the sem50 main root (or $1 = a pack root): last epoch in the nnU-Net training log, the [loader] line the fold
# printed vs the one recorded in its command file, and error lines in the pack fold log. Read-only. Run on TamIA.
R="${1:-$(cat /scratch/p/paulh/_packruns_sem50_root.txt)}"
for idx in "$R"/*/index.tsv; do pk=$(basename "$(dirname "$idx")"); read -ra G < "$(dirname "$idx")/GPU_MAP"; i=0
  while IFS=$'\t' read -r cmd log name done_; do
    plog="${cmd%.sh}.log"; f="${name#fold}"; k="${f%%_*}"
    res=$(sed -e 's/\\n/\n/g' "$cmd" | sed -n "s/^ *export nnUNet_results=//p" | head -1 | tr -d "\\\\'")
    L=$(ls -t "$res"/Dataset*/*/fold_$k/training_log_*.txt 2>/dev/null | head -1)
    ep=$(grep -oE "Epoch [0-9]+ *$" "$L" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
    got=$(grep -m1 "^\[loader\]" "$plog" 2>/dev/null | sed 's/ OMP_NUM_THREADS=[0-9]*//'); want=$(grep -o '\[loader\] nnUNet_n_proc_DA=[^"]*' "$cmd" | head -1 | sed 's/ OMP_NUM_THREADS=[^ ]*//')
    da=$(sed -e 's/\\n/\n/g' "$cmd" | sed -n "s/^ *export nnUNet_n_proc_DA=//p" | head -1); cp_=$(sed -e 's/\\n/\n/g' "$cmd" | sed -n "s/^ *export nnUNet_compile=//p" | head -1 | tr -d "\\\\'")
    lok=$(echo "$got" | grep -q "nnUNet_n_proc_DA=${da} nnUNet_compile=${cp_} " && echo loader_ok || echo "LOADER?(${got:-none})")
    err=$(grep -cE "Traceback|Error|Killed|out of memory" "$plog" 2>/dev/null)
    printf '%-4s GPU%s %-78s epoch=%-5s %s errors=%s\n' "$pk" "${G[$i]}" "${name}" "${ep:-NA}" "$lok" "${err:-0}"; i=$((i+1))
  done < "$idx"; done
