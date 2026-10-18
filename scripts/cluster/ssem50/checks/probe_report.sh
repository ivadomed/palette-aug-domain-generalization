# Probe report: per run median s/epoch over epochs 5-24, last epoch, peak GPU memory (nvidia-smi samples, 1 run per GPU), errors, [loader] line.
R=$(cat /scratch/p/paulh/_packruns_ssem50_root_probe.txt)/P_probe; V=/scratch/p/paulh/review_20261018/ssem50/probe_vram_samples.csv
read -ra G < $R/GPU_MAP; i=0
while IFS=$'\t' read -r cmd log name done_; do
  plog="${cmd%.sh}.log"; res=$(sed -e 's/\\n/\n/g' "$cmd" | sed -n "s/^ *export nnUNet_results=//p" | head -1 | tr -d "\\\\'")
  L=$(ls -t $res/Dataset*/*/fold_0/training_log_*.txt 2>/dev/null | head -1)
  med=$(awk '/Epoch [0-9]+ *$/{e=$NF} /Epoch time:/{if(e>=5&&e<=24) print $(NF-1)}' "$L" | sort -n | awk '{a[NR]=$1} END{if(NR) printf "%.1f (n=%d)", a[int((NR+1)/2)], NR; else print "NA"}')
  last=$(grep -oE "Epoch [0-9]+ *$" "$L" | tail -1)
  peak=$(awk -F, -v g=${G[$i]} '$2+0==g && $3+0>m{m=$3+0} END{printf "%.1f GiB", m/1024}' $V)
  util=$(awk -F, -v g=${G[$i]} '$2+0==g{s+=$4;n++} END{if(n) printf "%.0f%%", s/n}' $V)
  err=$(grep -cE "Traceback|Error|error:|Killed|OOM|out of memory" "$plog" 2>/dev/null)
  loader=$(grep -m1 "^\[loader\]" "$plog")
  printf '%s | GPU%s | med s/ep (ep5-24) %s | last %s | peak VRAM %s | mean util %s | error lines %s\n   %s\n' "${name#fold0_}" "${G[$i]}" "$med" "$last" "$peak" "$util" "$err" "$loader"
  i=$((i+1)); done < $R/index.tsv
