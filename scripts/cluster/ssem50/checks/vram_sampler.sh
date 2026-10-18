# Sample GPU memory of the running probe job every 60 s via an overlapping step (one run per GPU -> per-GPU max = per-run peak).
J=$1; OUT=/scratch/p/paulh/review_20261018/ssem50/probe_vram_samples.csv
until squeue -h -j $J -t R | grep -q .; do squeue -h -j $J | grep -q . || exit 0; sleep 60; done
while squeue -h -j $J -t R | grep -q .; do
  srun --jobid=$J --overlap --ntasks=1 nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sed "s/^/$(date +%H:%M:%S),/" >> $OUT
  sleep 60; done
