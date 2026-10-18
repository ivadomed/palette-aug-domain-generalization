R=$(cat /scratch/p/paulh/review_20261018/ssem50/dryrun_state_${1}.txt)
printf 'pack\tgpu\tfold_file\ttrainer\tdataset\tfold\tepochs\tDA\tcompile\tconfig\tval_cfg\tseed_off\tcd\tresults\n'
for p in $R/*/; do pk=$(basename $p); read -ra G < $p/GPU_MAP; i=0
 while IFS=$'\t' read -r cmd log name done_; do c=$(sed -e 's/\\n/\n/g' -e "s/\\\\'//g" "$cmd")
  g(){ printf '%s\n' "$c" | sed -n "s/^ *export $1=//p" | head -1 | tr -d "'"; }
  tr=$(printf '%s\n' "$c" | grep -oE '\-tr [A-Za-z0-9_]+' | head -1 | cut -c5-); dl=$(printf '%s\n' "$c" | grep -oE 'nnUNetv2_train [0-9]+ 3d_fullres [0-9]+' | head -1)
  cdl=$(printf '%s\n' "$c" | grep -oE '^ *cd [^ ;&]+' | head -1 | awk '{print $2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' $pk ${G[$i]} $(basename $cmd .sh) "$tr" "$(echo $dl|awk '{print $2}')" "$(echo $dl|awk '{print $4}')" "$(g NNUNET_NUM_EPOCHS)" "$(g nnUNet_n_proc_DA)" "$(g nnUNet_compile)" "$(basename "$(g AUGLAB_PARAMS_GPU_JSON)")" "$(g AUGLAB_VAL_PARAMS_GPU_JSON)" "$(g NNUNET_SEED_OFFSET)" "$cdl" "$(g nnUNet_results)"
  i=$((i+1)); done < $p/index.tsv; done
