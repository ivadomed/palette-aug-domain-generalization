#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --cpus-per-task=2
#SBATCH --mem=4000M
#SBATCH --time=00:30:00
#SBATCH --job-name=ssem50_size
# READ-ONLY: size of each setting's existing headline synthseg_EM run dir on TamIA (model + in-tree predictions) = what the ssem50 arm will need.
S=/scratch/p/paulh
for r in brats2024-glioma:brats2024-glioma_t1n_synthseg_EM_20260622_044535 brats2024-glioma:brats2024-glioma_t2f_synthseg_EM_20260917_113412 \
  brats2024-glioma:brats2024-glioma_t1c_synthseg_EM_20260921_140000 on-harmony:on-harmony_T1w_synthseg_EM_20260623_192811 on-harmony:on-harmony_dwi_ap_synthseg_EM_20260921_203727 \
  open-ms:open-ms_flair_synthseg_EM_train100_val000_20260706_061243 chaos:chaos_t2spir_synthseg_EM_20260620_112357 toothfairy2:toothfairy2_cbct_synthseg_EM_20260908_013027 \
  ispy2:ispy2_t1wce_synthseg_EM_20260905_163655 totalseg-pelvic:ct_synthseg_EM_20260916_072434 atlas-liver-hcc:atlas-liver-hcc_t1w_synthseg_EM_20260804_062605; do
  ds=${r%%:*}; id=${r#*:}
  find "$S/$ds" -maxdepth 9 -type d -name "$id" -prune 2>/dev/null | while read -r d; do printf '%s\t%s\t%s\t%s\n' "$ds" "$id" "$(du -sk "$d" | cut -f1)" "$d"; done
done
