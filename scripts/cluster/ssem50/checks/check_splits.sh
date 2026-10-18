R=$(cat /scratch/p/paulh/review_20261018/ssem50/dryrun_state_main.txt)
T=/scratch/p/paulh/review_20261018/ssem50/tree; P=/project/aip-jcohen/paulh/mri_synthesis_project
for f in $R/*/fold0_*.sh; do
  s=$(sed -e 's/\\n/\n/g' "$f" | grep -o "SPLITS_DIR=[^ ;]*" | head -1 | cut -d= -f2 | sed "s/[\\\\']//g")
  real="${s/#$T/$P}"
  printf '%s | %s | exists in real checkout/scratch: %s | test_cases.json: %s\n' "$(basename "$f" | cut -c1-36)" "$real" "$([ -e "$real" ] && echo yes || echo NO)" "$(ls "$real"/test_cases.json "$real"/*.json 2>/dev/null | head -2 | xargs -r -n1 basename | tr '\n' ' ')"
done | sort -t'|' -k2,2 -u
