# Compare the re-recorded B packs (with pansegdata) against the earlier B recording of the same root (same TS): every earlier fold file must
# be byte-identical except the git stamp line ([code] main=<sha>); the only new files may be the pansegdata ones. Read-only. Run on TamIA.
R=$(cat /scratch/p/paulh/_packruns_ssem50_root.txt); A=${R}_Bpacks_pre_panseg
for p in B1 B2 B3; do
  for f in $A/$p/fold*.sh; do b=$(basename $f)
    x=$(diff <(sed 's/\[code\] main=[0-9a-z]*/[code] main=<GIT>/' $f) <(sed 's/\[code\] main=[0-9a-z]*/[code] main=<GIT>/' $R/$p/$b 2>/dev/null) && echo same || echo DIFF)
    echo "$p $b $x"; done
  echo "$p new files: $(comm -13 <(ls $A/$p | sort) <(ls $R/$p | sort) | tr '\n' ' ')"
  echo "$p GPU_MAP old '$(cat $A/$p/GPU_MAP)' new '$(cat $R/$p/GPU_MAP)'"
  diff <(cut -f1 $A/$p/index.tsv | sed "s|$A|ROOT|") <(cut -f1 $R/$p/index.tsv | sed "s|$R|ROOT|") | sed 's/^/   index: /'
done
echo "git stamps: old $(grep -ho '\[code\] main=[0-9a-z]*' $A/B1/fold*.sh | sort -u) new $(grep -ho '\[code\] main=[0-9a-z]*' $R/B1/fold*.sh | sort -u)"
