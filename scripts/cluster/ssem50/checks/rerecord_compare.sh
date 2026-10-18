# Compare every recorded fold command of the REAL record with the dry-run record (staging tree). Normalisation: staging tree path ->
# real checkout path, each root's own TS -> <TS>, and the recorded git stamp '[code] main=<sha>' (none in the staging tree) -> <GIT>
# (checked separately: must be none -> 4d6cfe431fff). Then (a) whole-file diff per fold file, (b) index.tsv / GPU_MAP / CHAIN per pack.
S=/scratch/p/paulh; W=$S/review_20261018/ssem50; T=$W/tree; P=/project/aip-jcohen/paulh/mri_synthesis_project
for kind in main probe; do
  if [ $kind = main ]; then RR=$(cat $S/_packruns_ssem50_root.txt); else RR=$(cat $S/_packruns_ssem50_root_probe.txt); fi
  DR=$(cat $W/dryrun_state_$kind.txt); RT=$(cat $RR/TS); DT=$(cat $DR/TS)
  echo "=== $kind: real $RR (TS $RT)  vs  dry-run $DR (TS $DT)"
  nf=0; nsame=0; ndiff=0
  for pd in $DR/*/; do pk=$(basename $pd)
    for f in GPU_MAP CHAIN; do cmp -s $pd/$f $RR/$pk/$f && echo "  $pk/$f identical" || { echo "  $pk/$f DIFFERS"; ndiff=$((ndiff+1)); }; done
    d1=$(sed "s|$T|$P|g; s|$DR|<ROOT>|g; s|$DT|<TS>|g" $pd/index.tsv); d2=$(sed "s|$RR|<ROOT>|g; s|$RT|<TS>|g" $RR/$pk/index.tsv)
    [ "$d1" = "$d2" ] && echo "  $pk/index.tsv identical after normalisation ($(wc -l < $pd/index.tsv) rows)" || { echo "  $pk/index.tsv DIFFERS"; diff <(echo "$d1") <(echo "$d2") | head -4 | cut -c1-300 | sed 's/^/      /'; ndiff=$((ndiff+1)); }
    for c in $pd/fold*.sh; do nf=$((nf+1)); b=$(basename $c); r=$RR/$pk/${b/$DT/$RT}
      [ -f "$r" ] || { echo "  MISSING in real record: $pk/$b"; ndiff=$((ndiff+1)); continue; }
      x=$(diff <(sed "s|$T|$P|g; s|$DR|<ROOT>|g; s|$DT|<TS>|g; s|\[code\] main=[0-9a-z]*|[code] main=<GIT>|" $c) <(sed "s|$RR|<ROOT>|g; s|$RT|<TS>|g; s|\[code\] main=[0-9a-z]*|[code] main=<GIT>|" $r))
      gd=$(grep -o "\[code\] main=[0-9a-z]*" $c | head -1)/$(grep -o "\[code\] main=[0-9a-z]*" $r | head -1); [ "$gd" = "[code] main=none/[code] main=4d6cfe431fff" ] || echo "  UNEXPECTED code stamp $pk/$b: $gd"
      if [ -z "$x" ]; then nsame=$((nsame+1)); else echo "  DIFF $pk/$b:"; echo "$x" | head -8 | sed 's/^/      /'; ndiff=$((ndiff+1)); fi
    done
    extra=$(comm -13 <(ls $pd | sed "s|$DT|<TS>|g" | sort) <(ls $RR/$pk | sed "s|$RT|<TS>|g" | sort) | grep -v '^$' | tr '\n' ' ')
    [ -n "$extra" ] && echo "  files only in the real record ($pk): $extra"
  done
  echo "  -> $kind: $nf fold files compared, $nsame identical after normalisation, $ndiff differences"
done
