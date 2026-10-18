#!/usr/bin/env bash
# Simulated roster resolution (roster_lib.sh, unchanged): per setting, a throwaway PREDICTIONS_ROOT holding EMPTY dirs named like
# every existing auglab run dir of that setting on Vulcan + one fake new ssem50 run; roster_resolve must return the fake run for
# the new METHOD and the combined config's run (or a newer synthseg_EM run) for the old METHOD -- never each other.
set -uo pipefail
R=/project/aip-jcohen/paulh/mri_synthesis_project; T=$R/benchmark/02_tasks
W=/scratch/paulh/review_20261018/ssem50/roster_sim; rm -rf "$W"; mkdir -p "$W"
NEW=auglabAug_synthseg_EM_train050_val000; TS=20261018_999999; bad=0
while IFS=$'\t' read -r task ds key tc src items oldrun srctw newtw newpw ep tr ld cl; do
  [ "$task" = task ] && continue; [[ "$src" == *"(own)" ]] || continue
  dsdir=$(dirname $(dirname $(dirname "$R/$newtw"))); dn=$(basename "$dsdir")
  ( source "$dsdir/5_scripts_$dn/00_utils/$(grep -o 'env[a-z0-9_]*\.sh' "$R/$newtw" | head -1)" >/dev/null 2>&1
    real="$PREDICTIONS_ROOT/$MODEL_TYPE/$tc/auglab"
    export PREDICTIONS_ROOT="$W/$key"; base="$PREDICTIONS_ROOT/$MODEL_TYPE/$tc/auglab"; mkdir -p "$base"
    ls "$real" 2>/dev/null | while read d; do mkdir -p "$base/$d"; done
    mkdir -p "$base/${DATASET_NAME}_${tc}_${NEW}_${TS}"
    export TRAINING_CONTRAST="$tc"
    source $R/benchmark/00_commun_scripts/00_00_utils/roster_lib.sh
    roster_parse_wrapper "$R/$newpw"; pm="$W_METHOD"
    n_pred=$(grep -l "^METHOD=\"$pm\"" $(dirname "$R/$newpw")/*.sh | xargs -r grep -l "TRAINING_CONTRAST=\"\?$tc\|env$( [ "$tc" != "${tc,,}" ] && echo '' )" 2>/dev/null | wc -l)
    rn=$(roster_resolve "$NEW" auglab); ro=$(roster_resolve "$(grep -m1 '^METHOD=' "$R/$srctw" | sed 's/METHOD="\(.*\)"/\1/')" auglab)
    ok=OK; [ "$rn" = "${DATASET_NAME}_${tc}_${NEW}_${TS}" ] || ok=NEW_WRONG; [[ "$ro" == *"$NEW"* ]] && ok="$ok,OLD_RESOLVES_TO_NEW"
    printf '%-13s predict-wrapper METHOD=%-45s new->%-70s old->%s  %s\n' "$key" "$pm" "$rn" "${ro:-<none: run dir not on Vulcan>}" "$ok"
    [ "$ok" = OK ] ) || bad=1
done < /scratch/paulh/review_20261018/ssem50/roster.tsv
[ "$bad" = 0 ] && echo "ROSTER RESOLUTION OK" || echo "ROSTER RESOLUTION FAILED"
