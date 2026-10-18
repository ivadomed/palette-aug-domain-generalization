#!/bin/bash
# Vulcan L40S PREDICT job of ONE ssem50 companion fold command, submitted by scripts/cluster/ssem50/companions_vulcan.sh. The fold command
# was RECORDED (RUN_JOB_PACK_DIR) from the companion's own predict wrapper / shim, so paths, items, trainer, checkpoint and flags are
# exactly the standard ones. predict_common's command chains the per-item nnUNetv2_predict calls without `set -e`, so the job's exit code
# alone proves nothing: after it ran, every (-i, -o) pair is AUDITED (#predictions == #input cases) and the job fails on any mismatch
# (the eval job is afterok on these jobs).
#   sbatch --gres=gpu:l40s:1 ... vulcan_companion_predict_job.sh <recorded fold command .sh>
#SBATCH --account=aip-jcohen
set -uo pipefail
CMD="${1:?recorded fold command}"
cd /project/aip-jcohen/paulh/mri_synthesis_project
echo "[ssem50c-predict] host=$(hostname) job=${SLURM_JOB_ID:-?} gpu=${CUDA_VISIBLE_DEVICES:-?} cmd=${CMD}"
bash "${CMD}"; rc=$?
python3 -I - "${CMD}" <<'PY'; arc=$?
import glob, os, re, sys
s = open(sys.argv[1]).read().replace("\\'", "'")
pairs = re.findall(r"-i '([^']+)' -o '([^']+)'", s)
bad = 0 if pairs else 1
for i, o in pairs:
    ni = len(glob.glob(os.path.join(i, "*_0000.nii.gz"))); no = len(glob.glob(os.path.join(o, "*.nii.gz")))
    ok = ni > 0 and ni == no; bad |= not ok
    print(f"[ssem50c-predict] {'ok  ' if ok else 'FAIL'} {no}/{ni} predictions  {o}")
print("[ssem50c-predict] AUDIT", "OK" if not bad else "FAILED"); sys.exit(bad)
PY
echo "[ssem50c-predict] done rc=${rc} audit=${arc}"
[ "${rc}" = 0 ] && [ "${arc}" = 0 ]
