#!/bin/bash
# OPTIONAL Killarney path for the BraTS-SSA predict/eval matrix (run ON VULCAN, from a shell where a
# Killarney ControlMaster is up: Killarney needs Duo, a session cannot open it; Paul runs
# `ssh -fN killarney.alliancecan.ca` first). Steps, each verified:
#   0. Killarney squeue (pending jobs are reported; code is synced bookkeeping-only, files untouched)
#   1. code: git bundle <Killarney HEAD>..main -> fetch -> reset --mixed (skill git-workflow, cluster-sync),
#      then `git checkout --` of THIS directory only (new files, nothing a pending job imports)
#   2. data: the rebuilt BraTS-SSA imagesTs_*/labelsTs_* + the 16 source-matched dirs (symlinks dereferenced),
#      the archived 5_scripts_brats-ssa2024 (untracked, hop counts fixed 2026-10-20), glioma dataset.json x4
#   3. checkpoints: for every runs.txt row with action predict|srcmatch, plans/dataset/fingerprint json +
#      fold_{0,1,2}/checkpoint_best.pth ONLY, then md5 of every .pth compared Vulcan vs Killarney
#   4. venv check: BraTS trainer shim present in Killarney's nnunetv2 + importable trainer classes
#   5. launch: bash launch.sh on Killarney (RUN_JOB_GPU_TYPE=l40s), unless DRY=1
# Afterwards (when the controllers finish): bash killarney_stage.sh fetch  -> rsync predictions+metrics back.
# DO NOT run both this and the Vulcan controllers for the same family (cancel one side first).
set -euo pipefail
V=/project/aip-jcohen/paulh/mri_synthesis_project
K=/home/paulh/projects/aip-jcohen/paulh/mri_synthesis_project
H=killarney.alliancecan.ca
A=benchmark/03_archive/brats-ssa2024
GL=benchmark/02_tasks/brain_tumor/brats2024-glioma
P="${GL}/8_results_brats2024-glioma/01_predictions/brats2024_glioma_model"
LOG="${SCRATCH:?}/review_20261020/brats_ssa/logs/killarney_stage.log"
kssh() { ssh -o BatchMode=yes "${H}" "bash -lc $(printf '%q' "cd ${K} && { $*
}")"; }
say() { echo "[$(date '+%F %T')] $*" | tee -a "${LOG}"; }
ssh -o BatchMode=yes -o ConnectTimeout=20 "${H}" true || { echo "Killarney not reachable non-interactively (Duo). Run: ssh -fN ${H}"; exit 1; }

if [ "${1:-}" = fetch ]; then
  for sub in 01_predictions 02_metrics; do
    rsync -a --info=stats1 "${H}:${K}/${A}/8_results_brats-ssa2024/${sub}/brats2024_glioma_model/" "${V}/${A}/8_results_brats-ssa2024/${sub}/brats2024_glioma_model/" | tee -a "${LOG}"
  done
  say "fetched; re-run the 0-byte/row audit on Vulcan"; exit 0
fi

say "0. Killarney queue"; kssh 'squeue -u $USER -h | wc -l; squeue -u $USER -h -t PD | wc -l' | tee -a "${LOG}"

say "1. code sync (bundle, reset --mixed)"
KHEAD="$(kssh 'git rev-parse HEAD')"; VHEAD="$(git -C "${V}" rev-parse HEAD)"
if [ "${KHEAD}" != "${VHEAD}" ]; then
  git -C "${V}" bundle create "${SCRATCH}/k_bssa.bundle" "${KHEAD}..main"
  scp -q "${SCRATCH}/k_bssa.bundle" "${H}:/scratch/paulh/k_bssa.bundle"
  kssh 'git fetch /scratch/paulh/k_bssa.bundle main:refs/heads/_inc && git reset --mixed _inc && git branch -D _inc'
fi
kssh 'git checkout -- scripts/cluster/brats_ssa_pa_alone && git rev-parse HEAD' | tee -a "${LOG}"

say "2. data"
kssh "mkdir -p ${A}/2_nnUNet_brats-ssa2024/raw ${A}/8_results_brats-ssa2024 ${GL}/2_nnUNet_brats2024-glioma/raw"
rsync -a "${V}/${A}/5_scripts_brats-ssa2024/" "${H}:${K}/${A}/5_scripts_brats-ssa2024/"
rsync -a --copy-links "${V}/${A}/2_nnUNet_brats-ssa2024/raw/" "${H}:${K}/${A}/2_nnUNet_brats-ssa2024/raw/"
for d in 051_BraTS2024GliomaT1n 052_BraTS2024GliomaT2w 053_BraTS2024GliomaT2f 054_BraTS2024GliomaT1c; do
  kssh "mkdir -p ${GL}/2_nnUNet_brats2024-glioma/raw/Dataset${d}"
  scp -q "${V}/${GL}/2_nnUNet_brats2024-glioma/raw/Dataset${d}/dataset.json" "${H}:${K}/${GL}/2_nnUNet_brats2024-glioma/raw/Dataset${d}/"
done
kssh "for d in ${A}/2_nnUNet_brats-ssa2024/raw/*; do echo \$(basename \$d) \$(ls \$d | wc -l); done" | tee -a "${LOG}"

say "3. checkpoints (folds 0 1 2, checkpoint_best only) + md5"
: > "${SCRATCH}/review_20261020/brats_ssa/logs/k_ckpt_list.txt"
command grep -v '^#' "${V}/scripts/cluster/brats_ssa_pa_alone/runs.txt" | while IFS=$'\t' read -r mod key cat tr rid act; do
  [ "${act}" = reuse ] && continue
  td="$(cd "${V}/${P}/${mod}/${cat}/${rid}" && ls -d Dataset*/"${tr}"__nnUNetPlans__3d_fullres)"
  for f in plans.json dataset.json dataset_fingerprint.json fold_0/checkpoint_best.pth fold_1/checkpoint_best.pth fold_2/checkpoint_best.pth; do
    echo "${P}/${mod}/${cat}/${rid}/${td}/${f}"
  done >> "${SCRATCH}/review_20261020/brats_ssa/logs/k_ckpt_list.txt"
done
rsync -a --files-from="${SCRATCH}/review_20261020/brats_ssa/logs/k_ckpt_list.txt" "${V}/" "${H}:${K}/"
command grep '\.pth$' "${SCRATCH}/review_20261020/brats_ssa/logs/k_ckpt_list.txt" > "${SCRATCH}/review_20261020/brats_ssa/logs/k_pth.txt"
(cd "${V}" && xargs md5sum < "${SCRATCH}/review_20261020/brats_ssa/logs/k_pth.txt") | sort -k2 > "${SCRATCH}/review_20261020/brats_ssa/logs/md5_vulcan.txt"
scp -q "${SCRATCH}/review_20261020/brats_ssa/logs/k_pth.txt" "${H}:/scratch/paulh/k_pth.txt"
# md5 of ~60 x 250 MB files: run as a CPU job on Killarney, not on its login node
kssh 'source scripts/job_runner/run_job.sh; run_job --name bssa_md5 --gpus 0 --cpus 2 --mem 4G --time 01:00:00 --log /scratch/paulh/bssa_md5.log --wait -- bash -c "xargs md5sum < /scratch/paulh/k_pth.txt | sort -k2 > /scratch/paulh/md5_killarney.txt"'
scp -q "${H}:/scratch/paulh/md5_killarney.txt" "${SCRATCH}/review_20261020/brats_ssa/logs/md5_killarney.txt"
if diff "${SCRATCH}/review_20261020/brats_ssa/logs/md5_vulcan.txt" "${SCRATCH}/review_20261020/brats_ssa/logs/md5_killarney.txt"; then
  say "md5 OK: $(wc -l < "${SCRATCH}/review_20261020/brats_ssa/logs/md5_vulcan.txt") checkpoints identical"
else say "!! md5 MISMATCH, not launching"; exit 1; fi

say "4. venv"
kssh 'ls .venv/lib/python3*/site-packages/nnunetv2/training/nnUNetTrainer/ | command grep -i brats || echo "!! no BraTS trainer shim in Killarney venv"' | tee -a "${LOG}"

[ "${DRY:-0}" = 1 ] && { say "DRY=1: staged, not launched"; exit 0; }
say "5. launch"
kssh 'export RUN_JOB_GPU_TYPE=l40s BSSA_CTL_TIME=11:00:00; bash scripts/cluster/brats_ssa_pa_alone/launch.sh' | tee -a "${LOG}"
