cd /project/aip-jcohen/paulh/mri_synthesis_project/sub-workspaces/auglab_workspace/AugLab
git ls-files | grep -E '\.py$|configs/transform_params_gpu_default01-23_(Synthseg_EM|auglabAug_Synthseg_EM|auglabAug_Synthseg_EM_train050|auglabAug_ImageContrastV26_6_2GPUTransform_train050)\.json$' | LC_ALL=C sort | xargs md5sum
echo "## modified/untracked: $(git status --porcelain | wc -l)  HEAD $(git rev-parse --short HEAD)"
cd /project/aip-jcohen/paulh/mri_synthesis_project
for p in .venv/lib/python3.*/site-packages/*.pth; do grep -l -i 'auglab\|sub-workspaces' "$p" >/dev/null 2>&1 && echo "## pth $(basename $p): $(cat $p | head -3 | tr '\n' ' ')"; done
ls .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/ | grep -i 'DAExt\|Trainers.py' | while read f; do echo "## venv trainer $f $(md5sum < .venv/lib/python3.*/site-packages/nnunetv2/training/nnUNetTrainer/$f | cut -c1-32)"; done
