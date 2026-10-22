#!/bin/bash
#SBATCH --account=aip-jcohen
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --job-name=pa_em_smoke
# CPU smoke test of the pa_em AugLab config in THIS cluster's venv (pa_em, 2026-10-22): the real pipeline builder (AugTransformsGPU)
# instantiates RandomV26_6_2ContrastGPU with partition=em from the EM config and kmeans from the headline PA config, and one forward of
# the EM transform (p=1) on a synthetic 3-label volume gives finite output. Run before launching on a cluster whose venv was not tested.
#   sbatch --output=<dir>/em_smoke_%j.out scripts/cluster/pa_em/checks/em_smoke.sh
set -euo pipefail
REPO=/project/aip-jcohen/paulh/mri_synthesis_project; cd "${REPO}"
echo "[em_smoke] host=$(hostname) repo $(git rev-parse --short HEAD) AugLab $(git -C sub-workspaces/auglab_workspace/AugLab rev-parse --short HEAD)"
"${REPO}/.venv/bin/python" - <<'PY'
import random, torch, auglab
from auglab.transforms.gpu import fromSeg
from auglab.transforms.gpu.transforms import AugTransformsGPU
C = "/project/aip-jcohen/paulh/mri_synthesis_project/sub-workspaces/auglab_workspace/AugLab/auglab/configs/"
print("auglab imported from", auglab.__file__)
ok = True
for name, want in [("transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json", "kmeans"),
                   ("transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2EM_GPUTransform_train050.json", "em")]:
    got = [m.partition for m in AugTransformsGPU(C + name).modules() if isinstance(m, fromSeg.RandomV26_6_2ContrastGPU)]
    print(name, "->", got); ok &= got == [want]
D = 40
z, y, x = torch.meshgrid(*[torch.linspace(-1, 1, D)] * 3, indexing="ij"); r = (z**2 + y**2 + x**2).sqrt()
img = torch.where(r < 0.8, 0.4, 0.0) + torch.where(r < 0.5, 0.3, 0.0) + torch.where(r < 0.2, 0.2, 0.0) + 0.05 * torch.randn(D, D, D)
lab = (r < 0.5).long() + (r < 0.2).long()
x5 = torch.stack([img, img.flip(0)]).unsqueeze(1).float(); l5 = torch.stack([lab, lab.flip(0)]).unsqueeze(1)
torch.manual_seed(0); random.seed(0)
out = fromSeg.RandomV26_6_2ContrastGPU(p=1.0, partition="em", em_n_iters=20).apply_transform(x5, {"seg": l5}, {})
fin = bool(torch.isfinite(out).all()); chg = float((out - x5).abs().max())
print(f"EM forward: shape {tuple(out.shape)} finite {fin} max|out-in| {chg:.3f}"); ok &= fin and chg > 0
print("EM SMOKE OK" if ok else "EM SMOKE FAILED")
raise SystemExit(0 if ok else 1)
PY
