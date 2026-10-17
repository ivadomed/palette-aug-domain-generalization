#!/usr/bin/env python
"""NGF texture preservation of the PA round-2 variant configs, with the paper's NGF protocol unchanged.

Loads paper/generated_results/ngf_all_methods/scripts/ngf_all_methods.py (read only, the script behind the paper's NGF table),
appends new method keys at the END of its METHODS registry (so every existing key keeps its index, hence its per-draw seeds) and runs its main().
For each config two keys: <name> = the training config as used (PA transform at its training probability), <name>_p1 = same config with
the PA transform forced to probability 1 (every draw synthesized; same mechanism as the paper's forced `flatfill` key).
usage: python ngf_variants.py --key chaos/t1in --n-scans 20 --n-draws 5 --variants noblur --methods ctrl,ctrl_p1,... --out X.csv
"""
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "paper/generated_results/ngf_all_methods/scripts/ngf_all_methods.py"
os.environ.setdefault("NGF_REPO", str(REPO))
spec = importlib.util.spec_from_file_location("ngf_all_methods", SRC)
m = importlib.util.module_from_spec(spec)
sys.modules["ngf_all_methods"] = m
spec.loader.exec_module(m)

PA = "ImageContrastV26_6_2GPUTransform"
B = "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050"
NEW = {"ctrl": B + ".json", "p075": "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train075.json",
       "edge2x": B + "_edge2x.json", "alphawide": B + "_alphawide.json"}
for k, f in NEW.items():
    m.METHODS[k] = (f, None)
for k, f in NEW.items():
    m.METHODS[k + "_p1"] = (f, "p1:" + PA)
ALL = ",".join(list(NEW) + [k + "_p1" for k in NEW])

if __name__ == "__main__":
    if "--methods" not in sys.argv:
        sys.argv += ["--methods", ALL]
    m.main()
