#!/usr/bin/env python
"""Round-3 NGF: one ARM (env/arg --arm) per process, same per-draw seeds for every arm.
The paper's script seeds a draw with the method's index in METHODS, so here METHODS is rebuilt per process as
{"as_used": cfg, "p1": cfg with PA forced to p=1} (indices 0/1 for EVERY arm, ctrl included): arm differences are not seed noise.
usage: python ngf_variants3.py <arm> --key K ... (rest = ngf_all_methods.py args)"""
import importlib.util, os, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
os.environ.setdefault("NGF_REPO", str(REPO))
spec = importlib.util.spec_from_file_location("ngf_all_methods", REPO / "paper/generated_results/ngf_all_methods/scripts/ngf_all_methods.py")
m = importlib.util.module_from_spec(spec); sys.modules["ngf_all_methods"] = m; spec.loader.exec_module(m)
B = "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050"
FILES = {"ctrl": B + ".json", "unsharp_off": B + "_unsharpoff.json", "redist_off": B + "_redistoff.json",
         "edgehalf": B + "_edgehalf.json", "texhalf": B + "_texhalf.json",
         "structoff": B + "_structoff.json", "structoff_redistoff": B + "_structoff_redistoff.json"}
arm = sys.argv.pop(1)
m.METHODS.clear()
m.METHODS["as_used"] = (FILES[arm], None)
m.METHODS["p1"] = (FILES[arm], "p1:ImageContrastV26_6_2GPUTransform")
sys.argv += ["--methods", "as_used,p1"]
m.main()
