#!/usr/bin/env python
"""Generate the train/predict wrappers of the PA variants round 2 (2026-10-16) from each development setting's headline PA wrapper.

Each arm = the setting's headline PALETTE-Aug (PA) training with ONLY the train config swapped (the control keeps it), the plain
val000 trainer of the setting (on-harmony's headline was DualVal -> nnUNetTrainerOnHarmonyAugLabDefault, no synthetic-validation
config), and everything else (epochs, DA workers / compile = the dataset's loader defaults, env) as in the headline wrapper.
Development settings only: on-harmony T1w, open-ms flair, chaos t2spir (glioma / mandible / HCC are held out).
Arms (AugLab palette branch, configs in auglab/configs/):
  ctrl      headline config unchanged, retrained on TamIA                METHOD auglabAugCtrl_v26_6_2_train050_val000
  p075      PALETTE probability 0.5 -> 0.75                              METHOD auglabAugP075_v26_6_2_train075_val000
  edge2x    Scharr 0.25 -> 0.50, RandomConv 0.20 -> 0.40                 METHOD auglabAugEdge2x_v26_6_2_train050_val000
  alphawide PALETTE alpha_magnitude_range [0.5, 2.0] -> [0.25, 3.0]      METHOD auglabAugAlphaWide_v26_6_2_train050_val000
Round 3 (2026-10-17, gentler Auglab weakening; ctrl = round 2's, not regenerated; AugLab d623eef):
  unsharpoff UnsharpMaskTransform 0.25 -> 0                              METHOD auglabAugUnsharpOff_v26_6_2_train050_val000
  redistoff  RedistributeSegTransform 0.4 -> 0                           METHOD auglabAugRedistOff_v26_6_2_train050_val000
  edgehalf   ScharrTransform 0.25 -> 0.125, RandomConvTransform 0.20 -> 0.10   METHOD auglabAugEdgeHalf_v26_6_2_train050_val000
  texhalf    Scharr, RandomConv, UnsharpMask halved (0.125/0.10/0.125)   METHOD auglabAugTexHalf_v26_6_2_train050_val000
  (design: /scratch/paulh/review_20261017/weaken/DESIGN.md on Vulcan)
No METHOD contains REF_SUBSTR (auglabAug_v26_6_2_train050_val000), so the headline run stays the default `sig. vs ref`; the round's
own tables set sig_ref explicitly. Design: /scratch/paulh/review_20261016/new_variants/DESIGN.md (Vulcan).
  python scripts/cluster/pa_variants/make_wrappers.py [--round 2|3] [--check]   (default round 2; --check without --round checks both)
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OLDCFG = "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
CFG = "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_{}.json"
# arm -> (config, METHOD, one-line description)
ARMS = {
    "ctrl": (OLDCFG, "auglabAugCtrl_v26_6_2_train050_val000", "control: the headline config UNCHANGED"),
    "p075": (CFG.format("train075"), "auglabAugP075_v26_6_2_train075_val000", "PALETTE probability 0.5 -> 0.75"),
    "edge2x": (CFG.format("train050_edge2x"), "auglabAugEdge2x_v26_6_2_train050_val000",
               "ScharrTransform 0.25 -> 0.50 and RandomConvTransform 0.20 -> 0.40"),
    "alphawide": (CFG.format("train050_alphawide"), "auglabAugAlphaWide_v26_6_2_train050_val000",
                  "PALETTE alpha_magnitude_range [0.5, 2.0] -> [0.25, 3.0]"),
    "unsharpoff": (CFG.format("train050_unsharpoff"), "auglabAugUnsharpOff_v26_6_2_train050_val000",
                   "UnsharpMaskTransform 0.25 -> 0"),
    "redistoff": (CFG.format("train050_redistoff"), "auglabAugRedistOff_v26_6_2_train050_val000",
                  "RedistributeSegTransform 0.4 -> 0"),
    "edgehalf": (CFG.format("train050_edgehalf"), "auglabAugEdgeHalf_v26_6_2_train050_val000",
                 "ScharrTransform 0.25 -> 0.125 and RandomConvTransform 0.20 -> 0.10"),
    "texhalf": (CFG.format("train050_texhalf"), "auglabAugTexHalf_v26_6_2_train050_val000",
                "ScharrTransform 0.25 -> 0.125, RandomConvTransform 0.20 -> 0.10, UnsharpMaskTransform 0.25 -> 0.125"),
}
ORDERS = {2: ["ctrl", "p075", "edge2x", "alphawide"], 3: ["unsharpoff", "redistoff", "edgehalf", "texhalf"]}
O = "benchmark/02_tasks/brain_healthy/on-harmony/5_scripts_on-harmony"
M = "benchmark/02_tasks/brain_ms/open-ms/5_scripts_open-ms"
C = "benchmark/02_tasks/abdomen_healthy/chaos/5_scripts_chaos"
# kind, dir, source, destination stems per round (number per arm, in ORDERS[round]), contrast tag, trainer override
ROWS = [
    ("train", O, "04_train/04_26_train_t1w_auglabAug_v26_6_2_dualval.sh",
     {2: ["04_68", "04_69", "04_70", "04_71"], 3: ["04_72", "04_73", "04_74", "04_75"]}, "t1w", "nnUNetTrainerOnHarmonyAugLabDefault"),
    ("train", M, "04_train/04_22_train_auglabAug_v26_6_2_train050_val000.sh",
     {2: ["04_52", "04_53", "04_54", "04_55"], 3: ["04_56", "04_57", "04_58", "04_59"]}, "flair", None),
    ("train", C, "04_train/04_44_train_t2spir_auglabAug_v26_6_2_train050_val000.sh",
     {2: ["04_72", "04_73", "04_74", "04_75"], 3: ["04_76", "04_77", "04_78", "04_79"]}, "t2spir", None),
    ("predict", O, "05_predict/05_08_predict_t1w_auglabAug_v26_6_2_train050_val000.sh",
     {2: ["05_43", "05_44", "05_45", "05_46"], 3: ["05_47", "05_48", "05_49", "05_50"]}, "t1w", "nnUNetTrainerOnHarmonyAugLabDefault"),
    ("predict", M, "05_predict/05_20_predict_auglabAug_v26_6_2_train050_val000.sh",
     {2: ["05_43", "05_44", "05_45", "05_46"], 3: ["05_47", "05_48", "05_49", "05_50"]}, "flair", None),
    ("predict", C, "05_predict/05_34_predict_t2spir_auglabAug_v26_6_2_train050_val000.sh",
     {2: ["05_58", "05_59", "05_60", "05_61"], 3: ["05_62", "05_63", "05_64", "05_65"]}, "t2spir", None),
]
HDR_TRAIN = """#!/usr/bin/env bash
# GENERATED by scripts/cluster/pa_variants/make_wrappers.py from {src} (do not hand-edit).
# PA variants round {rnd}, arm "{arm}" ({desc}): the headline PALETTE-Aug training of this setting with ONLY the train config set to
# {cfg}. Validation on REAL images (val000){dv}. Epochs, folds, loader settings and env are those of the headline wrapper.
# Usage: bash {dst} [RUN_ID]   (RUN_ID to resume)
"""
HDR_PRED = """#!/usr/bin/env bash
# GENERATED by scripts/cluster/pa_variants/make_wrappers.py from {src} (do not hand-edit).
# Predict with the PA variants round {rnd} arm "{arm}" ({desc}); TRAINER = the trainer it was trained with{dv}.
# Usage: bash {dst} <RUN_ID> [...]   (same arguments as {src})
"""


def strip_header(s):
    lines = s.splitlines(keepends=True)
    i = 1 if lines and lines[0].startswith("#!") else 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        i += 1
    return "".join(lines[i:])


def sub1(s, pat, rep, what, count=1):
    t, n = re.subn(pat, rep, s, flags=re.M)
    if n != count:
        raise SystemExit(f"{what}: expected {count} substitution(s) of /{pat}/, got {n}")
    return t


def build(kind, d, src, dst, arm, trainer, rnd):
    cfg, meth, desc = ARMS[arm]
    body = strip_header((ROOT / d / src).read_text())
    body = sub1(body, r'^METHOD="[^"]*"', f'METHOD="{meth}"', f"{dst} METHOD")
    if trainer:
        body = sub1(body, r'^TRAINER="[^"]*"', f'TRAINER="{trainer}"', f"{dst} TRAINER")
    dv = f" (headline used a DualVal trainer; this uses {trainer})" if trainer else ""
    if kind == "train":
        body = sub1(body, re.escape(OLDCFG), cfg, f"{dst} config")
        if "AUGLAB_VAL_PARAMS_GPU_JSON" in body:
            body = sub1(body, r'^export AUGLAB_VAL_PARAMS_GPU_JSON=.*$', 'export AUGLAB_VAL_PARAMS_GPU_JSON=""', f"{dst} val json")
        body = re.sub(r'(LOG_DIR="[^"]*?)(auglabAug_v26_6_2_(?:train050_(?:dualval|val000)|dualval))', r'\g<1>' + meth, body)
        if "DA_WORKERS" in body or "NNUNET_COMPILE" in body:
            raise SystemExit(f"{src}: sets loader variables; the round must use the dataset defaults")
        hdr = HDR_TRAIN.format(src=Path(src).name, dst=Path(dst).name, cfg=cfg, dv=dv, arm=arm, desc=desc, rnd=rnd)
    else:
        hdr = HDR_PRED.format(src=Path(src).name, dst=Path(dst).name, dv=dv, arm=arm, desc=desc, rnd=rnd)
    return hdr + "\n" + body


def wrappers(rounds):
    for rnd in rounds:
        for kind, d, src, nums, tc, trainer in ROWS:
            for arm, num in zip(ORDERS[rnd], nums[rnd]):
                verb = "train" if kind == "train" else "predict"
                dst = f"{src.split('/')[0]}/{num}_{verb}_{tc}_{ARMS[arm][1]}.sh"
                yield kind, d, src, dst, arm, trainer, rnd


def main():
    check = "--check" in sys.argv
    rounds = [int(sys.argv[sys.argv.index("--round") + 1])] if "--round" in sys.argv else ([2, 3] if check else [2])
    bad = 0
    for kind, d, src, dst, arm, trainer, rnd in wrappers(rounds):
        out = build(kind, d, src, dst, arm, trainer, rnd)
        p = ROOT / d / dst
        if check:
            ok = p.exists() and p.read_text() == out
            bad += not ok
            print(("OK   " if ok else "DIFF ") + str(p.relative_to(ROOT)))
        else:
            others = [q for q in p.parent.glob(dst.split("/")[1][:5] + "_*") if q.name != p.name]
            if others:
                raise SystemExit(f"number {dst.split('/')[1][:5]} already used in {p.parent}: {[q.name for q in others]}")
            p.write_text(out)
            print("wrote", p.relative_to(ROOT))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
