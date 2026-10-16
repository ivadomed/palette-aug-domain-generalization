#!/usr/bin/env python
"""Generate the PALETTE + REDUCED-Auglab ("texlite") train/predict wrappers from each dataset's existing OURS wrappers.

The new method = the dataset's headline PALETTE-Aug (OURS) training with ONLY the train config swapped for
transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_texlite.json (same PALETTE block, same
spatial pipeline and flip; Scharr, RandomConv, GaussianNoise, GaussianBlur and SimulateLowRes off, UnsharpMask halved -- the
operators whose leave-one-out removal from PALETTE-Aug raised NGF texture fidelity, per-operator NGF sweep 2026-10-15). Validation = real images (val000): where the headline run used a DualVal trainer
the plain AugLabDefault trainer of the same dataset is used (no synthetic-validation pass, no val100 mirror). Epochs, folds,
DA workers, compile flag and env are those of the headline wrapper.
METHOD = auglabAugTexlite_v26_6_2_train050_val000 (does NOT contain the REF_SUBSTR auglabAug_v26_6_2_train050_val000, so the
headline run stays the unique `sig. vs ref` reference).

--arm keepres: second arm "texlite_keepres" = the same, with config ..._train050_texlite_keepres.json (GaussianBlur 0.2 and
SimulateLowRes 0.25 kept as in the headline config; only Scharr, RandomConv, GaussianNoise off and UnsharpMask halved),
METHOD auglabAugTexliteKeepres_v26_6_2_train050_val000, wrappers numbered one slot after the texlite ones (KEEPRES_NUM).
--arm edgeonly: third arm = the headline config with ONLY ScharrTransform and RandomConvTransform off (..._train050_edgeonly.json),
METHOD auglabAugEdgeonly_v26_6_2_train050_val000, development settings only (on-harmony T1w, open-ms flair, chaos t2spir; EDGEONLY_NUM).
Development settings (choice of the operator set) = on-harmony T1w, open-ms flair, chaos t2spir; glioma, mandible and HCC are held out
(their texlite/keepres wrappers exist but their runs were cancelled 2026-10-15).

Runs on any checkout (Vulcan: liver task under benchmark/02_tasks/liver_hcc; TamIA: still under benchmark/03_archive) and
derives every file from THAT checkout's own sibling, so relative hop counts stay right.
  python scripts/cluster/auglab_texlite/make_wrappers.py [--arm texlite|keepres|edgeonly] [--check]
"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OLDCFG = "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
# arm -> (train config, METHOD, label, operator change vs the headline config)
ARMS = {
    "texlite": ("transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_texlite.json",
                "auglabAugTexlite_v26_6_2_train050_val000", '"texlite"',
                "Scharr, RandomConv, GaussianNoise,\n# GaussianBlur, SimulateLowRes off and UnsharpMask halved: the texture-erasing operators of the per-operator NGF sweep"),
    "keepres": ("transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_texlite_keepres.json",
                "auglabAugTexliteKeepres_v26_6_2_train050_val000", '"texlite_keepres"',
                "Scharr, RandomConv, GaussianNoise\n# off and UnsharpMask halved; the resolution operators GaussianBlur and SimulateLowRes kept as in the headline config"),
    "edgeonly": ("transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050_edgeonly.json",
                 "auglabAugEdgeonly_v26_6_2_train050_val000", '"edgeonly"',
                 "ONLY Scharr and RandomConv\n# off; GaussianNoise, GaussianBlur, SimulateLowRes, UnsharpMask and every other key as in the headline config"),
}
ARM = sys.argv[sys.argv.index("--arm") + 1] if "--arm" in sys.argv else "texlite"
NEWCFG, NEWM, LABEL, OPS = ARMS[ARM]


def ds(*cands):
    for c in cands:
        p = ROOT / c
        if p.is_dir():
            return p
    raise SystemExit(f"none of {cands} exists under {ROOT}")


B = lambda: ds("benchmark/02_tasks/brain_tumor/brats2024-glioma/5_scripts_brats2024-glioma")
C = lambda: ds("benchmark/02_tasks/abdomen_healthy/chaos/5_scripts_chaos")
M = lambda: ds("benchmark/02_tasks/brain_ms/open-ms/5_scripts_open-ms")
TF = lambda: ds("benchmark/02_tasks/mandible_healthy/toothfairy2/5_scripts_toothfairy2")
O = lambda: ds("benchmark/02_tasks/brain_healthy/on-harmony/5_scripts_on-harmony")
A = lambda: ds("benchmark/02_tasks/liver_hcc/atlas-liver-hcc/5_scripts_atlas-liver-hcc", "benchmark/03_archive/atlas-liver-hcc/5_scripts_atlas-liver-hcc")
HS = lambda: ds("benchmark/02_tasks/mandible_healthy/hanseg/5_scripts_hanseg")
PD = lambda: ds("benchmark/02_tasks/mandible_healthy/pddca/5_scripts_pddca")
LH = lambda: ds("benchmark/02_tasks/liver_hcc/lld-mmri-hcc/5_scripts_lld-mmri-hcc", "benchmark/03_archive/lld-mmri-hcc/5_scripts_lld-mmri-hcc")
LS = lambda: ds("benchmark/02_tasks/liver_hcc/liverhccseg/5_scripts_liverhccseg", "benchmark/03_archive/liverhccseg/5_scripts_liverhccseg")

# kind, dir, source, destination, trainer override (None = keep the source's TRAINER)
ROWS = [
    ("train", B, "04_train/04_28_train_t1n_auglabAug_v26_6_2_train050_val000.sh", "04_train/04_34_train_t1n_auglabAugTexlite_v26_6_2_train050_val000.sh", None),
    ("train", C, "04_train/04_44_train_t2spir_auglabAug_v26_6_2_train050_val000.sh", "04_train/04_69_train_t2spir_auglabAugTexlite_v26_6_2_train050_val000.sh", None),
    ("train", M, "04_train/04_22_train_auglabAug_v26_6_2_train050_val000.sh", "04_train/04_49_train_flair_auglabAugTexlite_v26_6_2_train050_val000.sh", None),
    ("train", TF, "04_train/04_06_train_auglabAug_v26_6_2_dualval.sh", "04_train/04_21_train_cbct_auglabAugTexlite_v26_6_2_train050_val000.sh", "nnUNetTrainerToothFairy2AugLabDefault"),
    ("train", A, "04_train/04_06_train_auglabAug_v26_6_2_dualval.sh", "04_train/04_17_train_t1w_auglabAugTexlite_v26_6_2_train050_val000.sh", "nnUNetTrainerAtlasLiverHCCAugLabDefault"),
    ("train", O, "04_train/04_26_train_t1w_auglabAug_v26_6_2_dualval.sh", "04_train/04_65_train_t1w_auglabAugTexlite_v26_6_2_train050_val000.sh", "nnUNetTrainerOnHarmonyAugLabDefault"),
    ("predict", B, "05_predict/05_20_predict_t1n_auglabAug_v26_6_2_train050_val000.sh", "05_predict/05_30_predict_t1n_auglabAugTexlite_v26_6_2_train050_val000.sh", None),
    ("predict", C, "05_predict/05_34_predict_t2spir_auglabAug_v26_6_2_train050_val000.sh", "05_predict/05_55_predict_t2spir_auglabAugTexlite_v26_6_2_train050_val000.sh", None),
    ("predict", M, "05_predict/05_20_predict_auglabAug_v26_6_2_train050_val000.sh", "05_predict/05_40_predict_flair_auglabAugTexlite_v26_6_2_train050_val000.sh", None),
    ("predict", TF, "05_predict/05_07_predict_ours_val000.sh", "05_predict/05_13_predict_cbct_auglabAugTexlite_v26_6_2_train050_val000.sh", "nnUNetTrainerToothFairy2AugLabDefault"),
    ("predict", A, "05_predict/05_07_predict_auglabAug_v26_6_2_train050_val000_dualval.sh", "05_predict/05_12_predict_t1w_auglabAugTexlite_v26_6_2_train050_val000.sh", "nnUNetTrainerAtlasLiverHCCAugLabDefault"),
    ("predict", O, "05_predict/05_08_predict_t1w_auglabAug_v26_6_2_train050_val000.sh", "05_predict/05_40_predict_t1w_auglabAugTexlite_v26_6_2_train050_val000.sh", "nnUNetTrainerOnHarmonyAugLabDefault"),
    # mandible companions (the Mandible task's held-out modalities CT / MR-T1 live only there)
    ("predict", HS, "05_predict/05_07_predict_ours_val000.sh", "05_predict/05_13_predict_auglabAugTexlite_v26_6_2_train050_val000.sh", "nnUNetTrainerToothFairy2AugLabDefault"),
    ("predict", PD, "05_predict/05_07_predict_ours_val000.sh", "05_predict/05_13_predict_auglabAugTexlite_v26_6_2_train050_val000.sh", "nnUNetTrainerToothFairy2AugLabDefault"),
    # liver companions (the HCC task's held-out contrasts live only there); clone of their atlas flat-fill rung wrapper
    ("predict", LH, "05_predict/05_15_predict_atlas_baseline_kmeans_label_remap_voronoi_flatfill.sh", "05_predict/05_16_predict_atlas_auglabAugTexlite_v26_6_2_train050_val000.sh", None),
    ("predict", LS, "05_predict/05_15_predict_atlas_baseline_kmeans_label_remap_voronoi_flatfill.sh", "05_predict/05_16_predict_atlas_auglabAugTexlite_v26_6_2_train050_val000.sh", None),
]
HDR_TRAIN = """#!/usr/bin/env bash
# GENERATED by scripts/cluster/auglab_texlite/make_wrappers.py{argflag} from {src} (do not hand-edit).
# PALETTE + REDUCED Auglab ({label}): the headline PALETTE-Aug training of this setting with ONLY the train config swapped
# for {cfg} (same PALETTE block, same flip/spatial pipeline; {ops}). Validation on REAL images (val000){dv}.
# Epochs, folds (0 1 2), DA workers and env are those of the headline wrapper.
# Usage: bash {dst} [RUN_ID]   (RUN_ID to resume)
"""
HDR_PRED = """#!/usr/bin/env bash
# GENERATED by scripts/cluster/auglab_texlite/make_wrappers.py{argflag} from {src} (do not hand-edit).
# Predict with the PALETTE + REDUCED Auglab ({label}) model; TRAINER = the trainer it was trained with{dv}.
# Usage: bash {dst} <RUN_ID> [...]   (same arguments as {src})
"""


# keepres destination = the texlite destination one number up, METHOD swapped (numbers checked free on Vulcan AND TamIA)
KEEPRES_NUM = {"04_34": "04_35", "05_30": "05_31", "04_69": "04_70", "05_55": "05_56", "04_49": "04_50", "05_40": "05_41",
               "04_21": "04_22", "05_13": "05_14", "04_17": "04_18", "05_12": "05_13", "05_16": "05_17", "04_65": "04_66"}
# edgeonly: development settings only, two slots after texlite (rows without an entry are not generated for this arm)
EDGEONLY_NUM = {"04_49": "04_51", "05_40": "05_42", "04_69": "04_71", "05_55": "05_57", "04_65": "04_67"}
ARM_NUM = {"keepres": KEEPRES_NUM, "edgeonly": EDGEONLY_NUM}
ARGFLAG = "" if ARM == "texlite" else f" --arm {ARM}"


def arm_dst(dst):
    if ARM == "texlite":
        return dst
    d, f = dst.split("/")
    if f[:5] not in ARM_NUM[ARM]:
        return None
    return d + "/" + ARM_NUM[ARM][f[:5]] + f[5:].replace("auglabAugTexlite_v26_6_2_train050_val000", NEWM)


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


def build(kind, d, src, dst, trainer):
    s = (d / src).read_text()
    body = strip_header(s)
    body = sub1(body, r'^METHOD="[^"]*"', f'METHOD="{NEWM}"', f"{dst} METHOD")
    if trainer:
        body = sub1(body, r'^TRAINER="[^"]*"', f'TRAINER="{trainer}"', f"{dst} TRAINER")
    dv = f" (headline used a DualVal trainer; this uses {trainer})" if trainer else ""
    if kind == "train":
        body = sub1(body, re.escape(OLDCFG), NEWCFG, f"{dst} config")
        if "AUGLAB_VAL_PARAMS_GPU_JSON" in body:
            body = sub1(body, r'^export AUGLAB_VAL_PARAMS_GPU_JSON=.*$', 'export AUGLAB_VAL_PARAMS_GPU_JSON=""', f"{dst} val json")
        body = re.sub(r'(LOG_DIR="[^"]*?)(auglabAug_v26_6_2_(?:train050_(?:dualval|val000)|dualval))', r'\g<1>' + NEWM, body)
        # hop count to the repo root from THIS file's location (TamIA's archived liver copy carries a stale 4-hop path)
        hops = "../" * len((d / dst).parent.relative_to(ROOT).parts)
        if "AUGLAB_CONFIGS_DIR=" in body:   # (on-harmony resolves the configs from PROJECT_ROOT: no relative hops to fix)
            body = sub1(body, r'(AUGLAB_CONFIGS_DIR="\$\(cd "\$\(dirname "\$0"\)/)(?:\.\./)+(sub-workspaces/)', r'\g<1>' + hops + r'\g<2>', f"{dst} hops")
        hdr = HDR_TRAIN.format(src=Path(src).name, dst=Path(dst).name, cfg=NEWCFG, dv=dv, label=LABEL, ops=OPS, argflag=ARGFLAG)
    else:
        hdr = HDR_PRED.format(src=Path(src).name, dst=Path(dst).name, dv=dv, label=LABEL, argflag=ARGFLAG)
    return hdr + "\n" + body


def main():
    check = "--check" in sys.argv
    bad = 0
    for kind, dfn, src, dst, trainer in ROWS:
        dst = arm_dst(dst)
        if dst is None:
            continue
        d = dfn()
        out = build(kind, d, src, dst, trainer)
        p = d / dst
        if check:
            ok = p.exists() and p.read_text() == out
            bad += not ok
            print(("OK   " if ok else "DIFF ") + str(p.relative_to(ROOT)))
        else:
            p.write_text(out)
            print("wrote", p.relative_to(ROOT))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
