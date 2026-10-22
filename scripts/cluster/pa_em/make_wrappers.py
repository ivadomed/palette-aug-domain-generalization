#!/usr/bin/env python
"""Generate the train / predict wrappers of the "PA_EM + Auglab" arm (pa_em, 2026-10-22).

METHOD auglabAugEM_v26_6_2_train050_val000, AugLab config
transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2EM_GPUTransform_train050.json (= the headline PA+Auglab config
plus "partition": "em", "em_n_iters": 20; AugLab e0d8422; proof: scripts/cluster/pa_em/make_config.py).
Every wrapper is a CLONE of the matching wrapper of the ssem50 arm (scripts/cluster/ssem50/make_wrappers.py, 2026-10-18), so a pa_em
fold and an ssem50 fold differ ONLY in the AugLab config and METHOD: same plain <Dataset>AugLabDefault trainer (validation on clean real
images = val000 checkpoint selection), same policy epochs (incl. the chaos t1in EPOCH_FIX 200), same dataset loader defaults (no wrapper sets
DA_WORKERS / nnUNet_compile), same companions. Changes per file:
  all      : the generated header; METHOD value (auglabAug_synthseg_EM_train050_val000 -> auglabAugEM_v26_6_2_train050_val000, brats contrast
             prefix kept); LOG_DIR leaf (it ends in METHOD); the word ssem50 in an inline comment / usage message -> pa_em
  train    : the AugLab config file name
  numbering: new free 04_XX / 05_XX numbers per dataset (checked: no other file uses the number)
  python scripts/cluster/pa_em/make_wrappers.py [--check] [--diffs <dir>]
     --check : every generated file exists and equals the generated content (no writes)
     --diffs : one unified diff (ssem50 wrapper -> pa_em wrapper) per generated file + a summary of the differing line kinds
"""
import difflib, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
T = "benchmark/02_tasks"
OLDM = "auglabAug_synthseg_EM_train050_val000"
METHOD = "auglabAugEM_v26_6_2_train050_val000"
OLDCFG = "transform_params_gpu_default01-23_auglabAug_Synthseg_EM_train050.json"
NEWCFG = "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2EM_GPUTransform_train050.json"
D = {"brats": "brain_tumor/brats2024-glioma", "chaos": "abdomen_healthy/chaos", "onh": "brain_healthy/on-harmony",
     "openms": "brain_ms/open-ms", "tf2": "mandible_healthy/toothfairy2", "ispy2": "breast_cancer/ispy2",
     "pelvic": "pelvis_healthy/totalseg-pelvic", "atlas": "liver_hcc/atlas-liver-hcc", "panseg": "pancreas_disease/pansegdata",
     "duke": "breast_cancer/duke-breast-mri", "ispy1": "breast_cancer/ispy1", "acrin": "breast_cancer/acrin6698",
     "hanseg": "mandible_healthy/hanseg", "pddca": "mandible_healthy/pddca", "lld": "liver_hcc/lld-mmri-hcc",
     "liverhccseg": "liver_hcc/liverhccseg"}
# kind | ds key | setting | ssem50 number | pa_em number | name stem (between the number and METHOD)
F = [
 ("train", "brats", "brats_t1n", "04_62", "04_36", "train_t1n"), ("predict", "brats", "brats_t1n", "05_83", "05_87", "predict_t1n"),
 ("train", "brats", "brats_t2w", "04_63", "04_37", "train_t2w"), ("predict", "brats", "brats_t2w", "05_84", "05_88", "predict_t2w"),
 ("train", "brats", "brats_t2f", "04_64", "04_38", "train_t2f"), ("predict", "brats", "brats_t2f", "05_85", "05_89", "predict_t2f"),
 ("train", "brats", "brats_t1c", "04_65", "04_39", "train_t1c"), ("predict", "brats", "brats_t1c", "05_86", "05_90", "predict_t1c"),
 ("train", "chaos", "chaos_t1in", "04_83", "04_85", "train_t1in"), ("predict", "chaos", "chaos_t1in", "05_69", "05_71", "predict_t1in"),
 ("train", "chaos", "chaos_t2spir", "04_84", "04_86", "train_t2spir"), ("predict", "chaos", "chaos_t2spir", "05_70", "05_72", "predict_t2spir"),
 ("train", "onh", "onh_T1w", "04_79", "04_82", "train_t1w"), ("predict", "onh", "onh_T1w", "05_91", "05_94", "predict_t1w"),
 ("train", "onh", "onh_T2w", "04_80", "04_83", "train_t2w"), ("predict", "onh", "onh_T2w", "05_92", "05_95", "predict_t2w"),
 ("train", "onh", "onh_dwi_ap", "04_81", "04_84", "train_dwi_ap"), ("predict", "onh", "onh_dwi_ap", "05_93", "05_96", "predict_dwi_ap"),
 ("train", "openms", "openms_flair", "04_63", "04_65", "train_flair"), ("predict", "openms", "openms_flair", "05_54", "05_56", "predict_flair"),
 ("train", "openms", "openms_t1w", "04_64", "04_66", "train_t1w"), ("predict", "openms", "openms_t1w", "05_55", "05_57", "predict_t1w"),
 ("train", "tf2", "tf2_cbct", "04_23", "04_24", "train_cbct"), ("predict", "tf2", "tf2_cbct", "05_32", "05_33", "predict_cbct"),
 ("train", "ispy2", "ispy2_t1wce", "04_38", "04_40", "train_t1wce"), ("predict", "ispy2", "ispy2_t1wce", "05_50", "05_52", "predict_t1wce"),
 ("train", "ispy2", "ispy2_t2w", "04_39", "04_41", "train_t2w"), ("predict", "ispy2", "ispy2_t2w", "05_51", "05_53", "predict_t2w"),
 ("train", "pelvic", "pelvic_ct", "04_35", "04_37", "train_ct"), ("predict", "pelvic", "pelvic_ct", "05_24", "05_26", "predict_ct"),
 ("train", "pelvic", "pelvic_mri", "04_36", "04_38", "train_mri"), ("predict", "pelvic", "pelvic_mri", "05_25", "05_27", "predict_mri"),
 ("train", "atlas", "atlas_t1w", "04_19", "04_20", "train_t1w"), ("predict", "atlas", "atlas_t1w", "05_36", "05_37", "predict_t1w"),
 ("train", "panseg", "panseg_t1wce", "04_35", "04_37", "train_t1wce"), ("predict", "panseg", "panseg_t1wce", "05_32", "05_34", "predict_t1wce"),
 ("train", "panseg", "panseg_t2w", "04_36", "04_38", "train_t2w"), ("predict", "panseg", "panseg_t2w", "05_33", "05_35", "predict_t2w"),
 # companions (RUN_ID already a required argument in the ssem50 clones)
 ("companion", "duke", "ispy2_t1wce", "05_23", "05_25", "predict_ispy2_t1wce"), ("companion", "duke", "ispy2_t2w", "05_24", "05_26", "predict_ispy2_t2w"),
 ("companion", "ispy1", "ispy2_t1wce", "05_23", "05_25", "predict_ispy2_t1wce"), ("companion", "ispy1", "ispy2_t2w", "05_24", "05_26", "predict_ispy2_t2w"),
 ("companion", "acrin", "ispy2_t1wce", "05_23", "05_25", "predict_ispy2_t1wce"), ("companion", "acrin", "ispy2_t2w", "05_24", "05_26", "predict_ispy2_t2w"),
 ("companion", "hanseg", "tf2_cbct", "05_31", "05_32", "predict"), ("companion", "pddca", "tf2_cbct", "05_31", "05_32", "predict"),
 ("companion", "lld", "atlas_t1w", "05_24", "05_25", "predict_atlas"), ("companion", "liverhccseg", "atlas_t1w", "05_24", "05_25", "predict_atlas"),
]
HDR = """#!/usr/bin/env bash
# GENERATED by scripts/cluster/pa_em/make_wrappers.py from {src} (the ssem50 arm's wrapper; do not hand-edit).
# "PA_EM + Auglab" arm (pa_em): METHOD {meth}, {what}
# Config {cfg} = the headline PA+Auglab config + "partition": "em", "em_n_iters": 20 (PA's 1-D K-means partition replaced by SynthSeg's
# 1-D GMM EM; AugLab e0d8422; scripts/cluster/pa_em/make_config.py). Only METHOD / config / this header differ from {src}.
"""
WHAT = {"train": "plain AugLabDefault trainer (validation on clean real images), policy epochs, dataset loader defaults -- all as the\n"
                 "# ssem50 wrapper. Usage: bash {dst} [RUN_ID]   (RUN_ID to resume)",
        "predict": "predict with the arm's models. Usage: same arguments as {src}",
        "companion": "predict this companion test set with the {setting}-trained arm; RUN_ID REQUIRED.\n"
                     "# Usage: bash {dst} <RUN_ID> [FOLD] [ITEMS...]"}


def strip_header(s):
    lines = s.splitlines(keepends=True)
    i = 1 if lines and lines[0].startswith("#!") else 0
    while i < len(lines) and (lines[i].startswith("#") or not lines[i].strip()):
        i += 1
    return "".join(lines[i:])


def sub1(s, pat, rep, what, count=1, flags=re.M):
    t, n = re.subn(pat, rep, s, flags=flags)
    if n != count:
        raise SystemExit(f"{what}: expected {count} substitution(s) of /{pat}/, got {n}")
    return t


def paths(kind, ds, old, new, stem):
    d = ROOT / T / D[ds] / f"5_scripts_{Path(D[ds]).name}" / ("04_train" if kind == "train" else "05_predict")
    return d / f"{old}_{stem}_{OLDM}.sh", d / f"{new}_{stem}_{METHOD}.sh"


def build(kind, ds, setting, src):
    body = strip_header(src.read_text())
    body = sub1(body, r'^(METHOD="(?:[A-Za-z0-9]+_)?)' + re.escape(OLDM) + r'"$', r'\g<1>' + METHOD + '"', f"{src.name} METHOD")
    if kind == "train":
        body = sub1(body, re.escape(OLDCFG), NEWCFG, f"{src.name} config")
        body = sub1(body, r'^(LOG_DIR="[^"]*?)' + re.escape(OLDM) + r'("?)$', r'\g<1>' + METHOD + r'\g<2>', f"{src.name} LOG_DIR")
        if re.search(r"^\s*(export\s+)?(DA_WORKERS|nnUNet_compile|ALLOW_METHOD_LOADER_OVERRIDE|FOLD_SLOT_GPU)=", body, re.M):
            raise SystemExit(f"{src}: sets loader / slot variables")
        if "AUGLAB_VAL_PARAMS_GPU_JSON" in body and not re.search(r'^export AUGLAB_VAL_PARAMS_GPU_JSON=""', body, re.M):
            raise SystemExit(f"{src}: synthetic validation config set")
        if not re.search(r"^TRAINER=\"nnUNetTrainer[A-Za-z0-9]*AugLabDefault\"$", body, re.M):
            raise SystemExit(f"{src}: trainer is not a plain AugLabDefault trainer")
    if OLDM in body or "Synthseg_EM" in body:
        raise SystemExit(f"{src}: old METHOD / config left after substitution")
    body = body.replace("ssem50", "pa_em")   # inline EPOCH_FIX comment (chaos t1in) / companion usage message
    return body


def main():
    check = "--check" in sys.argv
    diffs = Path(sys.argv[sys.argv.index("--diffs") + 1]) if "--diffs" in sys.argv else None
    bad = n = 0; kinds = {}
    for kind, ds, setting, old, new, stem in F:
        src, dst = paths(kind, ds, old, new, stem)
        if not src.exists():
            raise SystemExit(f"missing ssem50 source {src}")
        what = WHAT[kind].format(dst=dst.name, src=src.name, setting=setting)
        out = HDR.format(src=src.name, meth=METHOD, what=what, cfg=NEWCFG) + "\n" + build(kind, ds, setting, src)
        n += 1
        if check:
            ok = dst.exists() and dst.read_text() == out; bad += not ok
            print(("OK   " if ok else "DIFF ") + str(dst.relative_to(ROOT)))
        else:
            others = [q.name for q in dst.parent.glob(new + "_*") if q.name != dst.name]
            if others:
                raise SystemExit(f"number {new} already used in {dst.parent}: {others}")
            if dst.exists() and dst.read_text() != out:
                raise SystemExit(f"{dst} exists with different content")
            dst.write_text(out); print("wrote", dst.relative_to(ROOT))
        if diffs:
            diffs.mkdir(parents=True, exist_ok=True)
            a = strip_header(src.read_text()).splitlines(True); b = strip_header(out).splitlines(True)
            dl = list(difflib.unified_diff(src.read_text().splitlines(True), out.splitlines(True),
                                           str(src.relative_to(ROOT)), str(dst.relative_to(ROOT)), n=0))
            (diffs / f"{Path(D[ds]).name}__{dst.name}.diff").write_text("".join(dl))
            # classify the BODY differences (header excluded): every changed body line must be METHOD / config / LOG_DIR / comment
            for line in difflib.ndiff(a, b):
                if line[:2] not in ("- ", "+ "):
                    continue
                t = line[2:].strip()
                k = ("METHOD" if t.startswith("METHOD=") else "config" if "AUGLAB_PARAMS_GPU_JSON" in t else
                     "LOG_DIR" if t.startswith("LOG_DIR=") else "epoch_fix_comment" if t.startswith("export NNUNET_NUM_EPOCHS") and "EPOCH_FIX" in t
                     else "usage_message" if t.startswith("RUN_ID=\"${1:?usage") else "OTHER:" + t)
                kinds.setdefault(f"{kind}:{k}", []).append(dst.name)
    if diffs:
        with open(diffs / "SUMMARY.txt", "w") as fh:
            for k, v in sorted(kinds.items()):
                fh.write(f"{k}\t{len(v)} lines\t{len(set(v))} files\n")
        print((diffs / "SUMMARY.txt").read_text())
        if any(":OTHER:" in k for k in kinds):
            print("UNEXPECTED BODY DIFFERENCES"); bad += 1
    print(f"{n} wrappers {'checked' if check else 'generated'}"); sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
