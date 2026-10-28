#!/usr/bin/env python
"""Generate the train / predict wrappers of the "flat fill + Auglab" arm (flatauglab, 2026-10-28).

METHOD auglabAugFlat_v26_6_2_train050_val000 = the HEADLINE PA+Auglab method with PA's real fill replaced by the rung-4.5 flat fill,
everything else identical. Configs (scripts/cluster/flatauglab/make_config.py, AugLab):
  TRAIN transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2FlatFillGPUTransform_train050.json
  VAL   transform_params_gpu_VALsynthonly_ImageContrastV26_6_2FlatFillGPUTransform.json   (DualVal trainers only: val100 mirror)
Sources = the wrappers that produced each setting's HEADLINE PA+Auglab run (the run every *_combined_01_results.yaml reads; trainer
checked against that run's directory). Same trainer class as the headline: DualVal where the headline was DualVal (brats t2w/t2f/t1c,
on-harmony T1w/dwi_ap, toothfairy2, ispy2, totalseg-pelvic, atlas, pansegdata), plain <DS>AugLabDefault (brats t1n, on-harmony T2w,
open-ms) or nnUNetTrainerCHAOSAugLabV26_6_2 (chaos; = AugLabDefault with a WandB-panel flag only) otherwise. The arm reports val000
(checkpoint_best on real validation) everywhere; DualVal also makes a ..._val100_ mirror (flat synthetic validation), not used.
  TRAIN    : changes METHOD, the train config file name, the val config file name (DualVal), the LOG_DIR leaf, the header.
             chaos t1in (CHAOS_T1IN_FIX): the headline source 04_09 hard-wires SINGLE_FOLD=0 / SINGLE_SLOT=3 / SINGLE_GPU=3 (one fold) and a
             300-epoch default; the headline run itself ran 205/199/199 epochs (folds 1-2 checkpoint_final at 200). The clone drops the three
             SINGLE_* lines and sets the default to 200 (chaos policy), exactly as rung 7's 04_61 (PV arm) did.
  PREDICT  : the setting's headline PA+Auglab val000 predict wrapper. Changes: METHOD (brats keeps its contrast prefix) + header.
  COMPANION: the companion's headline PA+Auglab predict wrapper for every external test set the combined config reads. Changes: METHOD,
             header, and a baked default RUN_ID ("${1:-<old run>}") replaced by a REQUIRED argument.
Not generated (no per-method wrapper): amos / sliver07 (chaos fov_crop RUNS_FILE, launch.sh chaos-fov-runs); totalsegmri-pancreas /
msd-pancreas (roster shim with an explicit RUN_ID, companions_vulcan.sh); amos-pancreas (not read by the combined config).
  python scripts/cluster/flatauglab/make_wrappers.py [--check] [--roster <tsv>] [--diffs <dir>]
"""
import difflib, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
T = "benchmark/02_tasks"
METHOD = "auglabAugFlat_v26_6_2_train050_val000"
OLDM = "auglabAug_v26_6_2_train050_val000"
OLDCFG = "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
NEWCFG = "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2FlatFillGPUTransform_train050.json"
OLDVAL = "transform_params_gpu_VALsynthonly_ImageContrastV26_6_2GPUTransform.json"
NEWVAL = "transform_params_gpu_VALsynthonly_ImageContrastV26_6_2FlatFillGPUTransform.json"
CHAOS_T1IN_FIX = "chaos_t1in"
D = {"brats": "brain_tumor/brats2024-glioma", "chaos": "abdomen_healthy/chaos", "onh": "brain_healthy/on-harmony",
     "openms": "brain_ms/open-ms", "tf2": "mandible_healthy/toothfairy2", "ispy2": "breast_cancer/ispy2",
     "pelvic": "pelvis_healthy/totalseg-pelvic", "atlas": "liver_hcc/atlas-liver-hcc", "panseg": "pancreas_disease/pansegdata",
     "duke": "breast_cancer/duke-breast-mri", "ispy1": "breast_cancer/ispy1", "acrin": "breast_cancer/acrin6698",
     "hanseg": "mandible_healthy/hanseg", "pddca": "mandible_healthy/pddca", "lld": "liver_hcc/lld-mmri-hcc",
     "liverhccseg": "liver_hcc/liverhccseg"}
# setting | ds | TRAINING_CONTRAST | name tag | headline PA+Auglab val000 run (combined config) | train src | new no | predict src | new no
#   | expected trainer (= the headline run's trainer dir) | epochs
S = [
 ("brats_t1n", "brats", "t1n", "t1n", "brats2024-glioma_t1n_auglabAug_v26_6_2_train050_val000_20260710_040303", "04_28_train_t1n_auglabAug_v26_6_2_train050_val000.sh", "04_66", "05_20_predict_t1n_auglabAug_v26_6_2_train050_val000.sh", "05_95", "nnUNetTrainerBraTS2024GliomaAugLabDefault", "2500"),
 ("brats_t2w", "brats", "t2w", "t2w", "brats2024-glioma_t2w_auglabAug_v26_6_2_train050_val000_20260725_113540", "04_32_train_t2w_auglabAug_v26_6_2_dualval.sh", "04_67", "05_22_predict_t2w_auglabAug_v26_6_2_train050_val000_dualval.sh", "05_96", "nnUNetTrainerBraTS2024GliomaT2wAugLabDualVal", "2500"),
 ("brats_t2f", "brats", "t2f", "t2f", "brats2024-glioma_t2f_auglabAug_v26_6_2_train050_val000_20260917_113412", "04_55_train_t2f_auglabAug_v26_6_2_dualval.sh", "04_68", "05_55_predict_t2f_auglabAug_v26_6_2_train050_val000_dualval.sh", "05_97", "nnUNetTrainerBraTS2024GliomaT2fAugLabDualVal", "2500"),
 ("brats_t1c", "brats", "t1c", "t1c", "brats2024-glioma_t1c_auglabAug_v26_6_2_train050_val000_20260921_140000", "04_75_train_t1c_auglabAug_v26_6_2_dualval.sh", "04_76", "05_75_predict_t1c_auglabAug_v26_6_2_train050_val000_dualval.sh", "05_98", "nnUNetTrainerBraTS2024GliomaT1cAugLabDualVal", "2500"),
 ("chaos_t1in", "chaos", "t1in", "t1in", "chaos_t1in_auglabAug_v26_6_2_train050_val000_20260615_213615", "04_09_train_auglabAug_v26_6_2_train050_val000.sh", "04_89", "05_09_predict_auglabAug_v26_6_2_train050_val000.sh", "05_75", "nnUNetTrainerCHAOSAugLabV26_6_2", "200"),
 ("chaos_t2spir", "chaos", "t2spir", "t2spir", "chaos_t2spir_auglabAug_v26_6_2_train050_val000_20260710_054202", "04_44_train_t2spir_auglabAug_v26_6_2_train050_val000.sh", "04_90", "05_34_predict_t2spir_auglabAug_v26_6_2_train050_val000.sh", "05_76", "nnUNetTrainerCHAOSAugLabV26_6_2", "200"),
 ("onh_T1w", "onh", "T1w", "t1w", "on-harmony_T1w_auglabAug_v26_6_2_train050_val000_20260727_075205", "04_26_train_t1w_auglabAug_v26_6_2_dualval.sh", "04_88", "05_08_predict_t1w_auglabAug_v26_6_2_train050_val000.sh", "05_24", "nnUNetTrainerOnHarmonyAugLabDualVal", "2000"),
 ("onh_T2w", "onh", "T2w", "t2w", "on-harmony_T2w_auglabAug_v26_6_2_train050_val000_20260710_040443", "04_23_train_t2w_auglabAug_v26_6_2_train050_val000.sh", "04_89", "05_14_predict_t2w_auglabAug_v26_6_2_train050_val000.sh", "05_25", "nnUNetTrainerOnHarmonyAugLabDefault", "2000"),
 ("onh_dwi_ap", "onh", "dwi_ap", "dwi_ap", "on-harmony_dwi_ap_auglabAug_v26_6_2_train050_val000_20260921_203727", "04_41_train_dwi_ap_auglabAug_v26_6_2_dualval.sh", "04_90", "05_20_predict_dwi_ap_auglabAug_v26_6_2_train050_val000.sh", "05_26", "nnUNetTrainerOnHarmonyAugLabDualVal", "2000"),
 ("openms_flair", "openms", "flair", "flair", "open-ms_flair_auglabAug_v26_6_2_train050_val000_20260710_054455", "04_22_train_auglabAug_v26_6_2_train050_val000.sh", "04_69", "05_20_predict_auglabAug_v26_6_2_train050_val000.sh", "05_60", "nnUNetTrainerOpenMSAugLabDefault", "2000"),
 ("openms_t1w", "openms", "t1w", "t1w", "open-ms_t1w_auglabAug_v26_6_2_train050_val000_20260710_054525", "04_23_train_t1w_auglabAug_v26_6_2_train050_val000.sh", "04_70", "05_21_predict_t1w_auglabAug_v26_6_2_train050_val000.sh", "05_61", "nnUNetTrainerOpenMSAugLabDefault", "2000"),
 ("tf2_cbct", "tf2", "cbct", "cbct", "toothfairy2_cbct_auglabAug_v26_6_2_train050_val000_20260908_013027", "04_06_train_auglabAug_v26_6_2_dualval.sh", "04_26", "05_07_predict_ours_val000.sh", "05_35", "nnUNetTrainerToothFairy2AugLabDualVal", "1000"),
 ("ispy2_t1wce", "ispy2", "t1wce", "t1wce", "ispy2_t1wce_auglabAug_v26_6_2_train050_val000_20260905_163655", "04_06_train_t1wce_auglabAug_v26_6_2_dualval.sh", "04_44", "05_41_predict_own_t1wce_auglabAug_v26_6_2_train050_val000.sh", "05_56", "nnUNetTrainerISPY2AugLabDualVal", "1000"),
 ("ispy2_t2w", "ispy2", "t2w", "t2w", "ispy2_t2w_auglabAug_v26_6_2_train050_val000_20260905_163655", "04_13_train_t2w_auglabAug_v26_6_2_dualval.sh", "04_45", "05_28_predict_own_t2w_auglabAug_v26_6_2_train050_val000.sh", "05_57", "nnUNetTrainerISPY2AugLabDualVal", "1000"),
 ("pelvic_ct", "pelvic", "ct", "ct", "ct_ours_20260916_072434 (per-fold dirs ct_ours_f{0,1,2}_20260916_072434)", "04_06_train_ct_auglabAug_v26_6_2_dualval.sh", "04_41", "05_07_predict_ct_auglabAug_v26_6_2_val000.sh", "05_30", "nnUNetTrainerTotalsegPelvicAugLabDualVal", "200"),
 ("pelvic_mri", "pelvic", "mri", "mri", "mri_ours_20260916_072453 (per-fold dirs mri_ours_f{0,1,2}_20260916_072453)", "04_13_train_mri_auglabAug_v26_6_2_dualval.sh", "04_42", "05_18_predict_mri_auglabAug_v26_6_2_val000.sh", "05_31", "nnUNetTrainerTotalsegPelvicAugLabDualVal", "200"),
 ("atlas_t1w", "atlas", "t1w", "t1w", "atlas-liver-hcc_t1w_auglabAug_v26_6_2_train050_val000_20260804_062605", "04_06_train_auglabAug_v26_6_2_dualval.sh", "04_22", "05_07_predict_auglabAug_v26_6_2_train050_val000_dualval.sh", "05_39", "nnUNetTrainerAtlasLiverHCCAugLabDualVal", "2000"),
 ("panseg_t1wce", "panseg", "t1wce", "t1wce", "pansegdata_t1wce_auglabAug_v26_6_2_train050_val000_20261005_201307", "04_06_train_t1wce_auglabAug_v26_6_2_dualval.sh", "04_41", "05_07_predict_t1wce_auglabAug_v26_6_2_train050_val000.sh", "05_38", "nnUNetTrainerPANSEGDATAAugLabDualVal", "2000"),
 ("panseg_t2w", "panseg", "t2w", "t2w", "pansegdata_t2w_auglabAug_v26_6_2_train050_val000_20261005_201414", "04_13_train_t2w_auglabAug_v26_6_2_dualval.sh", "04_42", "05_18_predict_t2w_auglabAug_v26_6_2_train050_val000.sh", "05_39", "nnUNetTrainerPANSEGDATAAugLabDualVal", "2000"),
]
# companion | for setting | headline src predict wrapper | new no | name stem | items the combined config reads
C = [
 ("duke", "ispy2_t1wce", "05_07_predict_ispy2_t1wce_auglabAug_v26_6_2_train050_val000.sh", "05_29", "ispy2_t1wce", "t1wce_uniap precontrast_uniap"),
 ("duke", "ispy2_t2w", "05_13_predict_ispy2_t2w_auglabAug_v26_6_2_train050_val000.sh", "05_30", "ispy2_t2w", "t1wce_uniap precontrast_uniap"),
 ("ispy1", "ispy2_t1wce", "05_07_predict_ispy2_t1wce_auglabAug_v26_6_2_train050_val000.sh", "05_29", "ispy2_t1wce", "t1wce precontrast"),
 ("ispy1", "ispy2_t2w", "05_13_predict_ispy2_t2w_auglabAug_v26_6_2_train050_val000.sh", "05_30", "ispy2_t2w", "t1wce precontrast"),
 ("acrin", "ispy2_t1wce", "05_07_predict_ispy2_t1wce_auglabAug_v26_6_2_train050_val000.sh", "05_29", "ispy2_t1wce", "dwi_uniap"),
 ("acrin", "ispy2_t2w", "05_13_predict_ispy2_t2w_auglabAug_v26_6_2_train050_val000.sh", "05_30", "ispy2_t2w", "dwi_uniap"),
 ("hanseg", "tf2_cbct", "05_07_predict_ours_val000.sh", "05_34", "", "ct mrt1 (S-I flipped _sif inputs, mandible_only_sif)"),
 ("pddca", "tf2_cbct", "05_07_predict_ours_val000.sh", "05_34", "", "ct (S-I flipped _sif inputs, mandible_only_sif)"),
 ("lld", "atlas_t1w", "05_07_predict_atlas_auglabAug_v26_6_2_train050_val000.sh", "05_27", "atlas", "ce-pre ce-art ce-pv ce-del inphase outphase t2wi dwi"),
 ("liverhccseg", "atlas_t1w", "05_07_predict_atlas_auglabAug_v26_6_2_train050_val000.sh", "05_27", "atlas", "ce-pre_T1w ce-art_T1w ce-pv_T1w ce-del_T1w"),
]
NOT_WRAPPED = [
 ("amos", "chaos_t1in chaos_t2spir", "ct mri(->t2w)", "fov_crop: RUNS_FILE for fov_crop_predict_evaluate.sh via 06_36_fov_crop_ladder_rungs.sh (launch.sh chaos-fov-runs)"),
 ("sliver07", "chaos_t1in chaos_t2spir", "ct(->sliver07_ct_liver)", "fov_crop: same RUNS_FILE / 06_36 call"),
 ("totalsegmri-pancreas", "panseg_t1wce panseg_t2w", "t1gre t2like", "pansegdata roster shim with explicit RUN_ID (companions_vulcan.sh)"),
 ("msd-pancreas", "panseg_t1wce panseg_t2w", "ct", "pansegdata roster shim with explicit RUN_ID (companions_vulcan.sh)"),
 ("amos-pancreas", "panseg_t1wce panseg_t2w", "-", "NOT read by pansegdata_combined_01_results.yaml -> skipped"),
]
HDR = """#!/usr/bin/env bash
# GENERATED by scripts/cluster/flatauglab/make_wrappers.py from {src} (do not hand-edit).
# "Flat fill + Auglab" arm (flatauglab): METHOD {meth}, {what}
# Config {cfg} = the headline PA+Auglab config with PA's real fill replaced by the rung-4.5 flat fill (scripts/cluster/flatauglab/make_config.py).
"""
WHAT_TRAIN = ("the setting's headline PA+Auglab training with ONLY METHOD, the config file(s) and the LOG_DIR leaf changed{fix}.\n"
              "# Trainer ({trainer}), epochs, folds and loader settings (dataset defaults, 04_00_common.sh) as in the source.\n"
              "# Reported checkpoint: checkpoint_best on real validation (val000). Usage: bash {dst} [RUN_ID]   (RUN_ID to resume)")
WHAT_PRED = "predict with the arm's models; only METHOD changed vs the source (same TRAINER). Usage: same arguments as {src}"
WHAT_COMP = ("predict this companion test set with the {setting}-trained arm; only METHOD changed and the RUN_ID is REQUIRED.\n"
             "# Usage: bash {dst} <RUN_ID> [FOLD] [ITEMS...]")


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


def new_method_value(old):   # keep a contrast prefix (brats: t2w_auglabAug_... -> t2w_<METHOD>)
    m = re.fullmatch(r"(.*?)" + re.escape(OLDM), old)
    if not m:
        raise SystemExit(f"unexpected source METHOD {old!r}")
    return m.group(1) + METHOD


def files():
    for r in S:
        key, ds, tc, tag, run, tsrc, tno, psrc, pno, tr, ep = r
        yield "train", D[ds], tsrc, f"04_train/{tno}_train_{tag}_{METHOD}.sh", r
        yield "predict", D[ds], psrc, f"05_predict/{pno}_predict_{tag}_{METHOD}.sh", r
    for comp, setting, psrc, pno, stem, items in C:
        r = next(x for x in S if x[0] == setting)
        yield "companion", D[comp], psrc, f"05_predict/{pno}_predict_{stem + '_' if stem else ''}{METHOD}.sh", r


def build(kind, d, src, dst, r):
    key, tr = r[0], r[9]
    text = (ROOT / T / d / f"5_scripts_{Path(d).name}" / ("04_train" if kind == "train" else "05_predict") / src).read_text()
    body = strip_header(text)
    old = re.search(r'^METHOD="([^"]*)"', body, re.M).group(1)
    body = sub1(body, r'^METHOD="[^"]*"', f'METHOD="{new_method_value(old)}"', f"{dst} METHOD")
    btr = re.search(r'^TRAINER="([^"]*)"', body, re.M)
    if not btr or btr.group(1) != tr:
        raise SystemExit(f"{src}: TRAINER {btr and btr.group(1)} != expected {tr} (headline run's trainer)")
    if kind == "train":
        body = sub1(body, re.escape(OLDCFG), NEWCFG, f"{dst} config")
        dual = tr.endswith("DualVal")
        if dual:
            body = sub1(body, r'^(export AUGLAB_VAL_PARAMS_GPU_JSON=".*)' + re.escape(OLDVAL) + r'(")$', r'\g<1>' + NEWVAL + r'\g<2>', f"{dst} val config")
        elif "AUGLAB_VAL_PARAMS_GPU_JSON" in body:
            raise SystemExit(f"{src}: non-DualVal source sets a val config")
        if re.search(r'^LOG_DIR=', body, re.M):
            body = sub1(body, r'^(LOG_DIR="[^"]*?)auglabAug_v26_6_2(?:_train050_val000|_train050_dualval|_dualval)?("?)$', r'\g<1>' + METHOD + r'\g<2>', f"{dst} LOG_DIR")
        fix = ""
        if key == CHAOS_T1IN_FIX:
            for v in ("SINGLE_FOLD", "SINGLE_SLOT", "SINGLE_GPU"):
                body = sub1(body, rf'^export {v}=.*\n', "", f"{dst} drop {v}")
            body = sub1(body, r'^export NNUNET_NUM_EPOCHS="\$\{NNUNET_NUM_EPOCHS:-300\}"$',
                        'export NNUNET_NUM_EPOCHS="${NNUNET_NUM_EPOCHS:-200}"   # flatauglab CHAOS_T1IN_FIX: chaos policy 200 (source default 300, see make_wrappers.py)',
                        f"{dst} epochs")
            fix = (", plus the three SINGLE_* one-fold pins dropped and the epoch default 300 -> 200 (chaos policy; the headline run ran\n"
                   "# 205/199/199 epochs) -- CHAOS_T1IN_FIX in make_wrappers.py, as rung 7's 04_61 did")
        if re.search(r"^\s*(export\s+)?(DA_WORKERS|nnUNet_compile|ALLOW_METHOD_LOADER_OVERRIDE|FOLD_SLOT_GPU|SINGLE_FOLD|SINGLE_SLOT|SINGLE_GPU)=", body, re.M):
            raise SystemExit(f"{src}: sets loader / slot / single-fold variables; the arm must use the dataset defaults and folds 0 1 2")
        m = re.search(r'NNUNET_NUM_EPOCHS(?:="\$\{NNUNET_NUM_EPOCHS:-|=)(\d+)', body)
        if not m or m.group(1) != r[10]:
            raise SystemExit(f"{dst}: epochs {m and m.group(1)} != policy {r[10]}")
        what = WHAT_TRAIN.format(fix=fix, dst=Path(dst).name, trainer=tr)
    elif kind == "predict":
        what = WHAT_PRED.format(src=src)
    else:
        if re.search(r'^RUN_ID="\$\{1:-', body, re.M):
            body = sub1(body, r'^RUN_ID="\$\{1:-[^}]*\}"$', 'RUN_ID="${1:?usage: <RUN_ID of the flatauglab arm> [FOLD] [ITEMS...]}"', f"{dst} RUN_ID")
        what = WHAT_COMP.format(setting=key, dst=Path(dst).name)
    if OLDM in body.replace(METHOD, "") or OLDCFG in body or (OLDVAL in body):
        raise SystemExit(f"{dst}: a headline method/config name survived: {[l for l in body.splitlines() if OLDM in l or OLDCFG in l or OLDVAL in l]}")
    return HDR.format(src=src, meth=METHOD, what=what, cfg=NEWCFG) + "\n" + body


def path(d, dst):
    return ROOT / T / d / f"5_scripts_{Path(d).name}" / dst


def main():
    check = "--check" in sys.argv
    diffs = Path(sys.argv[sys.argv.index("--diffs") + 1]) if "--diffs" in sys.argv else None
    bad = n = 0; kinds = {}
    for kind, d, src, dst, r in files():
        out = build(kind, d, src, dst, r); p = path(d, dst); n += 1
        srcp = p.parent / src
        if check:
            ok = p.exists() and p.read_text() == out; bad += not ok
            print(("OK   " if ok else "DIFF ") + str(p.relative_to(ROOT)))
        else:
            num = Path(dst).name[:5]
            others = [q.name for q in p.parent.glob(num + "_*") if q.name != p.name]
            if others:
                raise SystemExit(f"number {num} already used in {p.parent}: {others}")
            if p.exists() and p.read_text() != out:
                raise SystemExit(f"{p} exists with different content")
            p.write_text(out); print("wrote", p.relative_to(ROOT))
        if diffs:
            diffs.mkdir(parents=True, exist_ok=True)
            dl = difflib.unified_diff(srcp.read_text().splitlines(True), out.splitlines(True),
                                      str(srcp.relative_to(ROOT)), str(p.relative_to(ROOT)), n=0)
            (diffs / f"{Path(d).name}__{Path(dst).name}.diff").write_text("".join(dl))
            for line in difflib.ndiff(strip_header(srcp.read_text()).splitlines(True), strip_header(out).splitlines(True)):
                if line[:2] not in ("- ", "+ "):
                    continue
                t = line[2:].strip()
                k = ("METHOD" if t.startswith("METHOD=") else "config" if t.startswith("export AUGLAB_PARAMS_GPU_JSON") else
                     "val_config" if t.startswith("export AUGLAB_VAL_PARAMS_GPU_JSON") else
                     "LOG_DIR" if t.startswith("LOG_DIR=") else
                     "chaos_t1in_epochs" if t.startswith("export NNUNET_NUM_EPOCHS") and r[0] == CHAOS_T1IN_FIX else
                     "chaos_t1in_single_pin_dropped" if re.match(r"export SINGLE_(FOLD|SLOT|GPU)=", t) and r[0] == CHAOS_T1IN_FIX and line[0] == "-" else
                     "usage_message" if t.startswith("RUN_ID=") and kind == "companion" else "OTHER:" + t)
                kinds.setdefault(f"{kind}:{k}", []).append(Path(dst).name)
    if diffs:
        with open(diffs / "SUMMARY.txt", "w") as fh:
            for k, v in sorted(kinds.items()):
                fh.write(f"{k}\t{len(v)} lines\t{len(set(v))} files\n")
        print((diffs / "SUMMARY.txt").read_text())
        if any(":OTHER:" in k for k in kinds):
            print("UNEXPECTED BODY DIFFERENCES"); bad += 1
    if "--roster" in sys.argv:
        write_roster(Path(sys.argv[sys.argv.index("--roster") + 1]))
    print(f"{n} wrappers {'checked' if check else 'generated'}"); sys.exit(1 if bad else 0)


def loader_default(d):
    c = (ROOT / T / d / f"5_scripts_{Path(d).name}" / "04_train/04_00_common.sh").read_text()
    g = lambda v: re.search(rf'^{v}="?(\d+)', c, re.M).group(1)
    return f"DA={g('DA_WORKERS_DEFAULT')}/compile={g('NNUNET_COMPILE_DEFAULT')}"


def write_roster(out):
    rows = ["setting\tdataset\ttraining_contrast\ttest_source\ttest_items\theadline_PA_Auglab_run(combined config)\tsrc_train_wrapper\t"
            "new_train_wrapper\tnew_predict_wrapper\tepochs\ttrainer\tloader_default"]
    own_items = {"brats": "t1n t1c t2w t2f", "chaos": "t1in t1out t2spir ct", "onh": "T1w T2w bold dwi_ap epi_ap gre_echo1_mag",
                 "openms": "flair t1w t2w", "tf2": "cbct", "ispy2": "t1wce t2w", "pelvic": "ct mri", "atlas": "t1w", "panseg": "t1wce t2w"}
    for r in S:
        key, ds, tc, tag, run, tsrc, tno, psrc, pno, tr, ep = r
        d = D[ds]; dn = Path(d).name; sd = f"{T}/{d}/5_scripts_{dn}"
        base = [key, dn, tc]; tail = [run, f"{sd}/04_train/{tsrc}", f"{sd}/04_train/{tno}_train_{tag}_{METHOD}.sh"]
        rows.append("\t".join(base + [dn + " (own)", own_items[ds]] + tail + [f"{sd}/05_predict/{pno}_predict_{tag}_{METHOD}.sh", ep, tr, loader_default(d)]))
        for comp, setting, psrc_, cno, stem, items in C:
            if setting == key:
                cd = D[comp]; cn = Path(cd).name
                rows.append("\t".join(base + [cn, items] + tail + [f"{T}/{cd}/5_scripts_{cn}/05_predict/{cno}_predict_{stem + '_' if stem else ''}{METHOD}.sh", ep, tr, loader_default(d)]))
        for comp, settings, items, why in NOT_WRAPPED:
            if key in settings.split():
                rows.append("\t".join(base + [comp, items] + tail + [f"(no wrapper) {why}", ep, tr, loader_default(d)]))
    out.write_text("\n".join(rows) + "\n"); print(f"roster: {len(rows) - 1} rows -> {out}")


if __name__ == "__main__":
    main()
