#!/usr/bin/env python
"""Generate the train / predict wrappers of the SLAug competitor arm (slaug, METHOD auglab_slaug, category auglab).
Clone of scripts/cluster/ginipa/make_wrappers.py (2026-11-13); wrapper numbers = the first number free on Vulcan, TamIA and Killarney.

Sources = each setting's SRCSM wrappers (the latest single-config AugLab competitor; same trainer family <DS>AugLabDefault, same
epochs, folds, loader defaults, results tree). Every change is a substitution that must apply exactly once:
  TRAIN     METHOD srcsm -> auglab_slaug; TRAINER <DS>[<C>]AugLabDefault -> <DS>[<C>]SLAug (scripts/cluster/slaug/make_trainers.py);
            LOG_DIR leaf srcsm -> slaug; the AugLab config lines (AUGLAB_CONFIGS_DIR / _AUGLAB_CONFIGS + AUGLAB_PARAMS_GPU_JSON +
            AUGLAB_VAL_PARAMS_GPU_JSON) -> the PIN BLOCK below; header. Epochs are asserted equal to the policy (= the competitors' runs).
  PREDICT   METHOD, TRAINER, PIN BLOCK (pin first on PYTHONPATH: nnU-Net imports the trainer class to build the network), header.
  COMPANION the companion's SRCSM predict wrapper: METHOD, TRAINER, PIN BLOCK, header, and a baked default RUN_ID ("${1:-<old>}")
            replaced by a REQUIRED argument.
PIN BLOCK: `source <repo>/scripts/cluster/slaug/pin_env.sh` (cluster's pin from PIN.conf; refuses unless the pin on disk is at the
pinned commit, clean, config md5 as pinned) then AUGLAB_PARAMS_GPU_JSON=<pin>/auglab/configs/<config>, no val config. The shared
AugLab trees (sub-workspaces/) are never referenced. Training jobs do NOT inherit PYTHONPATH (train_common.sh single-quotes it), so the
launcher (slaug.sh record) inserts the pin into each recorded fold command and verifies it; prediction commands bake the caller's
PYTHONPATH at record time, so the predict wrappers' export suffices.
Not generated (no per-method wrapper, as for the flatauglab arm): amos / sliver07 (chaos fov_crop RUNS_FILE in post_run.sh),
totalsegmri-pancreas / msd-pancreas (roster shim with an explicit RUN_ID in post_run.sh), amos-pancreas (not read by the combined config).
Spine (healthy-spine-tum): collaborator-run, metrics only, no nnU-Net data in this repo -> not generated.
  python scripts/cluster/slaug/make_wrappers.py [--check] [--roster <tsv>]
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
T = "benchmark/02_tasks"
METHOD = "auglab_slaug"
OLDM = "srcsm"
D = {"brats": "brain_tumor/brats2024-glioma", "chaos": "abdomen_healthy/chaos", "onh": "brain_healthy/on-harmony",
     "openms": "brain_ms/open-ms", "tf2": "mandible_healthy/toothfairy2", "ispy2": "breast_cancer/ispy2",
     "pelvic": "pelvis_healthy/totalseg-pelvic", "atlas": "liver_hcc/atlas-liver-hcc", "panseg": "pancreas_disease/pansegdata",
     "duke": "breast_cancer/duke-breast-mri", "ispy1": "breast_cancer/ispy1", "acrin": "breast_cancer/acrin6698",
     "hanseg": "mandible_healthy/hanseg", "pddca": "mandible_healthy/pddca", "lld": "liver_hcc/lld-mmri-hcc",
     "liverhccseg": "liver_hcc/liverhccseg"}
# setting | ds | TRAINING_CONTRAST | name tag | srcsm train src | new no | srcsm predict src | new no | expected SLAug trainer | epochs
S = [
 ("brats_t1n", "brats", "t1n", "t1n", "04_26_train_t1n_srcsm.sh", "04_01", "05_18_predict_t1n_srcsm.sh", "05_32", "nnUNetTrainerBraTS2024GliomaSLAug", "2500"),
 ("brats_t2w", "brats", "t2w", "t2w", "04_27_train_t2w_srcsm.sh", "04_02", "05_19_predict_t2w_srcsm.sh", "05_33", "nnUNetTrainerBraTS2024GliomaT2wSLAug", "2500"),
 ("brats_t2f", "brats", "t2f", "t2f", "04_54_train_t2f_srcsm.sh", "04_04", "05_54_predict_t2f_srcsm.sh", "05_34", "nnUNetTrainerBraTS2024GliomaT2fSLAug", "2500"),
 ("brats_t1c", "brats", "t1c", "t1c", "04_74_train_t1c_srcsm.sh", "04_10", "05_74_predict_t1c_srcsm.sh", "05_35", "nnUNetTrainerBraTS2024GliomaT1cSLAug", "2500"),
 ("chaos_t1in", "chaos", "t1in", "t1in", "04_42_train_t1in_srcsm.sh", "04_94", "05_32_predict_srcsm.sh", "05_80", "nnUNetTrainerCHAOSSLAug", "200"),
 ("chaos_t2spir", "chaos", "t2spir", "t2spir", "04_43_train_t2spir_srcsm.sh", "04_95", "05_33_predict_t2spir_srcsm.sh", "05_81", "nnUNetTrainerCHAOSSLAug", "200"),
 ("onh_T1w", "onh", "T1w", "t1w", "04_20_train_t1w_srcsm.sh", "04_94", "05_07_predict_t1w_srcsm.sh", "05_30", "nnUNetTrainerOnHarmonySLAug", "2000"),
 ("onh_T2w", "onh", "T2w", "t2w", "04_21_train_t2w_srcsm.sh", "04_95", "05_13_predict_t2w_srcsm.sh", "05_31", "nnUNetTrainerOnHarmonySLAug", "2000"),
 ("onh_dwi_ap", "onh", "dwi_ap", "dwi_ap", "04_40_train_dwi_ap_srcsm.sh", "04_96", "05_19_predict_dwi_ap_srcsm.sh", "05_32", "nnUNetTrainerOnHarmonySLAug", "2000"),
 ("openms_flair", "openms", "flair", "flair", "04_20_train_srcsm.sh", "04_73", "05_18_predict_srcsm.sh", "05_64", "nnUNetTrainerOpenMSSLAug", "2000"),
 ("openms_t1w", "openms", "t1w", "t1w", "04_21_train_t1w_srcsm.sh", "04_74", "05_19_predict_t1w_srcsm.sh", "05_65", "nnUNetTrainerOpenMSSLAug", "2000"),
 ("tf2_cbct", "tf2", "cbct", "cbct", "04_05_train_srcsm.sh", "04_28", "05_06_predict_srcsm.sh", "05_37", "nnUNetTrainerToothFairy2SLAug", "1000"),
 ("ispy2_t1wce", "ispy2", "t1wce", "t1wce", "04_05_train_t1wce_srcsm.sh", "04_48", "05_24_predict_own_t1wce_srcsm.sh", "05_60", "nnUNetTrainerISPY2SLAug", "1000"),
 ("ispy2_t2w", "ispy2", "t2w", "t2w", "04_12_train_t2w_srcsm.sh", "04_49", "05_27_predict_own_t2w_srcsm.sh", "05_61", "nnUNetTrainerISPY2SLAug", "1000"),
 ("pelvic_ct", "pelvic", "ct", "ct", "04_05_train_ct_srcsm.sh", "04_45", "05_06_predict_ct_srcsm.sh", "05_34", "nnUNetTrainerTotalsegPelvicSLAug", "200"),
 ("pelvic_mri", "pelvic", "mri", "mri", "04_12_train_mri_srcsm.sh", "04_46", "05_17_predict_mri_srcsm.sh", "05_35", "nnUNetTrainerTotalsegPelvicSLAug", "200"),
 ("atlas_t1w", "atlas", "t1w", "t1w", "04_05_train_srcsm.sh", "04_24", "05_06_predict_srcsm.sh", "05_41", "nnUNetTrainerAtlasLiverHCCSLAug", "2000"),
 ("panseg_t1wce", "panseg", "t1wce", "t1wce", "04_05_train_t1wce_srcsm.sh", "04_46", "05_06_predict_t1wce_srcsm.sh", "05_43", "nnUNetTrainerPANSEGDATASLAug", "2000"),
 ("panseg_t2w", "panseg", "t2w", "t2w", "04_12_train_t2w_srcsm.sh", "04_47", "05_17_predict_t2w_srcsm.sh", "05_44", "nnUNetTrainerPANSEGDATASLAug", "2000"),
]
# companion | for setting | srcsm predict src | new no | name stem | items the combined config reads
C = [
 ("duke", "ispy2_t1wce", "05_06_predict_ispy2_t1wce_srcsm.sh", "05_33", "ispy2_t1wce", "t1wce_uniap precontrast_uniap"),
 ("duke", "ispy2_t2w", "05_12_predict_ispy2_t2w_srcsm.sh", "05_34", "ispy2_t2w", "t1wce_uniap precontrast_uniap"),
 ("ispy1", "ispy2_t1wce", "05_06_predict_ispy2_t1wce_srcsm.sh", "05_33", "ispy2_t1wce", "t1wce precontrast"),
 ("ispy1", "ispy2_t2w", "05_12_predict_ispy2_t2w_srcsm.sh", "05_34", "ispy2_t2w", "t1wce precontrast"),
 ("acrin", "ispy2_t1wce", "05_06_predict_ispy2_t1wce_srcsm.sh", "05_33", "ispy2_t1wce", "dwi_uniap"),
 ("acrin", "ispy2_t2w", "05_12_predict_ispy2_t2w_srcsm.sh", "05_34", "ispy2_t2w", "dwi_uniap"),
 ("hanseg", "tf2_cbct", "05_06_predict_srcsm.sh", "05_36", "", "ct mrt1 (S-I flipped _sif inputs, mandible_only_sif)"),
 ("pddca", "tf2_cbct", "05_06_predict_srcsm.sh", "05_36", "", "ct (S-I flipped _sif inputs, mandible_only_sif)"),
 ("lld", "atlas_t1w", "05_06_predict_atlas_srcsm.sh", "05_29", "atlas", "ce-pre ce-art ce-pv ce-del inphase outphase t2wi dwi"),
 ("liverhccseg", "atlas_t1w", "05_06_predict_atlas_srcsm.sh", "05_29", "atlas", "ce-pre_T1w ce-art_T1w ce-pv_T1w ce-del_T1w"),
]
NOT_WRAPPED = [
 ("amos", "chaos_t1in chaos_t2spir", "ct mri", "fov_crop: RUNS_FILE for fov_crop_predict_evaluate.sh (post_run.sh)"),
 ("sliver07", "chaos_t1in chaos_t2spir", "ct", "fov_crop: same RUNS_FILE (post_run.sh)"),
 ("totalsegmri-pancreas", "panseg_t1wce panseg_t2w", "t1gre t2like", "pansegdata roster shim with explicit RUN_ID (post_run.sh)"),
 ("msd-pancreas", "panseg_t1wce panseg_t2w", "ct", "pansegdata roster shim with explicit RUN_ID (post_run.sh)"),
]
PIN_SRC = 'source "$(dirname "$0")/../../../../../../scripts/cluster/slaug/pin_env.sh"   # SLAUG_PIN/_COMMIT/_CONFIG (refuses on mismatch)'
PIN_TRAIN = (PIN_SRC + "\n"
             'export AUGLAB_PARAMS_GPU_JSON="${SLAUG_PIN}/auglab/configs/${SLAUG_CONFIG}"\n'
             'export AUGLAB_VAL_PARAMS_GPU_JSON=""\n')
PIN_PRED = (PIN_SRC + "\n"
            'export PYTHONPATH="${SLAUG_PIN}:${PYTHONPATH:-}"   # nnU-Net imports the SLAug trainer class (pin only) to build the network\n')
HDR = """#!/usr/bin/env bash
# GENERATED by scripts/cluster/slaug/make_wrappers.py from {src} (do not hand-edit).
# SLAug competitor arm (slaug): METHOD {meth}, {what}
# AugLab code + config: the SLAug pin named by scripts/cluster/slaug/PIN.conf (checked by pin_env.sh), never sub-workspaces/.
"""
WHAT_TRAIN = ("the setting's SRCSM training with ONLY METHOD, TRAINER ({trainer}), the AugLab config\n"
              "# lines (-> the pin) and the LOG_DIR leaf changed. Epochs, folds and loader settings (dataset defaults, 04_00_common.sh) as in\n"
              "# the source. Training jobs must be recorded + submitted through scripts/cluster/slaug/slaug.sh, which puts the pin first on\n"
              "# the fold command's PYTHONPATH (train_common.sh does not pass PYTHONPATH on). Usage: bash {dst} [RUN_ID]")
WHAT_PRED = "predict with the arm's models; METHOD, TRAINER and the pin block changed vs the source. Usage: same arguments as {src}"
WHAT_COMP = ("predict this companion test set with the {setting}-trained arm; METHOD, TRAINER, the pin block changed and\n"
             "# the RUN_ID is REQUIRED. Usage: bash {dst} <RUN_ID> [FOLD] [ITEMS...]")


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


def new_trainer(old, expect):
    m = re.fullmatch(r"(nnUNetTrainer\w*?)AugLabDefault", old)
    if not m or m.group(1) + "SLAug" != expect:
        raise SystemExit(f"source TRAINER {old!r} does not map to the expected {expect!r}")
    return expect


def files():
    for r in S:
        key, ds, tc, tag, tsrc, tno, psrc, pno, tr, ep = r
        yield "train", D[ds], tsrc, f"04_train/{tno}_train_{tag}_{METHOD}.sh", r
        yield "predict", D[ds], psrc, f"05_predict/{pno}_predict_{tag}_{METHOD}.sh", r
    for comp, setting, psrc, pno, stem, items in C:
        r = next(x for x in S if x[0] == setting)
        yield "companion", D[comp], psrc, f"05_predict/{pno}_predict_{stem + '_' if stem else ''}{METHOD}.sh", r


def build(kind, d, src, dst, r):
    key, tr, ep = r[0], r[8], r[9]
    sub = "04_train" if kind == "train" else "05_predict"
    text = (ROOT / T / d / f"5_scripts_{Path(d).name}" / sub / src).read_text()
    body = strip_header(text)
    # keep a contrast prefix (brats t2f / t1c predict wrappers: METHOD="t2f_srcsm" -> "t2f_auglab_slaug"), as the source convention
    body = sub1(body, r'^METHOD="((?:t1n_|t2w_|t2f_|t1c_)?)' + OLDM + '"', r'METHOD="\g<1>' + METHOD + '"', f"{dst} METHOD")
    otr = re.search(r'^TRAINER="([^"]*)"', body, re.M).group(1)
    body = sub1(body, r'^TRAINER="[^"]*"', f'TRAINER="{new_trainer(otr, tr)}"', f"{dst} TRAINER")
    if kind == "train":
        body = sub1(body, r'^(_?AUGLAB_CONFIGS(?:_DIR)?)=.*\n', "", f"{dst} drop configs-dir line")
        body = sub1(body, r'^export AUGLAB_PARAMS_GPU_JSON=.*srcsm_semrandconv\.json"\n', "@@PIN@@\n", f"{dst} config line")
        body = re.sub(r'^export AUGLAB_VAL_PARAMS_GPU_JSON=""\n', "", body, flags=re.M)
        body = body.replace("@@PIN@@\n", PIN_TRAIN)
        body = sub1(body, r'^(LOG_DIR="[^"]*?)srcsm(")$', r"\g<1>slaug\g<2>", f"{dst} LOG_DIR")
        if re.search(r"^\s*(export\s+)?(DA_WORKERS|nnUNet_compile|ALLOW_METHOD_LOADER_OVERRIDE|FOLD_SLOT_GPU|SINGLE_FOLD|SINGLE_SLOT|SINGLE_GPU|TRAIN_FOLDS)=", body, re.M):
            raise SystemExit(f"{src}: sets loader / slot / fold variables; the arm must use the dataset defaults and folds 0 1 2")
        m = re.search(r'NNUNET_NUM_EPOCHS(?:="\$\{NNUNET_NUM_EPOCHS:-|=)(\d+)', body)
        if not m or m.group(1) != ep:
            raise SystemExit(f"{dst}: epochs {m and m.group(1)} != policy {ep}")
        what = WHAT_TRAIN.format(dst=Path(dst).name, trainer=tr)
    else:
        # the pin block goes right before the (last) `source ...05_01_predict*common.sh` line
        body = sub1(body, r'^(source "\$\(dirname "\$0"\)/05_[0-9][0-9]_predict\w*common\.sh".*)$', PIN_PRED.replace("\\", "\\\\") + r"\g<1>",
                    f"{dst} pin block")
        if kind == "companion":
            if re.search(r'^RUN_ID="\$\{1:-', body, re.M):
                body = sub1(body, r'^RUN_ID="\$\{1:-[^}]*\}"$', 'RUN_ID="${1:?usage: <RUN_ID of the slaug arm> [FOLD] [ITEMS...]}"', f"{dst} RUN_ID")
            what = WHAT_COMP.format(setting=key, dst=Path(dst).name)
        else:
            what = WHAT_PRED.format(src=src)
    if re.search(r"\bsrcsm\b", body) or "AugLabDefault" in body or "sub-workspaces" in body:
        raise SystemExit(f"{dst}: a source method / trainer / AugLab path survived:\n{body}")
    return HDR.format(src=src, meth=METHOD, what=what) + "\n" + body


def path(d, dst):
    return ROOT / T / d / f"5_scripts_{Path(d).name}" / dst


def main():
    check = "--check" in sys.argv
    bad = n = 0
    for kind, d, src, dst, r in files():
        out = build(kind, d, src, dst, r)
        p = path(d, dst)
        n += 1
        if check:
            ok = p.exists() and p.read_text() == out
            bad += not ok
            print(("OK   " if ok else "DIFF ") + str(p.relative_to(ROOT)))
        else:
            num = Path(dst).name[:5]
            others = [q.name for q in p.parent.glob(num + "_*") if q.name != p.name]
            if others:
                raise SystemExit(f"number {num} already used in {p.parent}: {others}")
            if p.exists() and p.read_text() != out:
                raise SystemExit(f"{p} exists with different content")
            p.write_text(out)
            print("wrote", p.relative_to(ROOT))
    if "--roster" in sys.argv:
        write_roster(Path(sys.argv[sys.argv.index("--roster") + 1]))
    print(f"{n} wrappers {'checked' if check else 'generated'}")
    sys.exit(1 if bad else 0)


def loader_default(d):
    c = (ROOT / T / d / f"5_scripts_{Path(d).name}" / "04_train/04_00_common.sh").read_text()
    def g(v):
        return re.search(rf'^{v}="?(\d+)', c, re.M).group(1)
    return f"{g('DA_WORKERS_DEFAULT')}/{g('NNUNET_COMPILE_DEFAULT')}"


def write_roster(out):
    rows = ["setting\tdataset\ttraining_contrast\ttest_source\ttest_items\tsrc_train_wrapper\tnew_train_wrapper\tnew_predict_wrapper\tepochs\ttrainer\tloader_default"]
    own_items = {"brats": "t1n t1c t2w t2f", "chaos": "t1in t1out t2spir ct", "onh": "T1w T2w bold dwi_ap epi_ap gre_echo1_mag",
                 "openms": "flair t1w t2w", "tf2": "cbct", "ispy2": "t1wce t2w", "pelvic": "ct mri", "atlas": "t1w", "panseg": "t1wce t2w"}
    for r in S:
        key, ds, tc, tag, tsrc, tno, psrc, pno, tr, ep = r
        d = D[ds]
        dn = Path(d).name
        sd = f"{T}/{d}/5_scripts_{dn}"
        base = [key, dn, tc]
        tail = [f"{sd}/04_train/{tsrc}", f"{sd}/04_train/{tno}_train_{tag}_{METHOD}.sh"]
        rows.append("\t".join(base + [dn + " (own)", own_items[ds]] + tail + [f"{sd}/05_predict/{pno}_predict_{tag}_{METHOD}.sh", ep, tr, loader_default(d)]))
        for comp, setting, _p, cno, stem, items in C:
            if setting == key:
                cd = D[comp]
                cn = Path(cd).name
                rows.append("\t".join(base + [cn, items] + tail + [f"{T}/{cd}/5_scripts_{cn}/05_predict/{cno}_predict_{stem + '_' if stem else ''}{METHOD}.sh", ep, tr, loader_default(d)]))
        for comp, settings, items, why in NOT_WRAPPED:
            if key in settings.split():
                rows.append("\t".join(base + [comp, items] + tail + [f"(no wrapper) {why}", ep, tr, loader_default(d)]))
    out.write_text("\n".join(rows) + "\n")
    print(f"roster: {len(rows) - 1} rows -> {out}")


if __name__ == "__main__":
    main()
