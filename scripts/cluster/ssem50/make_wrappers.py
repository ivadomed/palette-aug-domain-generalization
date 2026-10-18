#!/usr/bin/env python
"""Generate the train / predict wrappers of the "SynthSeg-EM (+Auglab; 50%)" competitor arm (ssem50, 2026-10-18).

METHOD auglabAug_synthseg_EM_train050_val000, AugLab config transform_params_gpu_default01-23_auglabAug_Synthseg_EM_train050.json
(scripts/cluster/ssem50/make_config.py: = PA's Auglab operator set + SynthSeg-EM at p 0.5). For each of the 19 training settings
of the paper (paper/scripts/meta_task_heatmap_paper.yaml minus healthy-spine-tum, which has no pipeline here) the setting's
EXISTING headline SynthSeg-EM wrappers are cloned:
  TRAIN    : the wrapper that produced the combined config's synthseg_EM run (plain <Dataset>AugLabDefault trainer, validation
             on clean real images, policy epochs, dataset loader defaults from 04_00_common.sh -- no wrapper sets DA_WORKERS /
             nnUNet_compile, checked below). Changes: METHOD, the config file name, the LOG_DIR leaf (so the new run never writes
             into the old synthseg_EM log dir) and the generated header. ONE setting deviates, by an explicit rule (EPOCH_FIX):
             chaos t1in -- see below.
  PREDICT  : the setting's own synthseg_EM predict wrapper. Changes: METHOD (contrast prefix of brats kept, e.g. t2w_<METHOD>) +
             header.
  COMPANION: the companion's synthseg_EM predict wrapper for every external test set the paper's combined config reads.
             Changes: METHOD, header, and the baked default RUN_ID ("${1:-<old synthseg_EM run>}") replaced by a REQUIRED argument
             ("${1:?...}") -- a clone with the old default would silently predict the OLD run under the new method name.
Not generated (by design, recorded in roster.tsv): amos / sliver07 for chaos (the paper reads their fov_crop metrics, produced by
00_commun_scripts/00_02_predict/fov_crop_predict_evaluate.sh from a RUNS_FILE -- written by launch.sh at record time, not by a
per-method wrapper; the 05_04 / 05_11 wrappers feed the LEGACY masked metrics); totalsegmri-pancreas / msd-pancreas (roster-driven:
they read pansegdata's roster_run_ids.tsv, ROSTER_ONLY=<METHOD>); amos-pancreas (not read by pansegdata_combined_01_results.yaml).

chaos t1in (EPOCH_FIX): the combined config's chaos t1in synthseg_EM run (chaos_t1in_synthseg_EM_train100_val000_20260611_120000) came
from 04_03_train_synthseg_EM.sh: 500-epoch default, stopped at epochs 293-296, no checkpoint_final, per-method loader override
(ALLOW_METHOD_LOADER_OVERRIDE DA_WORKERS=8 compile=1) and a 4-fold FOLD_SLOT_GPU map. Its later twin 04_36_train_t1in_synthseg_EM.sh
(same trainer, dataset, config; no loader override) defaults to 2500 epochs. Neither is the chaos policy (200). Source = 04_36
(dataset loader default), NNUNET_NUM_EPOCHS default forced to 200 (= chaos t2spir's synthseg_EM and the chaos policy).
  python scripts/cluster/ssem50/make_wrappers.py [--check] [--roster <tsv>] [--diffs <dir>]
     --check  : every generated file exists and equals the generated content (no writes)
     --roster : write the roster TSV (setting / companion rows)      --diffs : write one unified diff per generated file
"""
import difflib, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
T = "benchmark/02_tasks"
METHOD = "auglabAug_synthseg_EM_train050_val000"
OLDCFG = "transform_params_gpu_default01-23_Synthseg_EM.json"
NEWCFG = "transform_params_gpu_default01-23_auglabAug_Synthseg_EM_train050.json"
EPOCH_FIX = {"chaos_t1in": "200"}
D = {"brats": "brain_tumor/brats2024-glioma", "chaos": "abdomen_healthy/chaos", "onh": "brain_healthy/on-harmony",
     "openms": "brain_ms/open-ms", "tf2": "mandible_healthy/toothfairy2", "ispy2": "breast_cancer/ispy2",
     "pelvic": "pelvis_healthy/totalseg-pelvic", "atlas": "liver_hcc/atlas-liver-hcc", "panseg": "pancreas_disease/pansegdata",
     "duke": "breast_cancer/duke-breast-mri", "ispy1": "breast_cancer/ispy1", "acrin": "breast_cancer/acrin6698",
     "hanseg": "mandible_healthy/hanseg", "pddca": "mandible_healthy/pddca", "lld": "liver_hcc/lld-mmri-hcc",
     "liverhccseg": "liver_hcc/liverhccseg"}
# setting | task (paper yaml name) | ds key | TRAINING_CONTRAST | name tag | combined-config synthseg_EM run | train src | new train no
#   | predict src | new predict no | cluster | trainer | epochs (wrapper default after EPOCH_FIX) | loader default (04_00_common.sh)
S = [
 ("brats_t1n", "brats2024-glioma", "brats", "t1n", "t1n", "brats2024-glioma_t1n_synthseg_EM_20260622_044535", "04_22_train_t1n_synthseg_EM.sh", "04_62", "05_08_predict_synthseg_EM.sh", "05_83", "tamia"),
 ("brats_t2w", "brats2024-glioma", "brats", "t2w", "t2w", "brats2024-glioma_t2w_synthseg_EM_20260620_125354", "04_16_train_t2w_synthseg_EM.sh", "04_63", "05_11_predict_t2w_synthseg_EM.sh", "05_84", "tamia"),
 ("brats_t2f", "brats2024-glioma", "brats", "t2f", "t2f", "brats2024-glioma_t2f_synthseg_EM_20260917_113412", "04_52_train_t2f_synthseg_EM.sh", "04_64", "05_52_predict_t2f_synthseg_EM.sh", "05_85", "tamia"),
 ("brats_t1c", "brats2024-glioma", "brats", "t1c", "t1c", "brats2024-glioma_t1c_synthseg_EM_20260921_140000", "04_72_train_t1c_synthseg_EM.sh", "04_65", "05_72_predict_t1c_synthseg_EM.sh", "05_86", "tamia"),
 ("chaos_t1in", "chaos", "chaos", "t1in", "t1in", "chaos_t1in_synthseg_EM_train100_val000_20260611_120000", "04_36_train_t1in_synthseg_EM.sh", "04_83", "05_04_predict_synthseg_EM.sh", "05_69", "tamia"),
 ("chaos_t2spir", "chaos", "chaos", "t2spir", "t2spir", "chaos_t2spir_synthseg_EM_20260620_112357", "04_30_train_t2spir_synthseg_EM.sh", "04_84", "05_24_predict_t2spir_synthseg_EM.sh", "05_70", "tamia"),
 ("onh_T1w", "on-harmony", "onh", "T1w", "t1w", "on-harmony_T1w_synthseg_EM_20260623_192811", "04_09_train_t1w_synthseg_EM.sh", "04_79", "05_04_predict_t1w_synthseg_EM.sh", "05_91", "tamia"),
 ("onh_T2w", "on-harmony", "onh", "T2w", "t2w", "on-harmony_T2w_synthseg_EM_20260624_191329", "04_15_train_t2w_synthseg_EM.sh", "04_80", "05_10_predict_t2w_synthseg_EM.sh", "05_92", "tamia"),
 ("onh_dwi_ap", "on-harmony", "onh", "dwi_ap", "dwi_ap", "on-harmony_dwi_ap_synthseg_EM_20260921_203727", "04_37_train_dwi_ap_synthseg_EM.sh", "04_81", "05_16_predict_dwi_ap_synthseg_EM.sh", "05_93", "tamia"),
 ("openms_flair", "open-ms", "openms", "flair", "flair", "open-ms_flair_synthseg_EM_train100_val000_20260706_061243", "04_02_train_synthseg_EM.sh", "04_63", "05_02_predict_synthseg_EM.sh", "05_54", "tamia"),
 ("openms_t1w", "open-ms", "openms", "t1w", "t1w", "open-ms_t1w_synthseg_EM_train100_val000_20260708_083611", "04_16_train_t1w_synthseg_EM.sh", "04_64", "05_14_predict_t1w_synthseg_EM.sh", "05_55", "tamia"),
 ("tf2_cbct", "toothfairy2", "tf2", "cbct", "cbct", "toothfairy2_cbct_synthseg_EM_20260908_013027", "04_04_train_synthseg_EM.sh", "04_23", "05_05_predict_synthseg_EM.sh", "05_32", "tamia"),
 ("ispy2_t1wce", "ispy2", "ispy2", "t1wce", "t1wce", "ispy2_t1wce_synthseg_EM_20260905_163655", "04_04_train_t1wce_synthseg_EM.sh", "04_38", "05_40_predict_own_t1wce_synthseg_EM.sh", "05_50", "tamia"),
 ("ispy2_t2w", "ispy2", "ispy2", "t2w", "t2w", "ispy2_t2w_synthseg_EM_20260905_163655", "04_11_train_t2w_synthseg_EM.sh", "04_39", "05_26_predict_own_t2w_synthseg_EM.sh", "05_51", "tamia"),
 ("pelvic_ct", "totalseg-pelvic", "pelvic", "ct", "ct", "ct_synthseg_EM_20260916_072434", "04_04_train_ct_synthseg_EM.sh", "04_35", "05_05_predict_ct_synthseg_EM.sh", "05_24", "tamia"),
 ("pelvic_mri", "totalseg-pelvic", "pelvic", "mri", "mri", "mri_synthseg_EM_20260916_072453", "04_11_train_mri_synthseg_EM.sh", "04_36", "05_16_predict_mri_synthseg_EM.sh", "05_25", "tamia"),
 ("atlas_t1w", "atlas-liver-hcc", "atlas", "t1w", "t1w", "atlas-liver-hcc_t1w_synthseg_EM_20260804_062605", "04_04_train_synthseg_EM.sh", "04_19", "05_05_predict_synthseg_EM.sh", "05_36", "tamia"),
 ("panseg_t1wce", "pansegdata", "panseg", "t1wce", "t1wce", "pansegdata_t1wce_synthseg_EM_20261004_214021", "04_04_train_t1wce_synthseg_EM.sh", "04_35", "05_05_predict_t1wce_synthseg_EM.sh", "05_32", "vulcan"),
 ("panseg_t2w", "pansegdata", "panseg", "t2w", "t2w", "pansegdata_t2w_synthseg_EM_20261004_214552", "04_11_train_t2w_synthseg_EM.sh", "04_36", "05_16_predict_t2w_synthseg_EM.sh", "05_33", "vulcan"),
]
# companion | for setting | src predict wrapper | new no | name stem | items the combined config reads (column ids)
C = [
 ("duke", "ispy2_t1wce", "05_05_predict_ispy2_t1wce_synthseg_EM.sh", "05_23", "ispy2_t1wce", "t1wce_uniap precontrast_uniap"),
 ("duke", "ispy2_t2w", "05_11_predict_ispy2_t2w_synthseg_EM.sh", "05_24", "ispy2_t2w", "t1wce_uniap precontrast_uniap"),
 ("ispy1", "ispy2_t1wce", "05_05_predict_ispy2_t1wce_synthseg_EM.sh", "05_23", "ispy2_t1wce", "t1wce precontrast"),
 ("ispy1", "ispy2_t2w", "05_11_predict_ispy2_t2w_synthseg_EM.sh", "05_24", "ispy2_t2w", "t1wce precontrast"),
 ("acrin", "ispy2_t1wce", "05_05_predict_ispy2_t1wce_synthseg_EM.sh", "05_23", "ispy2_t1wce", "dwi_uniap"),
 ("acrin", "ispy2_t2w", "05_11_predict_ispy2_t2w_synthseg_EM.sh", "05_24", "ispy2_t2w", "dwi_uniap"),
 ("hanseg", "tf2_cbct", "05_05_predict_synthseg_EM.sh", "05_31", "", "ct mrt1 (S-I flipped _sif inputs, mandible_only_sif)"),
 ("pddca", "tf2_cbct", "05_05_predict_synthseg_EM.sh", "05_31", "", "ct (S-I flipped _sif inputs, mandible_only_sif)"),
 ("lld", "atlas_t1w", "05_05_predict_atlas_synthseg_EM.sh", "05_24", "atlas", "ce-pre ce-art ce-pv ce-del inphase outphase t2wi dwi"),
 ("liverhccseg", "atlas_t1w", "05_05_predict_atlas_synthseg_EM.sh", "05_24", "atlas", "ce-pre_T1w ce-art_T1w ce-pv_T1w ce-del_T1w"),
]
NOT_WRAPPED = [  # companion rows with no per-method wrapper (reason)
 ("amos", "chaos_t1in chaos_t2spir", "ct mri(->t2w)", "fov_crop: RUNS_FILE for fov_crop_predict_evaluate.sh via 06_36_fov_crop_ladder_rungs.sh (written by launch.sh record)"),
 ("sliver07", "chaos_t1in chaos_t2spir", "ct(->sliver07_ct_liver)", "fov_crop: same RUNS_FILE / 06_36 call"),
 ("totalsegmri-pancreas", "panseg_t1wce panseg_t2w", "t1gre t2like", "roster-driven: 05_02_run_all_predict.sh with ROSTER_ONLY=<METHOD> reads pansegdata's pin file"),
 ("msd-pancreas", "panseg_t1wce panseg_t2w", "ct", "roster-driven: 05_02_run_all_predict.sh with ROSTER_ONLY=<METHOD>"),
 ("amos-pancreas", "panseg_t1wce panseg_t2w", "-", "NOT read by pansegdata_combined_01_results.yaml -> skipped"),
]
HDR = """#!/usr/bin/env bash
# GENERATED by scripts/cluster/ssem50/make_wrappers.py from {src} (do not hand-edit).
# "SynthSeg-EM (+Auglab; 50%)" competitor arm (ssem50): METHOD {meth}, {what}
# Config {cfg} = PA's Auglab operator set + SynthSeg-EM at probability 0.5 (scripts/cluster/ssem50/make_config.py).
"""
WHAT_TRAIN = ("the setting's headline synthseg_EM training with ONLY METHOD, the config file and the LOG_DIR leaf changed{fix}.\n"
              "# Trainer (plain AugLabDefault, validation on clean real images), epochs, folds and loader settings (dataset defaults,\n"
              "# 04_00_common.sh) as in the source. Usage: bash {dst} [RUN_ID]   (RUN_ID to resume)")
WHAT_PRED = "predict with the arm's models; only METHOD changed vs the source. Usage: same arguments as {src}"
WHAT_COMP = ("predict this companion test set with the {setting}-trained arm; only METHOD changed and the RUN_ID is REQUIRED\n"
             "# (the source baked the old synthseg_EM run as default). Usage: bash {dst} <RUN_ID> [FOLD] [ITEMS...]")


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


def new_method_value(old):   # keep a contrast prefix (brats: t2w_synthseg_EM -> t2w_<METHOD>)
    m = re.fullmatch(r"(.*?)synthseg_EM(?:_train100_val000)?", old)
    if not m:
        raise SystemExit(f"unexpected source METHOD {old!r}")
    return m.group(1) + METHOD


def files():
    """yield (kind, ds dir, src name, dst name, setting row)"""
    for r in S:
        key, task, ds, tc, tag, run, tsrc, tno, psrc, pno, cl = r
        yield "train", D[ds], tsrc, f"04_train/{tno}_train_{tag}_{METHOD}.sh", r
        yield "predict", D[ds], psrc, f"05_predict/{pno}_predict_{tag}_{METHOD}.sh", r
    for comp, setting, psrc, pno, stem, items in C:
        r = next(x for x in S if x[0] == setting)
        yield "companion", D[comp], psrc, f"05_predict/{pno}_predict_{stem + '_' if stem else ''}{METHOD}.sh", r


def build(kind, d, src, dst, r):
    key = r[0]
    text = (ROOT / T / d / f"5_scripts_{Path(d).name}" / ("04_train" if kind == "train" else "05_predict") / src).read_text()
    body = strip_header(text)
    old = re.search(r'^METHOD="([^"]*)"', body, re.M).group(1)
    body = sub1(body, r'^METHOD="[^"]*"', f'METHOD="{new_method_value(old)}"', f"{dst} METHOD")
    if kind == "train":
        body = sub1(body, re.escape(OLDCFG), NEWCFG, f"{dst} config")
        body = sub1(body, r'^(LOG_DIR="[^"]*?)synthseg_EM("?)$', r'\g<1>' + METHOD + r'\g<2>', f"{dst} LOG_DIR")
        if re.search(r"^\s*(export\s+)?(DA_WORKERS|nnUNet_compile|ALLOW_METHOD_LOADER_OVERRIDE|FOLD_SLOT_GPU)=", body, re.M):
            raise SystemExit(f"{src}: sets loader / slot variables; the arm must use the dataset defaults")
        if "AUGLAB_VAL_PARAMS_GPU_JSON" in body and not re.search(r'^export AUGLAB_VAL_PARAMS_GPU_JSON=""', body, re.M):
            raise SystemExit(f"{src}: synthetic validation config set; the arm validates on clean real images")
        fix = ""
        if key in EPOCH_FIX:
            body = sub1(body, r'^export NNUNET_NUM_EPOCHS="\$\{NNUNET_NUM_EPOCHS:-\d+\}"$',
                        f'export NNUNET_NUM_EPOCHS="${{NNUNET_NUM_EPOCHS:-{EPOCH_FIX[key]}}}"   '
                        f'# ssem50 EPOCH_FIX: chaos policy {EPOCH_FIX[key]} (source default differs, see make_wrappers.py)',
                        f"{dst} epochs")
            fix = f", plus the epoch default forced to the chaos policy {EPOCH_FIX[key]} (EPOCH_FIX in make_wrappers.py)"
        what = WHAT_TRAIN.format(fix=fix, dst=Path(dst).name)
    elif kind == "predict":
        what = WHAT_PRED.format(src=src)
    else:
        if re.search(r'^RUN_ID="\$\{1:-', body, re.M):
            body = sub1(body, r'^RUN_ID="\$\{1:-[^}]*\}"$', 'RUN_ID="${1:?usage: <RUN_ID of the ssem50 arm> [FOLD] [ITEMS...]}"', f"{dst} RUN_ID")
        what = WHAT_COMP.format(setting=key, dst=Path(dst).name)
    return HDR.format(src=src, meth=METHOD, what=what, cfg=NEWCFG) + "\n" + body


def path(kind, d, dst):
    return ROOT / T / d / f"5_scripts_{Path(d).name}" / dst


def main():
    check = "--check" in sys.argv
    diffs = Path(sys.argv[sys.argv.index("--diffs") + 1]) if "--diffs" in sys.argv else None
    bad = n = 0
    for kind, d, src, dst, r in files():
        out = build(kind, d, src, dst, r); p = path(kind, d, dst); n += 1
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
    if "--roster" in sys.argv:
        write_roster(Path(sys.argv[sys.argv.index("--roster") + 1]))
    print(f"{n} wrappers {'checked' if check else 'generated'}"); sys.exit(1 if bad else 0)


def loader_default(d):
    c = (ROOT / T / d / f"5_scripts_{Path(d).name}" / "04_train/04_00_common.sh").read_text()
    g = lambda v: re.search(rf'^{v}="?(\d+)', c, re.M).group(1)
    return f"DA={g('DA_WORKERS_DEFAULT')}/compile={g('NNUNET_COMPILE_DEFAULT')}"


def write_roster(out):
    rows = ["task\tdataset\tsetting\ttraining_contrast\ttest_source\ttest_items\tsource_synthseg_EM_run(combined config)\t"
            "src_train_wrapper\tnew_train_wrapper\tnew_predict_wrapper\tepochs\ttrainer\tloader_default\tcluster"]
    own_items = {"brats": "t1n t1c t2w t2f", "chaos": "t1in t1out t2spir ct", "onh": "T1w T2w bold dwi_ap epi_ap gre_echo1_mag",
                 "openms": "flair t1w t2w", "tf2": "cbct", "ispy2": "t1wce t2w", "pelvic": "ct mri",
                 "atlas": "t1w (atlas_t1w)", "panseg": "t1wce t2w"}
    for r in S:
        key, task, ds, tc, tag, run, tsrc, tno, psrc, pno, cl = r
        d = D[ds]; dn = Path(d).name
        tw = (ROOT / T / d / f"5_scripts_{dn}/04_train" / tsrc).read_text()
        ep = EPOCH_FIX.get(key) or re.search(r'NNUNET_NUM_EPOCHS(?:="\$\{NNUNET_NUM_EPOCHS:-|=)(\d+)', tw).group(1)
        tr = re.search(r'^TRAINER="([^"]*)"', tw, re.M).group(1)
        nt = f"{T}/{d}/5_scripts_{dn}/04_train/{tno}_train_{tag}_{METHOD}.sh"
        npw = f"{T}/{d}/5_scripts_{dn}/05_predict/{pno}_predict_{tag}_{METHOD}.sh"
        base = [task, dn, key, tc]
        tail = [run, f"{T}/{d}/5_scripts_{dn}/04_train/{tsrc}", nt]
        rows.append("\t".join(base + [dn + " (own)", own_items[ds]] + tail + [npw, ep, tr, loader_default(d), r[10]]))
        for comp, setting, psrc, cno, stem, items in C:
            if setting != key:
                continue
            cd = D[comp]; cn = Path(cd).name
            cw = f"{T}/{cd}/5_scripts_{cn}/05_predict/{cno}_predict_{stem + '_' if stem else ''}{METHOD}.sh"
            rows.append("\t".join(base + [cn, items] + tail + [cw, ep, tr, loader_default(d), "tamia" if comp in ("hanseg", "pddca") else "vulcan|tamia"]))
        for comp, settings, items, why in NOT_WRAPPED:
            if key in settings.split():
                rows.append("\t".join(base + [comp, items] + tail + [f"(no wrapper) {why}", ep, tr, loader_default(d), "-"]))
    out.write_text("\n".join(rows) + "\n"); print(f"roster: {len(rows) - 1} rows -> {out}")


if __name__ == "__main__":
    main()
