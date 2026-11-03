#!/usr/bin/env python3
"""Label-level difficulty vs target type (preregistered in README.md of this folder).

Two subcommands:
  compute --setting K --type appearance|interface --script <ladder script> --raw-dataset <nnUNet raw DatasetXXX> --out-dir D
      Captures the setting's own ladder call (rung keys, in-domain, OOD contrasts / sources / groups) by intercepting
      its run_ladder* entry point, then re-runs the SHARED engine (ladder_ood_common.run_ladder, flat rung inserted by
      the engine) once per foreground label with the engine's CSV reader restricted to that label. Every per-label
      number therefore uses the engine's own fold cap, pooling, patient merging and Wilcoxon. Also: the unrestricted
      rerun (sanity check against the on-disk ladder_series.json), baseline in-domain Dice per label, label volume
      from labelsTr, and the hardest-tercile label set (settings with >= 8 labels). Writes D/<K>.json.
  analyze --out-dir D
      Reads every D/*.json, writes label_table.csv, setting_tests.csv, SUMMARY.md and gain_vs_difficulty.png.

Nothing is written into the repo tree except under outputs/ (gitignored); engine outputs go to a temp dir.
"""
from __future__ import annotations

import argparse
import contextlib
import csv as _csv
import io
import itertools
import json
import math
import re
import runpy
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_03_evaluate"))
import ladder_ood_common as L  # noqa: E402

# ── label-restricted CSV reader for the engine (only eval_all.csv files go through L.csv) ──────────────────────────
_ROWS: dict = {}
_FILTER: set | None = None


class _FilteredDictReader:
    def __init__(self, f, *a, **k):
        path = getattr(f, "name", None)
        if path is None or path not in _ROWS:
            rows = list(_csv.DictReader(f, *a, **k))
            if path is not None:
                _ROWS[path] = rows
        else:
            rows = _ROWS[path]
        self._it = iter(rows if _FILTER is None else [r for r in rows if r.get("label") in _FILTER])

    def __iter__(self):
        return self._it


L.csv = types.SimpleNamespace(DictReader=_FilteredDictReader)
L._write_combined_png = lambda *a, **k: None
L._write_per_contrast_png = lambda *a, **k: None
_REAL_RUN_LADDER = L.run_ladder


def capture(script: str) -> dict:
    cap = {}

    def stub(name):
        def f(**kw):
            cap["fn"], cap["kw"] = name, kw
        return f

    saved = {n: getattr(L, n) for n in ("run_ladder", "run_ladder_cross_dataset", "run_ladder_grouped")}
    for n in saved:
        setattr(L, n, stub(n))
    argv = sys.argv
    try:
        sys.argv = [script]
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(script, run_name="__main__")
    finally:
        sys.argv = argv
        for n, f in saved.items():
            setattr(L, n, f)
    if cap.get("fn") != "run_ladder":
        raise SystemExit(f"{script}: expected a run_ladder call, got {cap.get('fn')}")
    return cap["kw"]


def run_engine(kw: dict, labels: set | None, tmp: Path) -> dict:
    """The engine's ladder for `labels` (None = all). Grouped ladders: a group member with no case of these labels
    (e.g. CHAOS's own CT holds only the liver) is dropped from its group (it contributes nothing), empty groups too."""
    global _FILTER
    kw = {**kw, "ablations_root": tmp, "combined_png": False}
    groups = kw.get("ood_groups")
    for _ in range(10):
        _FILTER = labels
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return _REAL_RUN_LADDER(**kw)
        except ValueError as e:
            m = re.search(r"never-found=\[(.*?)\]", str(e))
            if not groups or not m or not m.group(1):
                raise
            gone = set(re.findall(r"'([^']+)'", m.group(1)))
            groups = {g: [x for x in mem if x not in gone] for g, mem in groups.items()}
            groups = {g: mem for g, mem in groups.items() if mem}
            kw = {**kw, "ood_groups": groups}
        finally:
            _FILTER = None
    raise RuntimeError("could not resolve ood_groups")


def step(dump: dict) -> dict:
    labs = dump["labels"]
    fi = labs.index(L.FLAT_RUNG_LABEL)
    if L.FILL_SWAP_LABEL_HINT not in labs[fi + 1].lower():
        raise ValueError(f"rung after flat is {labs[fi + 1]!r}, not the real-fill rung")
    d = dump["dice"]
    sig = dump["fill_swap_significance"]["dice"]
    return {"noise": d[fi - 1], "flat": d[fi], "texture": d[fi + 1], "gain": d[fi + 1] - d[fi],
            "p": sig["pooled_p"], "n_patients": sig["n_cases"], "group_sizes": dump.get("group_sizes")}


def roots(kw):
    out = [(Path(kw["metrics_root"]), None)]
    for s in kw.get("extra_ood_sources") or []:
        out.append((Path(L._src_root(s)), s))
    return out


def label_volumes(raw: Path, n_max: int = 20) -> dict:
    import nibabel as nib
    ds = json.loads((raw / "dataset.json").read_text())
    name_of = {}
    for name, v in ds["labels"].items():
        if name == "background":
            continue
        for i in (v if isinstance(v, list) else [v]):
            name_of[int(i)] = name
    files = sorted((raw / "labelsTr").glob("*" + ds.get("file_ending", ".nii.gz")))[:n_max]
    vols: dict = {n: [] for n in name_of.values()}
    for f in files:
        img = nib.load(str(f))
        vox_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
        cnt = np.bincount(np.asarray(img.dataobj).astype(np.int64).ravel())
        for i, n in name_of.items():
            if i < len(cnt) and cnt[i] > 0:
                vols[n].append(cnt[i] * vox_ml)
    return {n: (float(np.median(v)) if v else float("nan")) for n, v in vols.items()}, len(files)


def compute(a):
    kw = capture(a.script)
    rungs = kw["rungs"]
    out = {"setting": a.setting, "task": a.setting.split("_")[0], "type": a.type, "script": a.script,
           "in_domain": kw["in_domain"], "ood_contrasts": kw["ood_contrasts"],
           "ood_groups": kw.get("ood_groups"), "rung0": rungs[0][2]}
    tmp = Path(tempfile.mkdtemp(prefix=f"ld_{a.setting}_"))

    # sanity (i): unrestricted rerun vs on-disk ladder_series.json
    full = step(run_engine(kw, None, tmp))
    disk = json.loads((Path(kw["ablations_root"]) / "ladder_series.json").read_text())
    out["sanity"] = {"rerun": full, "on_disk": step(disk)}

    # label roster = labels in the flat/real runs' CSVs (all sources)
    flat_key = disk["run_keys"][disk["labels"].index(L.FLAT_RUNG_LABEL)]
    labels, nan_rows, tot_rows = set(), {}, {}
    for root, src in roots(kw):
        d = L.resolve_run_dir(root, L._src_key(src, flat_key))
        for fd in L.filter_fold_dirs(sorted(d.glob("fold*"))):
            p = fd / "eval_all.csv"
            if not p.exists():
                continue
            for r in _csv.DictReader(p.open()):
                lb = r["label"]
                labels.add(lb)
                tot_rows[lb] = tot_rows.get(lb, 0) + 1
                try:
                    ok = math.isfinite(float(r["dice"]))
                except ValueError:
                    ok = False
                nan_rows[lb] = nan_rows.get(lb, 0) + (0 if ok else 1)
    labels = sorted(labels)

    vols, n_vol = label_volumes(Path(a.raw_dataset))
    missing = [lb for lb in labels if lb not in vols]
    if missing:
        raise SystemExit(f"labels without a dataset.json entry: {missing}")
    out["n_volume_cases"] = n_vol

    base_dir = L.resolve_run_dir(Path(kw["metrics_root"]), rungs[0][2])
    per = {}
    global _FILTER
    for lb in labels:
        s = step(run_engine(kw, {lb}, tmp))
        _FILTER = {lb}
        try:
            cm = L.load_case_means(base_dir, "dice").get(kw["in_domain"], {})
        finally:
            _FILTER = None
        s["baseline_ind"] = float(np.mean(list(cm.values()))) * 100 if cm else float("nan")
        s["volume_ml"] = vols[lb]
        s["nan_rows_flat"], s["rows_flat"] = nan_rows.get(lb, 0), tot_rows.get(lb, 0)
        per[lb] = s
    out["labels"] = per
    out["label_mean_gain"] = float(np.nanmean([s["gain"] for s in per.values()]))

    if len(labels) >= 8:  # preregistered: hardest / easiest third by noise-fill (c) OOD Dice, ties by name
        k = math.ceil(len(labels) / 3)
        order = sorted(labels, key=lambda lb: (per[lb]["noise"], lb))
        out["terciles"] = {}
        for name, sel in (("hardest", order[:k]), ("easiest", order[-k:])):
            s = step(run_engine(kw, set(sel), tmp))
            s["labels"] = sel
            out["terciles"][name] = s
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.out_dir) / f"{a.setting}.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"{a.setting}: {len(labels)} labels; ladder step on disk {out['sanity']['on_disk']['gain']:+.3f}, "
          f"rerun {full['gain']:+.3f}, label-mean {out['label_mean_gain']:+.3f}")


# ── analysis ───────────────────────────────────────────────────────────────────────────────────────────────────────
def _rank(x):
    x = np.asarray(x, float)
    order = x.argsort(kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):  # average ties
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def _pearson(x, y):
    x, y = x - x.mean(), y - y.mean()
    den = math.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def _perm_p(stat_fn, x, y, n_perm=20000, seed=0):
    obs = stat_fn(x, y)
    if not np.isfinite(obs):
        return obs, float("nan")
    n = len(y)
    if n <= 8:
        perms = [np.array(p) for p in itertools.permutations(range(n))]
    else:
        rng = np.random.default_rng(seed)
        perms = [rng.permutation(n) for _ in range(n_perm)]
    null = np.array([stat_fn(x, y[p]) for p in perms])
    return obs, float((np.sum(np.abs(null) >= abs(obs) - 1e-12) + (0 if n <= 8 else 1)) / (len(null) + (0 if n <= 8 else 1)))


def spearman(x, y):
    return _perm_p(lambda a, b: _pearson(_rank(a), _rank(b)), np.asarray(x, float), np.asarray(y, float))


def partial_spearman(x, y, z):
    rz = _rank(z)
    A = np.c_[np.ones_like(rz), rz]

    def resid(v):
        rv = _rank(v)
        return rv - A @ np.linalg.lstsq(A, rv, rcond=None)[0]

    rx = resid(x)
    # permute y, recompute its residual each time (Freedman-Lane would also do; residual permutation kept simple)
    return _perm_p(lambda a, b: _pearson(a, resid(b)), rx, np.asarray(y, float))


def sign_test(k, n):
    """Two-sided exact binomial test of k successes in n (p=0.5)."""
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


MEASURES = {"a_flat_ood": "flat", "c_noise_ood": "noise", "d_mean_flat_texture": None, "b_baseline_ind": "baseline_ind"}


def analyze(a):
    D = Path(a.out_dir)
    settings = [json.loads(p.read_text()) for p in sorted(D.glob("*.json"))]
    rows, tests = [], []
    for s in settings:
        labs = sorted(s["labels"])
        g = np.array([s["labels"][lb]["gain"] for lb in labs])
        diff = {"a_flat_ood": [s["labels"][lb]["flat"] for lb in labs],
                "c_noise_ood": [s["labels"][lb]["noise"] for lb in labs],
                "d_mean_flat_texture": [(s["labels"][lb]["flat"] + s["labels"][lb]["texture"]) / 2 for lb in labs],
                "b_baseline_ind": [s["labels"][lb]["baseline_ind"] for lb in labs]}
        logv = np.log([s["labels"][lb]["volume_ml"] for lb in labs])
        for lb in labs:
            v = s["labels"][lb]
            rows.append({"setting": s["setting"], "task": s["task"], "type": s["type"], "label": lb,
                         "gain": v["gain"], "p_wilcoxon": v["p"], "n_patients": v["n_patients"],
                         "flat_ood": v["flat"], "texture_ood": v["texture"], "noise_ood": v["noise"],
                         "baseline_ind": v["baseline_ind"], "volume_ml": v["volume_ml"],
                         "nan_rows_flat": v["nan_rows_flat"], "rows_flat": v["rows_flat"]})
        t = {"setting": s["setting"], "task": s["task"], "type": s["type"], "n_labels": len(labs),
             "ladder_step_on_disk": s["sanity"]["on_disk"]["gain"], "ladder_step_rerun": s["sanity"]["rerun"]["gain"],
             "label_mean_gain": s["label_mean_gain"]}
        for m, x in diff.items():
            x = np.array(x, float)
            ok = np.isfinite(x) & np.isfinite(g)
            rho, p = spearman(x[ok], g[ok]) if ok.sum() >= 3 else (float("nan"), float("nan"))
            t[f"rho_{m}"], t[f"p_{m}"] = rho, p
            if len(labs) >= 8:
                okv = ok & np.isfinite(logv)
                pr, pp = partial_spearman(x[okv], g[okv], logv[okv])
                t[f"prho_{m}"], t[f"pp_{m}"] = pr, pp
        for name in ("hardest", "easiest"):
            tt = (s.get("terciles") or {}).get(name)
            t[f"{name}_tercile_gain"] = tt["gain"] if tt else float("nan")
            t[f"{name}_tercile_p"] = tt["p"] if tt else float("nan")
        tests.append(t)

    def write_csv(path, recs):
        keys = list(dict.fromkeys(k for r in recs for k in r))
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(recs)

    write_csv(D / "label_table.csv", rows)
    write_csv(D / "setting_tests.csv", tests)

    md = ["# Label-level difficulty vs target type: results", "",
          "Unit across settings = setting. Difficulty: higher Dice = easier; H_D predicts rho(gain, Dice) < 0.", "",
          "## Sanity check (flat -> texture OOD Dice step, pp)", "",
          "| setting | on-disk ladder | engine rerun | mean of per-label gains |", "|---|---|---|---|"]
    for t in tests:
        md.append(f"| {t['setting']} | {t['ladder_step_on_disk']:+.2f} | {t['ladder_step_rerun']:+.2f} | {t['label_mean_gain']:+.2f} |")
    md += ["", "## Per-setting Spearman rho(gain, difficulty) [permutation p]", "",
           "| setting | type | n | (a) flat OOD | (c) noise OOD | (d) (T+F)/2 | (b) baseline IND | partial log-vol (a) | (c) | (d) | hardest third gain [p] | easiest third gain [p] |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    f2 = lambda r, p: f"{r:+.2f} [{p:.2g}]" if np.isfinite(r) else "-"
    for t in tests:
        md.append(f"| {t['setting']} | {t['type']} | {t['n_labels']} | {f2(t['rho_a_flat_ood'], t['p_a_flat_ood'])} | "
                  f"{f2(t['rho_c_noise_ood'], t['p_c_noise_ood'])} | {f2(t['rho_d_mean_flat_texture'], t['p_d_mean_flat_texture'])} | "
                  f"{f2(t['rho_b_baseline_ind'], t['p_b_baseline_ind'])} | "
                  f"{f2(t.get('prho_a_flat_ood', float('nan')), t.get('pp_a_flat_ood', float('nan')))} | "
                  f"{f2(t.get('prho_c_noise_ood', float('nan')), t.get('pp_c_noise_ood', float('nan')))} | "
                  f"{f2(t.get('prho_d_mean_flat_texture', float('nan')), t.get('pp_d_mean_flat_texture', float('nan')))} | "
                  f"{f2(t['hardest_tercile_gain'], t['hardest_tercile_p'])} | {f2(t['easiest_tercile_gain'], t['easiest_tercile_p'])} |")
    md += ["", "## Sign test over settings (number with rho < 0; exact two-sided)", "",
           "| measure | interface (7) | p | all (11) | p | appearance (4) |", "|---|---|---|---|---|---|"]
    for m in MEASURES:
        r = [(t["type"], t[f"rho_{m}"]) for t in tests if np.isfinite(t[f"rho_{m}"])]
        ki = sum(1 for ty, x in r if ty == "interface" and x < 0); ni = sum(1 for ty, _ in r if ty == "interface")
        ka = sum(1 for _, x in r if x < 0); na = len(r)
        kp = sum(1 for ty, x in r if ty == "appearance" and x < 0); npp = sum(1 for ty, _ in r if ty == "appearance")
        md.append(f"| {m} | {ki}/{ni} | {sign_test(ki, ni):.3g} | {ka}/{na} | {sign_test(ka, na):.3g} | {kp}/{npp} |")
    # preregistered decision on (c)
    powered = [t for t in tests if t["n_labels"] >= 8]
    inter = [t for t in tests if t["type"] == "interface"]
    neg_c = sum(1 for t in inter if t["rho_c_noise_ood"] < 0)
    sig_c = sum(1 for t in powered if t["rho_c_noise_ood"] < 0 and t["p_c_noise_ood"] < 0.05)
    hard_pos = sum(1 for t in powered if t["hardest_tercile_gain"] > 0 and t["hardest_tercile_p"] < 0.05)
    if neg_c >= 5 and sig_c >= 3 and hard_pos >= 3:
        verdict = "H_D supported (difficulty explains the gain at label level)"
    elif (neg_c <= 4 or sig_c <= 1) and (len(powered) - hard_pos) >= 4:
        verdict = "H_T supported (no difficulty slope inside interface tasks)"
    else:
        verdict = "mixed / inconclusive"
    md += ["", "## Preregistered decision (difficulty (c))", "",
           f"- interface settings with rho_c < 0: {neg_c}/{len(inter)}",
           f"- powered settings with rho_c < 0 and p < 0.05: {sig_c}/{len(powered)}",
           f"- powered settings whose hardest third gains significantly (> 0, p < 0.05): {hard_pos}/{len(powered)}",
           f"- **verdict: {verdict}**"]
    (D / "SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    plot(D, settings)


def plot(D, settings):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"appearance": "#c0392b", "interface": "#2f6db5"}
    mk = {"glioma": "o", "brain": "s", "abdomen": "^", "pelvis": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, key, xl in ((axes[0], "flat", "(a) flat-fill OOD Dice of the label (%)"),
                        (axes[1], "noise", "(c) noise-fill OOD Dice of the label (%)")):
        for s in settings:
            xs = [v[key] for v in s["labels"].values()]
            ys = [v["gain"] for v in s["labels"].values()]
            ax.scatter(xs, ys, c=col[s["type"]], marker=mk.get(s["task"], "x"), s=28, alpha=0.75,
                       edgecolors="none")
            if s["task"] == "glioma":
                for lb, v in s["labels"].items():
                    ax.annotate(lb, (v[key], v["gain"]), fontsize=6, color=col["appearance"],
                                xytext=(2, 2), textcoords="offset points")
        ax.axhline(0, color="#888", lw=0.8)
        ax.set_xlabel(xl)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("texture - flat OOD Dice of the label (pp)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker=mk[t], color=col["appearance" if t == "glioma" else "interface"], label=t)
         for t in mk]
    axes[1].legend(handles=h, fontsize=8, title="task (red = appearance, blue = interface)", title_fontsize=8)
    fig.suptitle("Per-label texture gain vs label difficulty, 11 ladder settings (one point = setting x label)")
    fig.tight_layout()
    fig.savefig(D / "gain_vs_difficulty.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--setting", required=True)
    c.add_argument("--type", required=True, choices=["appearance", "interface"])
    c.add_argument("--script", required=True)
    c.add_argument("--raw-dataset", required=True)
    c.add_argument("--out-dir", required=True)
    an = sub.add_parser("analyze")
    an.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    compute(a) if a.cmd == "compute" else analyze(a)
