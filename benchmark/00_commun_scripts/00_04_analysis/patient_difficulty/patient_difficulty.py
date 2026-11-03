#!/usr/bin/env python3
"""Patient-level, difficulty-matched test: does the texture-fill gain follow DIFFICULTY or TARGET TYPE?
Preregistered in README.md of this folder (committed before any gain-vs-difficulty number was computed).

Subcommands
  compute  --setting K --type appearance|interface --script <ladder script> --out-dir D
      Captures the setting's own ladder call (label_difficulty.capture-style interception of run_ladder /
      run_ladder_cross_dataset), lets the engine insert the flat rung (ladder_ood_common._with_flat_rung), then reads
      the per-case OOD Dice of four rungs (baseline = rung 0, noise fill = rung 4 lblvor, flat fill = rung 4.5,
      texture fill = rung 5) with the engine's own readers (load_case_means / _labelled_case_values /
      resolve_run_dir / _src_key) and pools them exactly as the engine's OOD figure does:
        run_ladder (plain)        equal weight per OOD contrast / extra-source item, cases equal inside it
        run_ladder (ood_groups)   equal weight per contrast group, cases of all member items pooled inside it
        run_ladder_cross_dataset  the contrast_groups tree (dict = equal weight per child, list = pooled cases),
                                  in_domain_group removed
      Each case c gets the ladder weight w_c; a patient (engine patient key) gets W_p = sum of its w_c and
      gain_p = sum w_c (T_c - F_c) / W_p, difficulty_p = sum w_c N_c / W_p (noise fill), baseline_p likewise.
      Then sum_p W_p gain_p == ladder flat -> texture step (checked against the on-disk ladder_series.json).
      Writes D/<K>.json.
  difficulty --out-dir D     prints the difficulty distributions only (used to fix the band; no gain is read)
  analyze    --out-dir D     the preregistered tests -> SUMMARY.md, patient_table.csv, setting_table.csv, figure
"""
from __future__ import annotations

import argparse
import contextlib
import csv as _csv
import io
import itertools
import json
import math
import os
import runpy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "benchmark/00_commun_scripts/00_03_evaluate"))
os.environ.pop("LADDER_TABLE_AGG", None)
os.environ.pop("LADDER_OUT_ROOT", None)
import ladder_ood_common as L  # noqa: E402

# ── preregistered constants (README.md) ────────────────────────────────────────────────────────────────────────
BANDS = {"B1": (55.0, 75.0), "B2": (25.0, 60.0)}   # README: B1 primary, B2 sensitivity (rule fixed on difficulty only)
K_MIN = 5                  # min in-band patients for a setting to enter the band comparison
RESTRICT = (10.0, 85.0)    # A, supplementary ceiling/floor check
REWEIGHT_BINS = [(55, 60), (60, 65), (65, 70), (70, 75)]
N_PERM = 20000
SEED = 0
TASK_OF = {"glioma": "Glioma", "ms": "MS", "breast": "Breast", "hcc": "HCC", "brain": "Brain", "abdomen": "Abdomen",
           "pelvis": "Pelvis", "mandible": "Mandible", "pancreas": "Pancreas"}


# ── capture the ladder call ────────────────────────────────────────────────────────────────────────────────────
def capture(script: str) -> tuple[str, dict]:
    cap = {}

    def stub(name):
        def f(**kw):
            cap["fn"], cap["kw"] = name, kw
        return f

    saved = {n: getattr(L, n) for n in ("run_ladder", "run_ladder_cross_dataset", "run_ladder_grouped")}
    for n in saved:
        setattr(L, n, stub(n))
    sys.modules.pop("ladder_from_roster", None)
    argv = sys.argv
    try:
        sys.argv = [script]
        with contextlib.redirect_stdout(io.StringIO()):
            runpy.run_path(script, run_name="__main__")
    finally:
        sys.argv = argv
        for n, f in saved.items():
            setattr(L, n, f)
    if cap.get("fn") not in ("run_ladder", "run_ladder_cross_dataset"):
        raise SystemExit(f"{script}: unexpected ladder entry point {cap.get('fn')}")
    kw = cap["kw"]
    with contextlib.redirect_stdout(io.StringIO()):
        kw = L._with_flat_rung(lambda **k: k)(**kw)        # the engine's own flat-rung insertion
    return cap["fn"], kw


# ── per-rung case values + ladder weights, per engine mode ─────────────────────────────────────────────────────
def unit_cases_plain(kw, run_key):
    """{unit: {case_key: (patient, value)}} for run_ladder without ood_groups."""
    root = Path(kw["metrics_root"])
    ns = L._dataset_name(root)
    out = {}
    rd = L.resolve_run_dir(root, run_key)
    data = L.load_case_means(rd, "dice") if rd.is_dir() else {}
    for c in kw["ood_contrasts"]:
        if data.get(c):
            out[c] = {f"{ns}|{k}": (L._patient_key(f"{ns}|{k}", c), v) for k, v in data[c].items()}
    for src in kw.get("extra_ood_sources") or []:
        sroot = L._src_root(src)
        ds = L._dataset_name(sroot)
        rd = L.resolve_run_dir(sroot, L._src_key(src, run_key))
        if not rd.is_dir():
            continue
        for item, cases in L.load_case_means(rd, "dice").items():
            if cases:
                lab = f"{ds}/{item}"
                out[lab] = {f"{ds}|{k}": (L._patient_key(f"{ds}|{k}", lab), v) for k, v in cases.items()}
    return {u: {"w": 1.0, "cases": c} for u, c in out.items()}


def unit_cases_grouped(kw, run_key):
    vals = L._labelled_case_values(Path(kw["metrics_root"]), kw["ood_contrasts"], kw.get("extra_ood_sources") or [],
                                   run_key, "dice")
    out = {}
    for g, members in kw["ood_groups"].items():
        cs = {}
        for lbl in members:
            for k, v in vals.get(lbl, {}).items():
                cs[f"{lbl}§{k}"] = (L._patient_key(k), v)      # engine's _pk: no contrast in grouped mode
        out[g] = {"w": 1.0, "cases": cs}
    return out


def unit_cases_tree(kw, run_key):
    """run_ladder_cross_dataset with contrast_groups: leaves = '<prefix><item>' columns; returns leaf-pool units with
    their tree weights (dict = equal weight per finite child; list = one pooled unit)."""
    item_cases = {}
    for root, prefix in kw["prefixed_sources"]:
        root = Path(root)
        ds = L._dataset_name(root)
        rd = L.resolve_run_dir(root, run_key)
        if not rd.is_dir():
            continue
        for item, cases in L.load_case_means(rd, "dice").items():
            item_cases[f"{prefix}{item}"] = {f"{ds}|{k}": (L._patient_key(f"{ds}|{k}", f"{ds}/{item}"), v)
                                             for k, v in cases.items()}
    groups = dict(kw["contrast_groups"])
    if kw.get("in_domain_group"):
        groups.pop(kw["in_domain_group"], None)
    units = {}

    def leaf_names(node):
        if isinstance(node, str):
            return [node]
        return [x for n in node for x in leaf_names(n)]

    def walk(node, w, path):
        if isinstance(node, dict):
            kids = [(k, v) for k, v in node.items() if any(item_cases.get(x) for x in _leaves(v))]
            for k, v in kids:
                walk(v, w / len(kids), f"{path}/{k}")
            return
        cs = {}
        for x in leaf_names(node):
            for ck, pv in item_cases.get(x, {}).items():
                cs[f"{x}§{ck}"] = pv
        if cs:
            units[path] = {"w": w, "cases": cs}

    def _leaves(node):
        if isinstance(node, dict):
            return [x for v in node.values() for x in _leaves(v)]
        return leaf_names(node)

    walk(groups, 1.0, "")
    return units


def units_for(fn, kw, run_key):
    if fn == "run_ladder_cross_dataset":
        if not kw.get("contrast_groups"):
            raise SystemExit("cross-dataset ladder without contrast_groups: not handled")
        return unit_cases_tree(kw, run_key)
    if kw.get("ood_groups"):
        return unit_cases_grouped(kw, run_key)
    return unit_cases_plain(kw, run_key)


def rung_mean(units):
    """The engine's OOD figure for one rung from units (weights renormalised over units with finite cases)."""
    tot, acc = 0.0, 0.0
    for u in units.values():
        v = [x for _, x in u["cases"].values() if np.isfinite(x)]
        if v:
            tot += u["w"]
            acc += u["w"] * float(np.mean(v))
    return 100 * acc / tot if tot else float("nan")


def compute(a):
    fn, kw = capture(a.script)
    rungs = kw["rungs"]
    labs = [r[0] for r in rungs]
    fi = labs.index(L.FLAT_RUNG_LABEL)
    if L.FILL_SWAP_LABEL_HINT not in labs[fi + 1].lower():
        raise SystemExit(f"rung after flat is {labs[fi + 1]!r}")
    keys = {"baseline": rungs[0][2], "noise": rungs[fi - 1][2], "flat": rungs[fi][2], "texture": rungs[fi + 1][2]}
    U = {r: units_for(fn, kw, k) for r, k in keys.items()}
    # case-level paired table on the flat & texture pair (the tested step); weights from that pairing
    pairs = {}
    for u, d in U["flat"].items():
        tex = U["texture"].get(u, {"cases": {}})["cases"]
        common = [c for c in d["cases"] if c in tex and np.isfinite(d["cases"][c][1]) and np.isfinite(tex[c][1])]
        if common:
            pairs[u] = (d["w"], common)
    wsum = sum(w for w, _ in pairs.values())
    pat = {}
    for u, (w, cases) in pairs.items():
        wc = w / wsum / len(cases)
        for c in cases:
            pid = U["flat"][u]["cases"][c][0]
            F = U["flat"][u]["cases"][c][1]
            T = U["texture"][u]["cases"][c][1]
            N = U["noise"].get(u, {"cases": {}})["cases"].get(c, (None, float("nan")))[1]
            B = U["baseline"].get(u, {"cases": {}})["cases"].get(c, (None, float("nan")))[1]
            p = pat.setdefault(pid, {"W": 0.0, "g": 0.0, "f": 0.0, "t": 0.0, "nW": 0.0, "n": 0.0, "bW": 0.0, "b": 0.0,
                                     "units": set(), "n_cases": 0})
            p["W"] += wc; p["g"] += wc * (T - F); p["f"] += wc * F; p["t"] += wc * T
            if np.isfinite(N):
                p["nW"] += wc; p["n"] += wc * N
            if np.isfinite(B):
                p["bW"] += wc; p["b"] += wc * B
            p["units"].add(u); p["n_cases"] += 1
    patients = []
    for pid, p in pat.items():
        patients.append({"pid": pid, "W": p["W"], "gain": 100 * p["g"] / p["W"], "flat": 100 * p["f"] / p["W"],
                         "texture": 100 * p["t"] / p["W"],
                         "noise": 100 * p["n"] / p["nW"] if p["nW"] else float("nan"),
                         "baseline": 100 * p["b"] / p["bW"] if p["bW"] else float("nan"),
                         "units": sorted(p["units"]), "n_cases": p["n_cases"],
                         "noise_cover": p["nW"] / p["W"], "baseline_cover": p["bW"] / p["W"]})
    W = np.array([p["W"] for p in patients])
    g = np.array([p["gain"] for p in patients])
    disk_path = Path(kw["ablations_root"]) / "ladder_series.json"
    disk = json.loads(disk_path.read_text())
    di = disk["labels"].index(L.FLAT_RUNG_LABEL)
    out = {"setting": a.setting, "task": TASK_OF[a.setting.split("_")[0]], "type": a.type, "script": a.script,
           "mode": fn + ("+groups" if kw.get("ood_groups") else ""), "rung_keys": keys,
           "units": {u: {"w": w / wsum, "n_cases": len(c)} for u, (w, c) in pairs.items()},
           "sanity": {"patient_weighted_gain": float((W * g).sum() / W.sum()),
                      "rung_mean_gain": rung_mean(U["texture"]) - rung_mean(U["flat"]),
                      "rung_means": {r: rung_mean(U[r]) for r in U},
                      "on_disk_gain": disk["dice"][di + 1] - disk["dice"][di],
                      "on_disk_noise": disk["dice"][di - 1], "on_disk_baseline": disk["dice"][0],
                      "on_disk_n_patients": disk["fill_swap_significance"]["dice"]["n_cases"],
                      "n_patients": len(patients), "on_disk_path": str(disk_path),
                      "on_disk_run_keys": [disk["run_keys"][i] for i in (0, di - 1, di, di + 1)]},
           "patients": patients}
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(a.out_dir) / f"{a.setting}.json").write_text(json.dumps(out, indent=1, default=float))
    s = out["sanity"]
    print(f"{a.setting}: n={len(patients)} (disk {s['on_disk_n_patients']}); step patient-weighted "
          f"{s['patient_weighted_gain']:+.4f} / rung-mean {s['rung_mean_gain']:+.4f} / on-disk {s['on_disk_gain']:+.4f}; "
          f"noise {s['rung_means']['noise']:.3f} vs disk {s['on_disk_noise']:.3f}; "
          f"baseline {s['rung_means']['baseline']:.3f} vs disk {s['on_disk_baseline']:.3f}")


def load(D):
    return [json.loads(p.read_text()) for p in sorted(Path(D).glob("*.json"))]


def difficulty(a):
    """Difficulty distributions ONLY (noise-fill patient Dice); no gain is printed."""
    S = load(a.out_dir)
    print("setting | type | n | min p10 p25 p50 p75 p90 max (noise-fill OOD Dice, %) | baseline p50")
    for s in S:
        x = np.array([p["noise"] for p in s["patients"]], float)
        b = np.array([p["baseline"] for p in s["patients"]], float)
        q = np.nanpercentile(x, [0, 10, 25, 50, 75, 90, 100])
        print(f"{s['setting']:<16} {s['type']:<10} {len(x):>4} " + " ".join(f"{v:5.1f}" for v in q)
              + f" | {np.nanmedian(b):5.1f}")
    lo = np.arange(0, 100, 5)
    print("\nwindow counts: patients with noise Dice in [a, b] per setting, for candidate windows")
    for wa in lo:
        for wb in lo + 5:
            if wb - wa < 20 or wb > 100:
                continue
            cnt = [int(np.sum((np.array([p["noise"] for p in s["patients"]]) >= wa) &
                              (np.array([p["noise"] for p in s["patients"]]) <= wb))) for s in S]
            print(f"[{wa:3d},{wb:3d}] " + " ".join(f"{c:4d}" for c in cnt))
    print("columns: " + " ".join(s["setting"] for s in S))


# ── analysis (preregistered tests) ─────────────────────────────────────────────────────────────────────────────
from scipy.stats import rankdata  # noqa: E402


def spearman_perm(x, y, n_perm=N_PERM, seed=SEED):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan"), n
    rx, ry = rankdata(x), rankdata(y)
    rx = (rx - rx.mean()) / np.linalg.norm(rx - rx.mean())
    ry = (ry - ry.mean()) / np.linalg.norm(ry - ry.mean())
    obs = float(rx @ ry)
    if n <= 8:
        null = np.array([rx @ ry[list(p)] for p in itertools.permutations(range(n))])
        p = float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
    else:
        rng = np.random.default_rng(seed)
        cnt = 0
        for _ in range(n_perm // 1000):
            P = np.argsort(rng.random((1000, n)), axis=1)
            cnt += int(np.sum(np.abs(ry[P] @ rx) >= abs(obs) - 1e-12))
        p = (cnt + 1) / (n_perm + 1)
    return obs, p, n


def centred_ranks_within(v, units):
    v = np.asarray(v, float)
    out = np.empty_like(v)
    for u in set(units):
        m = np.array([x == u for x in units])
        r = rankdata(v[m])
        out[m] = r - r.mean()
    return out


def weighted_signflip(g, w, n_perm=N_PERM, seed=SEED):
    g, w = np.asarray(g, float), np.asarray(w, float)
    n = len(g)
    if n == 0:
        return float("nan")
    obs = abs(float((w * g).sum()))
    if n <= 16:
        S = np.array(list(itertools.product([1, -1], repeat=n)))
        return float(np.mean(np.abs(S @ (w * g)) >= obs - 1e-12))
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm // 1000):
        S = rng.choice([1.0, -1.0], size=(1000, n))
        cnt += int(np.sum(np.abs(S @ (w * g)) >= obs - 1e-12))
    return (cnt + 1) / (n_perm + 1)


def binom2(k, n):
    if n == 0:
        return float("nan")
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def task_perm(task_vals: dict, task_type: dict):
    """Exact relabeling test over qualifying tasks; statistic = rank sum of appearance tasks."""
    names = sorted(task_vals)
    k = sum(task_type[t] == "appearance" for t in names)
    if k == 0 or k == len(names):
        return float("nan"), float("nan"), 0
    r = dict(zip(names, rankdata([task_vals[t] for t in names])))
    obs = sum(r[t] for t in names if task_type[t] == "appearance")
    null = [sum(r[t] for t in c) for c in itertools.combinations(names, k)]
    mu = k * (len(names) + 1) / 2
    p1 = float(np.mean([s >= obs - 1e-12 for s in null]))
    p2 = float(np.mean([abs(s - mu) >= abs(obs - mu) - 1e-12 for s in null]))
    return p1, p2, len(null)


def band_stats(s, lo, hi):
    P = [p for p in s["patients"] if np.isfinite(p["noise"]) and lo <= p["noise"] <= hi]
    if not P:
        return {"n": 0, "gain": float("nan"), "p": float("nan"), "median_difficulty": float("nan")}
    g = np.array([p["gain"] for p in P]); w = np.array([p["W"] for p in P])
    return {"n": len(P), "gain": float((w * g).sum() / w.sum()), "p": weighted_signflip(g, w),
            "median_difficulty": float(np.median([p["noise"] for p in P]))}


def reweighted(s):
    vals = []
    for lo, hi in REWEIGHT_BINS:
        P = [p for p in s["patients"] if np.isfinite(p["noise"]) and lo <= p["noise"] < hi + (1e-9 if hi == 75 else 0)]
        if not P:
            return float("nan"), False
        w = np.array([p["W"] for p in P]); g = np.array([p["gain"] for p in P])
        vals.append(float((w * g).sum() / w.sum()))
    return float(np.mean(vals)), True


def type_comparison(rows, key, okkey):
    q = [r for r in rows if r[okkey]]
    out = {}
    for ty in ("appearance", "interface"):
        v = [r[key] for r in q if r["type"] == ty]
        kpos = sum(1 for x in v if x > 0)
        out[ty] = {"n": len(v), "pos": kpos, "p": binom2(kpos, len(v))}
    tv, tt = {}, {}
    for r in q:
        tv.setdefault(r["task"], []).append(r[key]); tt[r["task"]] = r["type"]
    tv = {t: float(np.mean(v)) for t, v in tv.items()}
    p1, p2, nrel = task_perm(tv, tt)
    out.update({"task_values": tv, "task_type": tt, "p_one": p1, "p_two": p2, "n_relabel": nrel,
                "n_tasks": {ty: sum(1 for t in tt if tt[t] == ty) for ty in ("appearance", "interface")}})
    return out


def decide_B(rows, tc, band):
    app = [r for r in rows if r[f"{band}_ok"] and r["type"] == "appearance"]
    itf = [r for r in rows if r[f"{band}_ok"] and r["type"] == "interface"]
    app_pos = sum(r[f"{band}_gain"] > 0 for r in app)
    itf_pos = sum(r[f"{band}_gain"] > 0 for r in itf)
    tv = tc["task_values"]; tt = tc["task_type"]
    med_a = float(np.median([v for t, v in tv.items() if tt[t] == "appearance"]))
    med_i = float(np.median([v for t, v in tv.items() if tt[t] == "interface"]))
    if (tc["p_one"] <= 0.05 and app_pos >= len(app) - 1
            and (tc["interface"]["p"] > 0.05 or itf_pos <= len(itf) / 2)):
        v = "H_T supported (type difference survives difficulty matching)"
    elif tc["p_one"] > 0.20 and itf_pos >= 2 * len(itf) / 3 and med_a > 0 and med_i >= 0.5 * med_a:
        v = "H_D supported (no type difference at matched difficulty)"
    else:
        v = "mixed / inconclusive"
    return v, med_a, med_i


def analyze(a):
    D = Path(a.out_dir)
    S = load(D)
    if len(S) != 19:
        raise SystemExit(f"expected 19 settings, found {len(S)}")
    # sanity first
    bad = [s["setting"] for s in S
           if abs(s["sanity"]["patient_weighted_gain"] - s["sanity"]["on_disk_gain"]) > 1e-6
           or abs(s["sanity"]["rung_means"]["noise"] - s["sanity"]["on_disk_noise"]) > 1e-6
           or abs(s["sanity"]["rung_means"]["baseline"] - s["sanity"]["on_disk_baseline"]) > 1e-6
           or s["sanity"]["n_patients"] != s["sanity"]["on_disk_n_patients"]]
    if bad:
        raise SystemExit(f"SANITY FAILED for {bad}; no result is reported")
    rows, prow = [], []
    for s in S:
        P = s["patients"]
        g = [p["gain"] for p in P]
        r = {"setting": s["setting"], "task": s["task"], "type": s["type"], "n": len(P),
             "step_on_disk": s["sanity"]["on_disk_gain"], "step_patients": s["sanity"]["patient_weighted_gain"]}
        for name, key in (("noise", "noise"), ("baseline", "baseline"), ("flat", "flat")):
            r[f"rho_{name}"], r[f"p_{name}"], _ = spearman_perm([p[key] for p in P], g)
        Pr = [p for p in P if RESTRICT[0] <= p["noise"] <= RESTRICT[1]]
        r["rho_noise_restr"], r["p_noise_restr"], r["n_restr"] = spearman_perm([p["noise"] for p in Pr],
                                                                             [p["gain"] for p in Pr])
        if all(len(p["units"]) == 1 for p in P) and len({p["units"][0] for p in P}) > 1:
            units = [p["units"][0] for p in P]
            r["rho_noise_within_unit"], r["p_noise_within_unit"], _ = spearman_perm(
                centred_ranks_within([p["noise"] for p in P], units), centred_ranks_within(g, units))
        else:
            r["rho_noise_within_unit"] = r["p_noise_within_unit"] = float("nan")
        for b, (lo, hi) in BANDS.items():
            bs = band_stats(s, lo, hi)
            r[f"{b}_n"], r[f"{b}_gain"], r[f"{b}_p"], r[f"{b}_med_diff"] = bs["n"], bs["gain"], bs["p"], bs["median_difficulty"]
            r[f"{b}_ok"] = bs["n"] >= K_MIN
        r["RW_gain"], r["RW_ok"] = reweighted(s)
        rows.append(r)
        for p in P:
            prow.append({"setting": s["setting"], "task": s["task"], "type": s["type"], "W": p["W"], "gain": p["gain"],
                         "noise": p["noise"], "baseline": p["baseline"], "flat": p["flat"], "texture": p["texture"],
                         "n_units": len(p["units"])})

    def wcsv(path, recs):
        with open(path, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in recs for k in r)))
            w.writeheader(); w.writerows(recs)
    wcsv(D / "setting_table.csv", rows)
    wcsv(D / "patient_table.csv", prow)   # no case ids (gitignored anyway)

    f2 = lambda x, p: f"{x:+.2f} [{p:.2g}]" if np.isfinite(x) else "-"
    md = ["# Patient-level difficulty-matched test: results", "",
          "Preregistered in README.md. Unit across settings = setting / task. Difficulty = noise-fill (rung 4) OOD Dice "
          "of the patient (higher = easier); gain = texture - flat OOD Dice (pp).", "",
          "## Sanity check (passed: patient-weighted mean of gains == on-disk ladder flat -> texture step)", "",
          "| setting | type | patients | on-disk step | patient-weighted |", "|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['setting']} | {r['type']} | {r['n']} | {r['step_on_disk']:+.3f} | {r['step_patients']:+.3f} |")
    md += ["", "## A. Within-setting Spearman rho(gain, difficulty) [permutation p]", "",
           "| setting | type | n | noise (primary) | noise, [10,85] only (n) | noise, within unit | baseline | flat (coupled) |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['setting']} | {r['type']} | {r['n']} | {f2(r['rho_noise'], r['p_noise'])} | "
                  f"{f2(r['rho_noise_restr'], r['p_noise_restr'])} ({r['n_restr']}) | "
                  f"{f2(r['rho_noise_within_unit'], r['p_noise_within_unit'])} | {f2(r['rho_baseline'], r['p_baseline'])} | "
                  f"{f2(r['rho_flat'], r['p_flat'])} |")
    itf = [r for r in rows if r["type"] == "interface"]
    app = [r for r in rows if r["type"] == "appearance"]
    md += ["", "Sign test over settings (number with rho < 0, exact two-sided):", "",
           "| measure | interface | p | appearance | p |", "|---|---|---|---|---|"]
    for m in ("noise", "noise_restr", "noise_within_unit", "baseline", "flat"):
        vi = [r[f"rho_{m}"] for r in itf if np.isfinite(r[f"rho_{m}"])]
        va = [r[f"rho_{m}"] for r in app if np.isfinite(r[f"rho_{m}"])]
        ki, ka = sum(x < 0 for x in vi), sum(x < 0 for x in va)
        md.append(f"| {m} | {ki}/{len(vi)} | {binom2(ki, len(vi)):.3g} | {ka}/{len(va)} | {binom2(ka, len(va)):.3g} |")
    neg = sum(r["rho_noise"] < 0 for r in itf)
    sig = sum(r["rho_noise"] < 0 and r["p_noise"] < 0.05 for r in itf)
    negr = sum(r["rho_noise_restr"] < 0 for r in itf if np.isfinite(r["rho_noise_restr"]))
    if neg >= 8 and sig >= 3 and negr >= 7:
        va_ = "supports H_D"
    elif neg <= 6 or sig <= 1:
        va_ = "contradicts H_D"
    else:
        va_ = "mixed"
    md += ["", f"Preregistered A decision: interface rho<0 {neg}/10, significant negative {sig}, [10,85] rho<0 {negr}/10 "
               f"-> **{va_}**"]

    verdicts = {}
    for band, title in (("B1", "PRIMARY band B1 = [55, 75]"), ("B2", "sensitivity band B2 = [25, 60]")):
        tc = type_comparison(rows, f"{band}_gain", f"{band}_ok")
        v, med_a, med_i = decide_B(rows, tc, band)
        verdicts[band] = v
        md += ["", f"## B. {title} (K = {K_MIN})", "",
               "| setting | type | in band | median difficulty in band | band-gain [sign-flip p] | qualifies |",
               "|---|---|---|---|---|---|"]
        for r in rows:
            md.append(f"| {r['setting']} | {r['type']} | {r[band + '_n']} | {r[band + '_med_diff']:.1f} | "
                      f"{f2(r[band + '_gain'], r[band + '_p'])} | {'yes' if r[band + '_ok'] else 'NO (< K)'} |")
        md += ["", f"- appearance settings with band-gain > 0: {tc['appearance']['pos']}/{tc['appearance']['n']} "
                   f"(sign test p = {tc['appearance']['p']:.3g}); interface: {tc['interface']['pos']}/{tc['interface']['n']} "
                   f"(p = {tc['interface']['p']:.3g})",
               "- task values (mean of qualifying settings within task): " + ", ".join(
                   f"{t} ({tc['task_type'][t][:3]}) {x:+.2f}" for t, x in sorted(tc["task_values"].items(), key=lambda z: -z[1])),
               f"- separation, task-label permutation over {tc['n_relabel']} relabelings "
               f"({tc['n_tasks']['appearance']} appearance vs {tc['n_tasks']['interface']} interface tasks): "
               f"one-sided p = {tc['p_one']:.3g}, two-sided p = {tc['p_two']:.3g}",
               f"- median task value: appearance {med_a:+.2f}, interface {med_i:+.2f}",
               f"- **preregistered decision: {v}**"]
    tc = type_comparison(rows, "RW_gain", "RW_ok")
    md += ["", "## B, secondary: reweighting to a uniform difficulty histogram inside B1 (4 bins of 5 pp)", "",
           "| setting | type | reweighted gain | covers all bins |", "|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['setting']} | {r['type']} | {r['RW_gain']:+.2f} | {'yes' if r['RW_ok'] else 'no'} |"
                  if r["RW_ok"] else f"| {r['setting']} | {r['type']} | - | no |")
    enough = min(tc["n_tasks"].values()) >= 3
    md += ["", f"- qualifying tasks: {tc['n_tasks']}; " + (
        f"separation one-sided p = {tc['p_one']:.3g} (two-sided {tc['p_two']:.3g}, {tc['n_relabel']} relabelings); "
        f"appearance {tc['appearance']['pos']}/{tc['appearance']['n']} > 0, interface {tc['interface']['pos']}/{tc['interface']['n']} > 0"
        if enough else "fewer than 3 tasks of a type qualify -> descriptive only (preregistered)")]
    (D / "SUMMARY.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    plot(D, S, rows)


def plot(D, S, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"appearance": "#c0392b", "interface": "#2f6db5"}
    order = ["Glioma", "MS", "Breast", "HCC", "Brain", "Abdomen", "Pelvis", "Mandible", "Pancreas"]
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1.1])
    for i, t in enumerate(order):
        ax = fig.add_subplot(gs[i // 4, i % 4]) if i < 8 else fig.add_subplot(gs[2, 0])
        for s in [s for s in S if s["task"] == t]:
            x = [p["noise"] for p in s["patients"]]; y = [p["gain"] for p in s["patients"]]
            ax.scatter(x, y, s=8 if len(x) > 200 else 16, alpha=0.35 if len(x) > 200 else 0.8,
                       color=col[s["type"]], edgecolors="none", label=s["setting"].split("_", 1)[1])
        ax.axvspan(*BANDS["B1"], color="#f0c96b", alpha=0.25, lw=0)
        ax.axvspan(*BANDS["B2"], color="#999999", alpha=0.10, lw=0)
        ax.axhline(0, color="#666", lw=0.7)
        ax.set_xlim(-2, 100)
        ax.set_title(t, fontsize=10, color=col[S[[s["task"] for s in S].index(t)]["type"]])
        ax.set_xlabel("noise-fill OOD Dice of patient (%)", fontsize=8)
        ax.set_ylabel("texture - flat (pp)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="best", markerscale=1.5)
    ax = fig.add_subplot(gs[2, 1:])
    xs = []
    for j, r in enumerate(sorted(rows, key=lambda r: (r["type"] != "appearance", order.index(r["task"]), r["setting"]))):
        for band, mk, off in (("B1", "o", -0.15), ("B2", "s", 0.15)):
            if r[f"{band}_ok"]:
                ax.scatter(j + off, r[f"{band}_gain"], marker=mk, color=col[r["type"]],
                           facecolors=col[r["type"]] if band == "B1" else "none", s=40)
        xs.append(r["setting"])
    ax.axhline(0, color="#666", lw=0.7)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("band-gain, texture - flat (pp)", fontsize=8)
    ax.set_title("Band-gain per setting (filled circle: B1 [55,75] primary; open square: B2 [25,60]); "
                 "settings with < 5 band patients omitted", fontsize=9)
    fig.suptitle("Per-patient texture gain vs noise-fill difficulty (red = appearance, blue = interface; "
                 "yellow band = B1, grey = B2)", fontsize=11)
    fig.tight_layout()
    fig.savefig(D / "gain_vs_difficulty_patients.png", dpi=140)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--setting", required=True)
    c.add_argument("--type", required=True, choices=["appearance", "interface"])
    c.add_argument("--script", required=True)
    c.add_argument("--out-dir", required=True)
    for n in ("difficulty", "analyze"):
        sub.add_parser(n).add_argument("--out-dir", required=True)
    a = ap.parse_args()
    {"compute": compute, "difficulty": difficulty, "analyze": analyze}[a.cmd](a)
