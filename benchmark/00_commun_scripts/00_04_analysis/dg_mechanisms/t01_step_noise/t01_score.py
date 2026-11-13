#!/usr/bin/env python3
"""T01 scoring (registration research/dg_mechanisms/plans/2026-11-12-PREDICTION_T01-T05.md, sha256 0b9f8710...).

Reads the per-fold arm values written by t01_fold_values.py (indood_folds.json, ladder_folds.json), forms every registered
step per setting and fold (d_k = B_k - A_k, same fold), SD_pair (ddof=1), theta = 2.776 SD_pair / sqrt(3) (t(4), F017),
pooled SD per step = sqrt(mean over settings of SD_pair^2), theta_pool; sensitivity with t(2) = 4.303. Re-scores the
patterns of mechanisms/v2/patterns.tsv whose rule carries the fixed 3.2 (P01, P03, P04, P05, P12, P13; P10 as written,
R10 classified against theta) under theta(s, step) and theta_pool(step): stable = passes both, fragile = one, noise =
neither; settings that newly pass are listed as new members.

Reproduction gate (registered: per-fold estimates average back to the published value within 0.1 Dice):
  IND/OOD steps: both arms' 3-fold means within 0.1 of the printed cell of sec/_suppl_ind_ood.tex (and the printed
  fold std within 0.1); R20: of paper/generated_results/sem50/sem50_ind_ood.md; ladder steps R09, R10, R11: mean of d_k
  within 0.1 of the printed step of sec/_suppl_texture_steps.tex and the flat / flat+AugLab levels within 0.1; R06, R07:
  the levels / steps they are made of; R04 (rung 1-3 steps, not printed in the paper): the mirror ladder series (1e-6,
  asserted in t01_fold_values.py) -- reported as 'series'.
Usage: python3 t01_score.py --indood J --ladder J --tex-indood TEX --tex-steps TEX --sem50-md MD --out DIR
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

T4, T2 = 2.776, 4.303
SQ3 = math.sqrt(3.0)
PA, REF = "palette_alone", "auglabAug_v26_6_2_train050_val000"
ARMS7 = ["synthseg_noEM", "synthseg_EM", "baseline", "srcsm", "auglab_default", PA, REF]
SHORT = {"synthseg_noEM": "SynthSeg-noEM", "synthseg_EM": "SynthSeg-EM", "baseline": "Plain", "srcsm": "SRCSM",
         "auglab_default": "AugLab", PA: "Palette", REF: "Palette+AugLab", "sem50": "SynthSeg-EM 50%",
         "ginipa": "GIN-IPA"}

# ── registered pattern members (mechanisms/v2/patterns.tsv, workspace commit c506f86) ──
REG = {
    "P01": ["Glioma T1w", "Glioma T1-CE", "Glioma T2w", "Glioma FLAIR", "MS FLAIR", "MS T1w", "Abdomen T1in",
            "Abdomen T2spir", "Brain T1w", "Spine CT", "Pelvis CT", "Pelvis MRI", "Pancreas T2w"],
    "P03": ["Glioma FLAIR", "Pelvis CT", "Breast T1-CE", "Glioma T2w"],
    "P04": ["Glioma T1-CE", "MS T1w", "MS FLAIR"],
    "P05": ["Glioma T1-CE", "MS T1w"],
    "P10": ["Glioma FLAIR", "Mandible CBCT", "HCC T1-CE"],
    "P12": None,          # every setting except Mandible CBCT (R19 >= 3.2 in 20/21); filled below
    "P13": ["Glioma T1w", "Glioma T1-CE", "MS FLAIR", "MS T1w", "Breast T2w", "HCC T1-CE", "Pelvis MRI",
            "Pancreas T1-CE", "Pancreas T2w"],   # strong members (R21 >= 3.2)
}


# ── paper tables ──
def parse_indood_tex(p: Path) -> dict:
    """{setting: {"IND"|"OOD": {arm: (mean, std) or None}}} of tab:suppl-ind-ood-dice (the per-setting table)."""
    txt = p.read_text().split(r"\label{tab:suppl-ind-ood-dice}")[0]
    out, name = {}, None
    cell = re.compile(r"([\d.]+)\}?\\,\{\\scriptsize\$\\pm\$([\d.]+)\}")
    for line in txt.splitlines():
        if " & IND & " not in line and " & OOD & " not in line:
            continue
        line = re.sub(r"^\\rowcolor\{[^}]*\}", "", line.strip())
        cells = [c.strip() for c in line.rstrip("\\").rstrip().rstrip("\\").split(" & ")]
        if cells[0]:
            name = cells[0]
        vals = {}
        for arm, c in zip(ARMS7, cells[3:3 + 7]):
            m = cell.search(c)
            vals[arm] = (float(m.group(1)), float(m.group(2))) if m else None
        out.setdefault(name, {})[cells[1]] = vals
    return out


def parse_steps_tex(p: Path) -> dict:
    """{setting: dict(flat, t, n, fa, t_aug)} of tab:texture-steps (setting rows only)."""
    out, task = {}, None
    num = r"\$([+-]?[\d.]+)(?:\^\{\*\})?\$"
    for line in p.read_text().splitlines():
        m = re.match(r"\\textbf\{(\w+)\} &", line.strip())
        if m:
            task = m.group(1)
            continue
        m = re.match(r"\\quad ([\w-]+)(?:\$\^\\ddagger\$)? & ([\d.]+) & " + num + r" \(.*?\) & " + num +
                     r" \(.*?\) & ([\d.]+) & " + num, line.strip())
        if m:
            out[f"{task} {m.group(1)}"] = dict(flat=float(m.group(2)), t=float(m.group(3)), n=float(m.group(4)),
                                              fa=float(m.group(5)), t_aug=float(m.group(6)))
    return out


def parse_sem50_md(p: Path) -> dict:
    """{setting: {"IND"|"OOD": {"synthseg_EM": mean, "sem50": mean}}} from the 'Per setting' table."""
    out, name, inside = {}, None, False
    for line in p.read_text().splitlines():
        if line.startswith("## "):
            inside = line.startswith("## Per setting")
            continue
        if not inside or not line.startswith("| ") or set(line) <= set("|- "):
            continue
        c = [x.strip() for x in line.strip().strip("|").split("|")]
        if c[0] == "setting":
            continue
        name = c[0] or name
        f = lambda s: float(s.split("+-")[0]) if re.match(r"-?[\d.]+", s) else None
        out.setdefault(name, {})[c[1]] = {"synthseg_EM": f(c[3]), "sem50": f(c[4])}
    return out


# ── steps ──
def diff_folds(a: dict, b: dict):
    """{fold: b - a} on the folds both have; None if not exactly 3."""
    if not a or not b:
        return None
    ks = sorted(set(a) & set(b))
    return {k: b[k] - a[k] for k in ks} if len(ks) == 3 else None


def steps_for(s_io: dict, s_ld: dict) -> dict:
    """{step: (d_folds, arms (A, B), path, kind)} of one setting."""
    out = {}
    F = s_io["folds"] if s_io else {}

    def io(name, a, b, row):
        fa, fb = (F.get(a) or {}).get(row), (F.get(b) or {}).get(row)
        d = diff_folds(fa, fb)
        if d is not None:
            out[name] = (d, (a, b), "indood", row)

    if s_io:
        io("R03", "baseline", PA, "OOD")
        io("R05", PA, REF, "OOD")
        io("R08", "synthseg_EM", PA, "OOD")
        io("R21", REF, PA, "IND")
        io("R20_IND", "synthseg_EM", "sem50", "IND")
        io("R20_OOD", "synthseg_EM", "sem50", "OOD")
        means = {k: np.mean(list(F[k]["IND"].values())) for k in ARMS7 if k in F and F[k].get("IND")}
        best = max(means, key=means.get)
        for k in ARMS7:
            if k != best and k in means:
                io(f"R19:{SHORT[k]}", k, best, "IND")
        out["_best_ind"] = best
    if s_ld:
        L = s_ld["folds"]

        def ld(name, a, b):
            d = diff_folds(L.get(a), L.get(b))
            if d is not None:
                out[name] = (d, (a, b), "ladder", "OOD")

        ld("R04a", "r0_plain", "r1_kmeans")
        ld("R04b", "r1_kmeans", "r2_label_remap")
        ld("R04c", "r2_label_remap", "r3_noise")
        ld("R06", "r45_flat", "flat_auglab")
        ld("R09", "r45_flat", "r5_real")
        ld("R10", "r45_flat", "r3_noise")
        ld("R11", "flat_auglab", "r6_pa_auglab")
        ld("R09-R10", "r3_noise", "r5_real")
        ld("R05_ladder", "r5_real", "r6_pa_auglab")
        if "R11" in out and "R09" in out:
            r11, r09 = out["R11"][0], out["R09"][0]
            out["R07"] = ({k: r11[k] - r09[k] for k in r09}, ("R09", "R11"), "ladder", "OOD")
    return out


def sd_pair(d: dict) -> float:
    return float(np.std(list(d.values()), ddof=1))


def main():
    ap = argparse.ArgumentParser()
    for k in ("indood", "ladder", "tex-indood", "tex-steps", "sem50-md", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    IO = json.loads(Path(a.indood).read_text())
    LD = json.loads(Path(a.ladder).read_text())
    tex_io, tex_st, sem = parse_indood_tex(Path(a.tex_indood)), parse_steps_tex(Path(a.tex_steps)), parse_sem50_md(Path(a.sem50_md))
    settings = list(IO["settings"])
    for s in LD["settings"]:
        if s not in settings:
            settings.append(s)

    # ── gate per (setting, arm, row) for the IND/OOD path ──
    gate_rows, lvl_ok = [], {}
    for s in settings:
        si = IO["settings"].get(s)
        if not si:
            continue
        for arm in ARMS7 + ["sem50"]:
            for row in ("IND", "OOD"):
                f = (si["folds"].get(arm) or {}).get(row)
                if not f:
                    continue
                mean, std = float(np.mean(list(f.values()))), float(np.std(list(f.values())))
                if arm == "sem50" or (arm == "synthseg_EM" and False):
                    pub = (sem.get(s) or {}).get(row, {}).get("sem50")
                    pstd = None
                    src = "sem50_ind_ood.md"
                else:
                    pv = (tex_io.get(s) or {}).get(row, {}).get(arm)
                    pub, pstd = (pv if pv else (None, None))
                    src = "_suppl_ind_ood.tex"
                ok = pub is not None and abs(mean - pub) <= 0.1 + 1e-9 and (pstd is None or abs(std - pstd) <= 0.1 + 1e-9)
                lvl_ok[(s, arm, row)] = ok
                gate_rows.append([s, "indood", SHORT.get(arm, arm), row, f"{mean:.3f}", f"{std:.3f}",
                                  "" if pub is None else pub, "" if pstd is None else pstd, src, "ok" if ok else "FAIL"])
    for s in settings:
        sl, st = LD["settings"].get(s), tex_st.get(s)
        if not sl:
            continue
        for nm, key in (("r45_flat", "flat"), ("flat_auglab", "fa")):
            f = sl["folds"].get(nm)
            if not f:
                continue
            mean = float(np.mean(list(f.values())))
            pub = st[key] if st else None
            ok = pub is not None and abs(mean - pub) <= 0.1 + 1e-9
            lvl_ok[(s, nm, "OOD")] = ok
            gate_rows.append([s, "ladder", nm, "OOD", f"{mean:.3f}", f"{float(np.std(list(f.values()))):.3f}",
                              "" if pub is None else pub, "", "_suppl_texture_steps.tex", "ok" if ok else "FAIL"])

    # ── steps ──
    PUB_LD = {"R09": "t", "R10": "n", "R11": "t_aug"}
    rows, per = [], {}
    for s in settings:
        stp = steps_for(IO["settings"].get(s), LD["settings"].get(s))
        best = stp.pop("_best_ind", None)
        for name, (d, (A, B), path, row) in stp.items():
            m = float(np.mean(list(d.values())))
            sd = sd_pair(d)
            pub, gate = None, False
            if path == "indood":
                if name.startswith("R20"):
                    pe = (sem.get(s) or {}).get(row, {})
                    pub = None if pe.get("sem50") is None or pe.get("synthseg_EM") is None else pe["sem50"] - pe["synthseg_EM"]
                    pa_ = (tex_io.get(s) or {}).get(row, {}).get("synthseg_EM")
                    gate = lvl_ok.get((s, "sem50", row), False) and lvl_ok.get((s, "synthseg_EM", row), False)
                else:
                    pa_, pb_ = ((tex_io.get(s) or {}).get(row, {}).get(x) for x in (A, B))
                    pub = None if not pa_ or not pb_ else pb_[0] - pa_[0]
                    gate = lvl_ok.get((s, A, row), False) and lvl_ok.get((s, B, row), False)
            else:
                st = tex_st.get(s)
                if name in PUB_LD and st:
                    pub = st[PUB_LD[name]]
                    gate = abs(m - pub) <= 0.1 + 1e-9 and lvl_ok.get((s, "r45_flat" if name != "R11" else "flat_auglab", "OOD"), False)
                elif name == "R06" and st:
                    pub = st["fa"] - st["flat"]
                    gate = lvl_ok.get((s, "r45_flat", "OOD"), False) and lvl_ok.get((s, "flat_auglab", "OOD"), False)
                elif name == "R07" and st:
                    pub = st["t_aug"] - st["t"]
                    d9 = np.mean(list(stp["R09"][0].values()))
                    d11 = np.mean(list(stp["R11"][0].values()))
                    gate = abs(d9 - st["t"]) <= 0.1 + 1e-9 and abs(d11 - st["t_aug"]) <= 0.1 + 1e-9
                elif name == "R09-R10" and st:
                    pub = st["t"] - st["n"]
                    d9 = np.mean(list(stp["R09"][0].values()))
                    d10 = np.mean(list(stp["R10"][0].values()))
                    gate = abs(d9 - st["t"]) <= 0.1 + 1e-9 and abs(d10 - st["n"]) <= 0.1 + 1e-9
                else:   # R04*, R05_ladder: mirror series (asserted at extraction, 1e-6)
                    ser = LD["settings"][s]["series"]
                    pub = ser.get(B, np.nan) - ser.get(A, np.nan) if A in ser and B in ser else None
                    gate = pub is not None and abs(m - pub) <= 1e-6
            per[(s, name)] = dict(d=d, mean=m, sd=sd, pub=pub, gate=gate, path=path, row=row, A=A, B=B, best=best)
    names = sorted({k[1] for k in per}, key=lambda x: (x[:3], x))
    pool = {}
    for n in names:
        sds = [v["sd"] for (s, nn), v in per.items() if nn == n and np.isfinite(v["sd"])]
        pool[n] = (float(np.sqrt(np.mean(np.square(sds)))), len(sds)) if sds else (np.nan, 0)
    for (s, n), v in per.items():
        v["theta"] = T4 * v["sd"] / SQ3
        v["theta_pool"] = T4 * pool[n][0] / SQ3
        v["theta_t2"] = T2 * v["sd"] / SQ3
        v["theta_pool_t2"] = T2 * pool[n][0] / SQ3

    with open(out / "step_sd.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["setting", "step", "d_fold0", "d_fold1", "d_fold2", "mean", "sd_pair", "theta", "theta_pool",
                    "published_value", "gate_ok", "arm_A", "arm_B", "row", "path", "theta_t2", "theta_pool_t2",
                    "sd_pool", "n_settings_pooled"])
        for s in settings:
            for n in names:
                v = per.get((s, n))
                if not v:
                    continue
                dd = [v["d"].get(f"fold{k}", v["d"].get(str(k), np.nan)) for k in range(3)]
                w.writerow([s, n] + [f"{x:.3f}" for x in dd] + [f"{v['mean']:.3f}", f"{v['sd']:.3f}",
                           f"{v['theta']:.3f}", f"{v['theta_pool']:.3f}",
                           "" if v["pub"] is None or not np.isfinite(v["pub"]) else f"{v['pub']:.3f}",
                           "yes" if v["gate"] else "NO", SHORT.get(v["A"], v["A"]), SHORT.get(v["B"], v["B"]),
                           v["row"], v["path"], f"{v['theta_t2']:.3f}", f"{v['theta_pool_t2']:.3f}",
                           f"{pool[n][0]:.3f}", pool[n][1]])
    with open(out / "gate.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["setting", "path", "arm", "row", "mean_of_folds", "fold_std_pop", "published", "published_std",
                    "source", "gate"])
        w.writerows(gate_rows)
    with open(out / "pooled_sd.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["step", "n_settings", "sd_pool", "theta_pool_t4", "theta_pool_t2", "median_sd_pair", "max_sd_pair",
                    "setting_of_max"])
        for n in names:
            vs = {s: v["sd"] for (s, nn), v in per.items() if nn == n}
            smax = max(vs, key=vs.get)
            w.writerow([n, pool[n][1], f"{pool[n][0]:.3f}", f"{T4 * pool[n][0] / SQ3:.3f}", f"{T2 * pool[n][0] / SQ3:.3f}",
                        f"{np.median(list(vs.values())):.3f}", f"{vs[smax]:.3f}", smax])

    # ── pattern re-scoring ──
    def g(s, n, key="mean"):
        v = per.get((s, n))
        return None if v is None else v[key]

    def th(s, n, scheme):
        v = per.get((s, n))
        if v is None:
            return None
        if scheme == "fixed":
            return 4.5 if n == "R07" else 3.2
        return v[{"s": "theta", "pool": "theta_pool", "s_t2": "theta_t2", "pool_t2": "theta_pool_t2"}[scheme]]

    def r01(s):
        F = IO["settings"].get(s, {}).get("folds", {}).get("baseline", {})
        if not F.get("IND") or not F.get("OOD"):
            return None
        return float(np.mean(list(F["OOD"].values())) / np.mean(list(F["IND"].values())))

    def rule(p, s, sc):
        """True / False / None (not evaluable)."""
        try:
            if p == "P01":
                r = r01(s)
                return None if r is None or g(s, "R03") is None else (r < 0.30 and g(s, "R03") >= th(s, "R03", sc))
            if p == "P03":
                if any(g(s, n) is None for n in ("R05", "R08", "R09")):
                    return None
                return (g(s, "R05") >= th(s, "R05", sc) and g(s, "R08") <= -th(s, "R08", sc)
                        and abs(g(s, "R09")) < th(s, "R09", sc))
            if p == "P04":
                if any(g(s, n) is None for n in ("R09", "R09-R10")):
                    return None
                return g(s, "R09") >= th(s, "R09", sc) and g(s, "R09-R10") >= th(s, "R09-R10", sc)
            if p == "P05":
                if any(g(s, n) is None for n in ("R09", "R07")):
                    return None
                return g(s, "R09") >= th(s, "R09", sc) and g(s, "R07") <= -th(s, "R07", sc)
            if p == "P10":      # as written (1.5 cuts); the theta part is reported separately
                if any(g(s, n) is None for n in ("R09", "R10")):
                    return None
                if sc == "fixed":
                    return g(s, "R10") >= 1.5 and g(s, "R09") < 1.5
                return g(s, "R10") >= 1.5 and g(s, "R09") < 1.5 and g(s, "R10") >= th(s, "R10", sc)
            if p == "P12":
                if g(s, "R20_IND") is None:
                    return None
                best = per.get((s, "R20_IND"))["best"]
                r19 = 0.0 if best == "synthseg_EM" else g(s, "R19:SynthSeg-EM")
                t = 0.0 if best == "synthseg_EM" else th(s, "R19:SynthSeg-EM", sc)
                return r19 >= t and r19 > 0 and g(s, "R20_IND") > 0
            if p == "P13":     # strong member: R21 >= threshold (membership R21 > 0 has no 3.2)
                return None if g(s, "R21") is None else g(s, "R21") >= th(s, "R21", sc)
        except TypeError:
            return None
        return None

    REG["P12"] = [s for s in settings if s != "Mandible CBCT" and s in IO["settings"]]
    KEYS = {"P01": ["R03"], "P03": ["R05", "R08", "R09"], "P04": ["R09", "R09-R10"], "P05": ["R09", "R07"],
            "P10": ["R10", "R09"], "P12": ["R19:SynthSeg-EM", "R20_IND"], "P13": ["R21"]}
    prow = []
    for p, ks in KEYS.items():
        for s in settings:
            res = {sc: rule(p, s, sc) for sc in ("fixed", "s", "pool", "s_t2", "pool_t2")}
            reg = s in (REG[p] or [])
            if res["s"] is None and res["pool"] is None and not reg:
                continue
            n_pass = int(bool(res["s"])) + int(bool(res["pool"]))
            n_pass2 = int(bool(res["s_t2"])) + int(bool(res["pool_t2"]))
            lab = {2: "stable", 1: "fragile", 0: "noise"}[n_pass]
            lab2 = {2: "stable", 1: "fragile", 0: "noise"}[n_pass2]
            if res["s"] is None and res["pool"] is None:
                lab = lab2 = "not evaluable"
            status = lab if reg else ("new member (" + lab + ")" if n_pass else "passes fixed 3.2 only, not registered")
            if not reg and not n_pass and not res["fixed"]:
                continue
            vals = "; ".join(f"{k}={g(s, k):+.2f} (theta {g(s, k, 'theta'):.2f}, pool {g(s, k, 'theta_pool'):.2f})"
                             for k in ks if g(s, k) is not None)
            if p == "P01" and r01(s) is not None:
                vals = f"R01={r01(s):.3f}; " + vals
            if p == "P12" and per.get((s, "R20_IND")):
                vals = f"best IND arm={SHORT[per[(s, 'R20_IND')]['best']]}; " + vals
            prow.append([p, s, "yes" if reg else "no", "yes" if res["fixed"] else ("NA" if res["fixed"] is None else "no"),
                         {True: "yes", False: "no", None: "NA"}[res["s"]], {True: "yes", False: "no", None: "NA"}[res["pool"]],
                         status, lab2 if reg else ("new (" + lab2 + ")" if n_pass2 else "-"), vals])
    with open(out / "pattern_membership_revised.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["pattern", "setting", "registered_member", "pass_fixed_3.2_recomputed", "pass_theta_s", "pass_theta_pool",
                    "status", "status_t2_sensitivity", "readouts"])
        w.writerows(prow)

    # P10: R10 vs theta, classification of the positive-noise part (provisional pattern)
    # cross-path check: Palette / Palette+AugLab OOD on the IND/OOD path vs the ladder's rung 5 / rung 6
    cross = []
    for s in settings:
        si, sl = IO["settings"].get(s), LD["settings"].get(s)
        if not si or not sl:
            continue
        for arm, nm in ((PA, "r5_real"), (REF, "r6_pa_auglab")):
            fi, fl = (si["folds"].get(arm) or {}).get("OOD"), sl["folds"].get(nm)
            if fi and fl:
                cross.append([s, SHORT[arm], f"{np.mean(list(fi.values())):.3f}", f"{np.mean(list(fl.values())):.3f}",
                              f"{np.mean(list(fi.values())) - np.mean(list(fl.values())):+.3f}",
                              ",".join(si["kind"][c] + ":" + c for c in si["cols"]), ",".join(sl["strata"])])
    with open(out / "cross_path_check.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["setting", "arm", "indood_OOD", "ladder_OOD", "diff", "indood_columns", "ladder_strata"])
        w.writerows(cross)

    # console summary
    print("GATE: level rows", len(gate_rows), "fail", sum(1 for r in gate_rows if r[-1] != "ok"))
    for r in gate_rows:
        if r[-1] != "ok":
            print("  gate FAIL", r)
    print("steps failing gate:", sorted({(s, n) for (s, n), v in per.items() if not v["gate"]}))
    print("\npooled SD per step:")
    for n in names:
        print(f"  {n:22s} n={pool[n][1]:2d} sd_pool={pool[n][0]:.3f} theta_pool={T4 * pool[n][0] / SQ3:.3f}")
    print("\npattern membership:")
    for r in prow:
        print("  ", "\t".join(map(str, r)))
    print("\ncross-path:")
    for r in cross:
        print("  ", "\t".join(r[:5]))


if __name__ == "__main__":
    main()
