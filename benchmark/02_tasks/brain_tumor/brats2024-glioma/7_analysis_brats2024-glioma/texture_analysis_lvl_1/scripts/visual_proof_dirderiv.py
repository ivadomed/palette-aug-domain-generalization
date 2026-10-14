"""Visual proof v11 (FINDINGS section 38): is the texture that is ACTUALLY in the image used by the decision? Directional derivative, no intervention.

Sections 33-37 measured the sensitivity pattern of each voxel's decision (full input gradient G and its split by skip depth): they say at what scale and
through which path the model WOULD respond, not whether the fine detail present in the image pushes the decision. Here the gradient is projected on the
image's own fine-detail component: hp(x) = x - gaussian(x, sigma). d/d eps f(x + eps * hp(x)) at eps = 0 equals <G, hp(x)>: the first-order change of the
decision's log-odds when the texture that is there is scaled up (eps > 0) or removed (eps = -1). One backward pass per fold and arm, nothing modified.

Per seed (section-36 voxels, outcome pattern over noise / flat / real; gradient of each arm's OWN predicted class, fold-mean), per arm, per sigma in SIGMAS:
  cos_hp    = <G, hp> / (|G| |hp|)                            over the body: alignment of the decision's sensitivity with the image's fine detail (signed)
  cos_lp    = <G, lp> / (|G| |lp|), lp = gaussian(x) - mean   the complementary smooth direction
  cos_null  = mean |<G, hp_r>| / (|G| |hp_r|) over N_NULL phase-randomised copies of hp (same power spectrum, random phase): the chance level of |cos_hp|
  texfrac   = |<G, hp>| / (|<G, hp>| + |<G, lp>|)              share of the directional sensitivity along the texture vs along the smooth content
  attr      = <G, hp> / margin                                 first-order fraction of the decision margin attributable to the texture present (removing it, eps = -1, changes
                                                               the log-odds by -<G, hp>; attr = 1 means the texture carries the whole margin to first order)
  + the same cos_hp / attr restricted to a ball of radius r_loc (local) and 3 r_loc (mid) around the voxel, and to the target region (in).
Comparison: per-case median over seeds, median over cases, Wilcoxon over cases: real - flat, real - noise, per cell and per outcome pattern.
Outputs: outputs/plots/dirderiv/<cell>/seeds.csv (+ <case>_dd.npz seed slices, NOT committed); --report: dirderiv.md, dirderiv_cells.csv, dirderiv_patterns.csv,
dirderiv_summary.png/.pdf, dirderiv_patterns.png, dirderiv_examples_<cell>.png (patient slices, NOT committed).
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_erf_tasks as vt  # noqa: E402
import visual_proof_erf_outcome as vo  # noqa: E402
import visual_proof_pathgrad as pg  # noqa: E402
e7, v2 = vt.e7, vt.v2; FOLDS = (0, 1, 2)
ARMS = ("noise", "flat", "real"); PAL = {"noise": "#2ca02c", "flat": "#1f77b4", "real": "#d62728"}
SEEDS_IN = vo.OUT; OUT = Path(os.environ.get("VP_DIRDERIV_OUT") or (vt.OUT.parent / "dirderiv")); OUT.mkdir(parents=True, exist_ok=True)
MAX_SEEDS = int(os.environ.get("DD_MAX_SEEDS", 24)); MAX_CASES = vo.MAX_CASES; SIGMAS = (1.0, 2.0); N_NULL = 3
ORDER = vo.ORDER; PATS = ["CCC", "FFC", "FCF", "CFF", "CCF", "CFC", "FCC", "FFF"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def phase_randomise(h, rng):
    """Same power spectrum as h, random phases (Hermitian-symmetric so the result is real)."""
    F = np.fft.rfftn(h); ph = np.exp(2j * np.pi * rng.random(F.shape)); out = np.fft.irfftn(np.abs(F) * ph, s=h.shape)
    return out.astype(np.float32)


def cos(a, b, m):
    return float((a[m] * b[m]).sum() / (np.sqrt((a[m] ** 2).sum() * (b[m] ** 2).sum()) + 1e-30))


def run_cell(key, dev, sl=None):
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; r_loc = cell["r_loc"]; od = OUT / key; od.mkdir(exist_ok=True)
    f_in = SEEDS_IN / key / "seeds.csv"
    if not f_in.exists() or f_in.stat().st_size < 20:
        log(f"[{key}] no section-36 seeds"); return
    SD = pd.read_csv(f_in, dtype=str); SD = SD[SD.case != "case"]
    for c in ("seed", "z", "y", "x", "cls", "band", "depth") + tuple(f"{a}_pred" for a in ARMS) + tuple(f"{a}_margin" for a in ARMS):
        SD[c] = pd.to_numeric(SD[c])
    prs = {a: v2.load_predictor(d, dev) for a, d in vo.models(key).items()}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    cases = list(cell["cases"]) if key.startswith("glioma") else list(cell["cases"])[:MAX_CASES]
    cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases
    f_csv = od / "seeds.csv"
    try:
        done = set(pd.read_csv(f_csv).case.astype(str)) if f_csv.exists() and f_csv.stat().st_size > 0 else set()
    except pd.errors.EmptyDataError:
        done = set()
    rng = np.random.default_rng(0)
    for case in cases:
        if case in done or case not in set(SD.case):
            continue
        t0 = time.time(); d = SD[SD.case == case]
        data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tg = np.isin(gt, T)
        body = img != 0 if (img == 0).mean() > 0.05 else (img > np.percentile(img, 5))
        box = e7.patch_at(tg, ps, img.shape); ip, bp, gp = img[box].astype(np.float32), body[box], gt[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
        if any(w for _, w in padw):
            ip, bp, gp = np.pad(ip, padw), np.pad(bp, padw), np.pad(gp, padw)
        V = [(int(r.z), int(r.y), int(r.x)) for r in d.itertuples()]
        # the image's own texture and smooth directions, once per case (restricted to the body: outside it hp = 0)
        HP, LP, NULL = {}, {}, {}
        for s in SIGMAS:
            lo = ndi.gaussian_filter(ip, s); hp = (ip - lo) * bp; lp = (lo - lo[bp].mean()) * bp
            HP[s], LP[s] = hp.astype(np.float32), lp.astype(np.float32); NULL[s] = [phase_randomise(hp, rng) * bp for _ in range(N_NULL)]
        G = {}
        for a, pr in prs.items():
            net = pr.network; cls = [int(getattr(r, f"{a}_pred")) for r in d.itertuples()]; acc = None
            for k in FOLDS:
                net.load_state_dict(pr.list_of_parameters[k]); net.eval(); g = pg.path_grads(net, ip, V, cls, dev, None); acc = g if acc is None else [x_ + y_ for x_, y_ in zip(acc, g)]
            G[a] = [(x_ / len(FOLDS)).cpu().numpy() for x_ in acc]; torch.cuda.empty_cache()
        zz, yy, xx = np.indices(ip.shape, sparse=True); rows = []; sv = {"img": [], "gt": [], "z": [], "pattern": [], "hp": []}
        for i, r in enumerate(d.itertuples()):
            v = V[i]; dist = np.sqrt((zz - v[0]) ** 2 + (yy - v[1]) ** 2 + (xx - v[2]) ** 2)
            masks = {"": bp, "_local": (dist <= r_loc) & bp, "_mid": (dist <= 3 * r_loc) & bp, "_in": np.isin(gp, T) & bp}
            row = dict(cell=key, task=cell["task"], case=case, seed=int(r.seed), region=r.region, pattern=r.pattern, band=int(r.band), depth=float(r.depth), cls=int(r.cls))
            for a in ARMS:
                g = G[a][i]; m = float(getattr(r, f"{a}_margin")); row[f"{a}_margin"] = m; row[f"{a}_pred"] = int(getattr(r, f"{a}_pred"))
                for s in SIGMAS:
                    t = f"s{int(s)}"; hp, lp = HP[s], LP[s]
                    for suf, mk in masks.items():
                        if mk.sum() < 8:
                            continue
                        dh = float((g[mk] * hp[mk]).sum()); dl = float((g[mk] * lp[mk]).sum())
                        row[f"{a}_cos_hp_{t}{suf}"] = cos(g, hp, mk); row[f"{a}_attr_{t}{suf}"] = dh / (abs(m) + 1e-6)
                        if suf == "":
                            row[f"{a}_cos_lp_{t}"] = cos(g, lp, mk); row[f"{a}_texfrac_{t}"] = abs(dh) / (abs(dh) + abs(dl) + 1e-30)
                            row[f"{a}_cos_null_{t}"] = float(np.mean([abs(cos(g, n, mk)) for n in NULL[s]]))
                            row[f"{a}_abscos_hp_{t}"] = abs(row[f"{a}_cos_hp_{t}"])
                sv.setdefault(f"{a}_g", []).append(g[v[0]])
            sv["img"].append(ip[v[0]]); sv["gt"].append(gp[v[0]] == int(r.cls) if int(r.cls) else (gp[v[0]] > 0)); sv["z"].append(v[0]); sv["pattern"].append(r.pattern); sv["hp"].append(HP[1.0][v[0]])
            rows.append(row)
        np.savez_compressed(od / f"{case}_dd.npz", seeds=np.array(V), pattern=np.array(sv.pop("pattern")), z=np.array(sv.pop("z")),
                            **{k: np.array(vv).astype(np.float16 if k != "gt" else bool) for k, vv in sv.items()})
        pd.DataFrame(rows).to_csv(f_csv, mode="a", index=False, header=not (f_csv.exists() and f_csv.stat().st_size > 0)); done.add(case)
        R = pd.DataFrame(rows)
        log(f"[{key}] {case}: {len(d)} seeds; cos_hp s1 (null) noise/flat/real: " + "/".join(f"{R[f'{a}_cos_hp_s1'].median():+.3f} ({R[f'{a}_cos_null_s1'].median():.3f})" for a in ARMS)
            + "; attr s1: " + "/".join(f"{R[f'{a}_attr_s1'].median():+.3f}" for a in ARMS) + f"; {time.time() - t0:.0f} s")


# ----------------------------------------------------------------------------------------------------------------------------------- report
def wp(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return float(wilcoxon(x).pvalue) if len(x) >= 5 and np.any(x != 0) else np.nan


METRICS = [("cos_hp_s1", "cos(G, hp1): signed alignment of the decision's sensitivity with the image's own sigma-1 fine detail"),
           ("abscos_hp_s1", "|cos(G, hp1)|"), ("cos_null_s1", "|cos| chance level (phase-randomised hp1)"),
           ("cos_hp_s2", "cos(G, hp2), sigma 2"), ("cos_lp_s1", "cos(G, lp1): alignment with the smooth content"),
           ("texfrac_s1", "share of the directional sensitivity along the texture vs the smooth content"),
           ("attr_s1", "<G, hp1> / margin: first-order fraction of the decision margin carried by the texture present"),
           ("attr_s2", "<G, hp2> / margin"), ("cos_hp_s1_local", "cos(G, hp1) within r_loc of the voxel"), ("attr_s1_local", "attr within r_loc"),
           ("cos_hp_s1_mid", "cos(G, hp1) within 3 r_loc"), ("cos_hp_s1_in", "cos(G, hp1) inside the target region")]


def report():
    cells = [k for k in ORDER if (OUT / k / "seeds.csv").exists() and (OUT / k / "seeds.csv").stat().st_size > 20]
    S = pd.concat([pd.read_csv(OUT / k / "seeds.csv", dtype=str) for k in cells], ignore_index=True); S = S[S.case != "case"]
    num = [c for c in S.columns if c not in ("cell", "task", "case", "region", "pattern")]; S[num] = S[num].apply(pd.to_numeric, errors="coerce")
    St = S[S.region == "target"]
    L = ["# Directional derivative along the image's own fine detail (FINDINGS section 38): is the texture that is there used by the decision?", "",
         "**DEGENERATE GLOBALLY (audit 2026-10-13, `--check`): nnU-Net is instance-normed, so f(a x) = f(x) and <G, x> = 0 exactly (measured cos(G, x) = 0.0000). Hence <G, hp> = -<G, lp>, texfrac = 0.5 by identity, "
         "attr is not an attribution (gradient x input sums to zero for any split of the image) and the phase-randomised null is biased against the real hp. The GLOBAL tables below (cos_hp, cos_lp, texfrac, attr, cos_null, above chance, per pattern) are WITHDRAWN. "
         "Only the _local / _mid / _in variants (inner product restricted to a ball or region, which breaks the identity) are valid first-order derivatives; the gradient itself matches finite differences (r 0.999). See FINDINGS 38.**", "",
         "Seeds = section-36 voxels; gradient of each arm's OWN predicted class, fold-mean. hp = x - gaussian(x, sigma) over the body; <G, hp> = d/d eps f(x + eps hp) at 0. "
         "Units: per-case median over seeds, median over cases, Wilcoxon over cases.", ""]
    rows = []
    for k in cells:
        d = St[St.cell == k]; pc = d.groupby("case").median(numeric_only=True); r = dict(cell=k, task=d.task.iloc[0], n_cases=len(pc), n_seeds=len(d))
        for m, _ in METRICS:
            for a in ARMS:
                if f"{a}_{m}" in pc.columns:
                    r[f"{a}_{m}"] = pc[f"{a}_{m}"].median()
            if f"real_{m}" in pc.columns:
                for b in ("flat", "noise"):
                    dd = pc[f"real_{m}"] - pc[f"{b}_{m}"]; r[f"d_{m}_r{b[0]}"] = dd.median(); r[f"frac_{m}_r{b[0]}"] = (dd > 0).mean(); r[f"p_{m}_r{b[0]}"] = wp(dd)
        rows.append(r)
    T = pd.DataFrame(rows); T.to_csv(OUT / "dirderiv_cells.csv", index=False)
    for m, desc in METRICS:
        if f"real_{m}" not in T.columns:
            continue
        L += [f"## {m}: {desc}", "", "| cell | task | cases | noise | flat | real | real - flat | frac > 0 | p | real - noise | frac > 0 | p |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in T.itertuples():
            L.append(f"| {r.cell} | {r.task} | {r.n_cases} | {getattr(r, f'noise_{m}'):+.3f} | {getattr(r, f'flat_{m}'):+.3f} | {getattr(r, f'real_{m}'):+.3f} | {getattr(r, f'd_{m}_rf'):+.3f} | "
                     f"{getattr(r, f'frac_{m}_rf'):.2f} | {getattr(r, f'p_{m}_rf'):.2g} | {getattr(r, f'd_{m}_rn'):+.3f} | {getattr(r, f'frac_{m}_rn'):.2f} | {getattr(r, f'p_{m}_rn'):.2g} |")
        L += ["", f"real > flat in {(T[f'd_{m}_rf'] > 0).sum()} / {len(T)} cells (p < 0.05: {((T[f'p_{m}_rf'] < 0.05) & (T[f'd_{m}_rf'] > 0)).sum()} +, {((T[f'p_{m}_rf'] < 0.05) & (T[f'd_{m}_rf'] < 0)).sum()} -); "
              f"real > noise in {(T[f'd_{m}_rn'] > 0).sum()} / {len(T)} (p < 0.05: {((T[f'p_{m}_rn'] < 0.05) & (T[f'd_{m}_rn'] > 0)).sum()} +, {((T[f'p_{m}_rn'] < 0.05) & (T[f'd_{m}_rn'] < 0)).sum()} -)", ""]
    # above chance? |cos_hp| vs the phase-randomised null, per arm
    L += ["## Above chance: |cos(G, hp1)| minus its phase-randomised null, per arm (per-case medians; Wilcoxon over cases)", "", "| cell | cases | noise d (p) | flat d (p) | real d (p) |", "|---|---|---|---|---|"]
    for k in cells:
        d = St[St.cell == k]; pc = d.groupby("case").median(numeric_only=True); L.append(f"| {k} | {len(pc)} | " + " | ".join(f"{(pc[f'{a}_abscos_hp_s1'] - pc[f'{a}_cos_null_s1']).median():+.3f} ({wp(pc[f'{a}_abscos_hp_s1'] - pc[f'{a}_cos_null_s1']):.1g})" for a in ARMS) + " |")
    # per pattern
    rows = []
    for p in PATS:
        for k in cells:
            d = St[(St.cell == k) & (St.pattern == p)]; pc = d.groupby("case").median(numeric_only=True)
            if len(pc) < 3:
                continue
            r = dict(pattern=p, cell=k, task=d.task.iloc[0], n=len(pc))
            for m in ("cos_hp_s1", "attr_s1", "texfrac_s1"):
                for a in ARMS:
                    r[f"{a}_{m}"] = pc[f"{a}_{m}"].median()
                dd = pc[f"real_{m}"] - pc[f"flat_{m}"]; r[f"d_{m}_rf"] = dd.median(); r[f"p_{m}_rf"] = wp(dd); dn = pc[f"real_{m}"] - pc[f"noise_{m}"]; r[f"d_{m}_rn"] = dn.median(); r[f"p_{m}_rn"] = wp(dn)
            rows.append(r)
    Pt = pd.DataFrame(rows); Pt.to_csv(OUT / "dirderiv_patterns.csv", index=False)
    L += ["", "## Per outcome pattern (noise, flat, real): cos_hp_s1 and attr_s1 per arm, real - flat (p), cells with >= 3 cases", ""]
    for p in PATS:
        q = Pt[Pt.pattern == p]
        if not len(q):
            continue
        L += [f"### {p}", "", "| cell | n | cos noise/flat/real | d cos rf (p) | d cos rn (p) | attr noise/flat/real | d attr rf (p) | d attr rn (p) |", "|---|---|---|---|---|---|---|---|"]
        for r in q.itertuples():
            L.append(f"| {r.cell} | {r.n} | {r.noise_cos_hp_s1:+.3f}/{r.flat_cos_hp_s1:+.3f}/{r.real_cos_hp_s1:+.3f} | {r.d_cos_hp_s1_rf:+.3f} ({r.p_cos_hp_s1_rf:.1g}) | {r.d_cos_hp_s1_rn:+.3f} ({r.p_cos_hp_s1_rn:.1g}) | "
                     f"{r.noise_attr_s1:+.3f}/{r.flat_attr_s1:+.3f}/{r.real_attr_s1:+.3f} | {r.d_attr_s1_rf:+.3f} ({r.p_attr_s1_rf:.1g}) | {r.d_attr_s1_rn:+.3f} ({r.p_attr_s1_rn:.1g}) |")
        L.append("")
    # figures: per task, paired per-case values of cos_hp_s1 (with null) and attr_s1, three arms
    tasks = [t for t in vt.TASKS if (T.task == t).any()]
    fig, axs = plt.subplots(2, len(tasks), figsize=(2.3 * len(tasks), 5.4), squeeze=False)
    for j, t in enumerate(tasks):
        d = St[St.task == t]; pc = d.groupby(["cell", "case"]).median(numeric_only=True)
        for i, (m, yl) in enumerate((("cos_hp_s1", "cos(G, image fine detail)"), ("attr_s1", "<G, hp> / margin"))):
            ax = axs[i, j]
            for x_, a in enumerate(ARMS):
                y = pc[f"{a}_{m}"].values; ax.scatter(np.full(len(y), x_) + np.random.default_rng(0).uniform(-0.15, 0.15, len(y)), y, s=5, color=PAL[a], alpha=0.5, lw=0)
                ax.hlines(np.median(y), x_ - 0.3, x_ + 0.3, color=PAL[a], lw=2)
                if m == "cos_hp_s1":
                    ax.hlines(pc[f"{a}_cos_null_s1"].median(), x_ - 0.3, x_ + 0.3, color="0.4", lw=1, ls=":")
            ax.axhline(0, color="k", lw=0.5); ax.set_xticks(range(3)); ax.set_xticklabels(ARMS, fontsize=7); ax.tick_params(labelsize=6)
            if i == 0:
                ax.set_title(f"{t} (n={len(pc)})", fontsize=8)
            if j == 0:
                ax.set_ylabel(yl, fontsize=7)
    fig.suptitle("Is the image's own fine detail used? Directional derivative of each voxel's decision along hp(x) = x - gaussian(x, 1): per-case medians; dotted = phase-randomised chance level of |cos|", fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(OUT / "dirderiv_summary.png", dpi=170); fig.savefig(OUT / "dirderiv_summary.pdf"); plt.close(fig)
    for k in [c for c in ("glioma_t1n_t2f", "glioma_t1n_t2w", "glioma_t2w_t2f", "glioma_t2w_t1n", "breast_t2w_t1wce", "pancreas_t1wce_t2w", "mand_cbct_hanseg_ct", "abd_t1in_ct") if c in cells]:
        examples(k, St[St.cell == k])
    (OUT / "dirderiv.md").write_text("\n".join(L) + "\n"); print("\n".join(L[:6 + len(cells)])); print("DIRDERIV_DONE")


def examples(key, d, W=30):
    """Per pattern the typical seed (by real - flat cos_hp_s1): image | hp(x) | per arm: G and the product G * hp (where the texture pushes the decision, signed)."""
    picks = []
    for p in ("CCC", "FFC", "FCF", "CFF", "FFF"):
        dp = d[d.pattern == p]
        if len(dp):
            dd = dp.real_cos_hp_s1 - dp.flat_cos_hp_s1; picks.append((p, dp.loc[(dd - dd.median()).abs().idxmin()]))
    if not picks:
        return
    cols = ["image", "hp(x)"] + [f"{a}: {w}" for a in ARMS for w in ("|G|", "G*hp")]
    fig, axs = plt.subplots(len(picks), len(cols), figsize=(2.0 * len(cols), 2.1 * len(picks)), squeeze=False)
    for i, (p, s) in enumerate(picks):
        z = np.load(OUT / key / f"{s.case}_dd.npz"); sd = pd.read_csv(OUT / key / "seeds.csv", dtype=str); sd = sd[sd.case == s.case].reset_index(drop=True); j = int(sd.index[sd.seed.astype(int) == int(s.seed)][0])
        v = (int(z["seeds"][j][1]), int(z["seeds"][j][2])); sl = (slice(max(0, v[0] - W), v[0] + W), slice(max(0, v[1] - W), v[1] + W))
        img = z["img"][j].astype(np.float32); g = z["gt"][j]; hp = z["hp"][j].astype(np.float32); lo_, hi_ = np.percentile(img[img != 0] if (img != 0).any() else img, [1, 99])
        vg = max(np.percentile(np.abs(z[f"{a}_g"][j].astype(np.float32))[sl], 99.5) for a in ARMS); vp = max(np.percentile(np.abs(z[f"{a}_g"][j].astype(np.float32) * hp)[sl], 99.5) for a in ARMS) + 1e-12
        for c, nm in enumerate(cols):
            ax = axs[i, c]; ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.imshow(img[sl], cmap="gray", vmin=lo_, vmax=hi_); ax.set_ylabel(f"{p}\n{s.case}", fontsize=6)
            elif c == 1:
                ax.imshow(hp[sl], cmap="gray", vmin=-np.percentile(np.abs(hp[sl]), 99), vmax=np.percentile(np.abs(hp[sl]), 99)); ax.set_title("x - gaussian(x, 1)", fontsize=6)
            else:
                a, w = nm.split(": "); G = z[f"{a}_g"][j].astype(np.float32)
                if w == "|G|":
                    ax.imshow(np.abs(G)[sl], cmap="magma", vmin=0, vmax=vg); ax.set_title(f"{a}: |G|  {'correct' if p[ARMS.index(a)] == 'C' else 'FAILED'}", fontsize=6, color=PAL[a])
                else:
                    ax.imshow((G * hp)[sl], cmap="RdBu_r", vmin=-vp, vmax=vp); ax.set_title(f"{a}: G*hp  cos {getattr(s, f'{a}_cos_hp_s1'):+.2f} attr {getattr(s, f'{a}_attr_s1'):+.2f}", fontsize=6, color=PAL[a])
            ax.contour(g[sl], [0.5], colors="#00b4ff" if c < 2 else "white", linewidths=0.5); ax.plot(v[1] - sl[1].start, v[0] - sl[0].start, "+", color="cyan", ms=6, mew=0.8)
    fig.suptitle(f"{key}: does the texture present push the decision at the voxel (+)? G*hp > 0 (red) = that fine detail raises the decision's log-odds, < 0 (blue) lowers it; the sum over the image is <G, hp>. Contour = GT.", fontsize=6.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT / f"dirderiv_examples_{key}.png", dpi=170); plt.close(fig)


def fwd(prs, a, x, seeds, cls, dev):
    """Fold-mean own-class log-odds of arm `a` at the seeds for input x (one forward per fold, no grad)."""
    import torch
    pr = prs[a]; net = pr.network; xt = torch.from_numpy(x.astype(np.float32)).to(dev)[None, None]; acc = np.zeros(len(seeds))
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        for k in FOLDS:
            net.load_state_dict(pr.list_of_parameters[k]); net.eval(); o = net(xt); o = o[0] if isinstance(o, (list, tuple)) else o
            acc += np.array([float(e7.log_odds(o, c)[0][tuple(s)]) for s, c in zip(seeds, cls)])
    return acc / len(FOLDS)


def check(key, dev, sl=None, n_seeds=6):
    """Sanity checks of the measurement (FINDINGS 38): (1) forward invariance to an affine intensity change (instance norm), (2) <G, x> = 0 and <G, 1> = 0
    as its consequence, (3) the gradient against central finite differences along hp, global and within r_loc."""
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; r_loc = cell["r_loc"]
    SD = pd.read_csv(SEEDS_IN / key / "seeds.csv", dtype=str); SD = SD[SD.case != "case"]
    for c in ("seed", "z", "y", "x", "cls") + tuple(f"{a}_pred" for a in ARMS):
        SD[c] = pd.to_numeric(SD[c])
    prs = {a: v2.load_predictor(d, dev) for a, d in vo.models(key).items()}; pr0 = prs["flat"]; ps = pr0.configuration_manager.patch_size
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    cases = list(cell["cases"]); cases = cases[slice(*[int(v) for v in sl.split(":")])] if sl else cases[:3]
    rows = []
    for case in cases:
        d = SD[SD.case == case]
        if not len(d):
            continue
        d = d.iloc[:n_seeds]
        data, gt = v2.preprocess(pr0, cell["img"](case), cell["gt"](case)); img = data[0]; tg = np.isin(gt, T)
        body = img != 0 if (img == 0).mean() > 0.05 else (img > np.percentile(img, 5))
        box = e7.patch_at(tg, ps, img.shape); ip, bp = img[box].astype(np.float32), body[box]
        padw = [(0, max(0, int(p_) - s_)) for p_, s_ in zip(ps, ip.shape)]
        if any(w for _, w in padw):
            ip, bp = np.pad(ip, padw), np.pad(bp, padw)
        V = [(int(r.z), int(r.y), int(r.x)) for r in d.itertuples()]
        lo = ndi.gaussian_filter(ip, 1.0); hp = ((ip - lo) * bp).astype(np.float32)
        zz, yy, xx = np.indices(ip.shape, sparse=True)
        for a in ARMS:
            cls = [int(getattr(r, f"{a}_pred")) for r in d.itertuples()]
            acc = None
            for k in FOLDS:
                prs[a].network.load_state_dict(prs[a].list_of_parameters[k]); prs[a].network.eval()
                g = pg.path_grads(prs[a].network, ip, V, cls, dev, None); acc = g if acc is None else [x_ + y_ for x_, y_ in zip(acc, g)]
            G = [(x_ / len(FOLDS)).cpu().numpy() for x_ in acc]
            f0 = fwd(prs, a, ip, V, cls, dev); f_aff = fwd(prs, a, 1.3 * ip + 0.4, V, cls, dev)
            fd = {}
            for eps in (0.05, 0.1):
                fd[eps] = (fwd(prs, a, ip + eps * hp, V, cls, dev) - fwd(prs, a, ip - eps * hp, V, cls, dev)) / (2 * eps)
            for i, r in enumerate(d.itertuples()):
                v = V[i]; g = G[i]; dist = np.sqrt((zz - v[0]) ** 2 + (yy - v[1]) ** 2 + (xx - v[2]) ** 2); ball = ((dist <= r_loc) & bp).astype(np.float32)
                hpl = hp * ball
                fdl = {eps: float((fwd(prs, a, ip + eps * hpl, [v], [cls[i]], dev) - fwd(prs, a, ip - eps * hpl, [v], [cls[i]], dev))[0] / (2 * eps)) for eps in (0.05, 0.1)}
                rows.append(dict(cell=key, case=case, seed=int(r.seed), arm=a, f0=f0[i], f_affine=f_aff[i], d_affine=f_aff[i] - f0[i],
                                 cos_G_x=cos(g, ip * bp, bp), cos_G_1=cos(g, bp.astype(np.float32), bp),
                                 dd_grad=float((g * hp).sum()), fd_005=fd[0.05][i], fd_01=fd[0.1][i],
                                 dd_grad_local=float((g * hpl).sum()), fd_local_005=fdl[0.05], fd_local_01=fdl[0.1]))
            log(f"[check {key}] {case} {a}: |f(1.3x+0.4)-f(x)| max {np.abs(f_aff - f0).max():.2e}; cos(G,x) {np.median([r_['cos_G_x'] for r_ in rows if r_['arm'] == a and r_['case'] == case]):+.4f}; "
                f"<G,hp> vs FD(0.05): " + " ".join(f"{x_:+.3f}/{y_:+.3f}" for x_, y_ in zip([(g * hp).sum() for g in G], fd[0.05])))
        torch.cuda.empty_cache()
    R = pd.DataFrame(rows); R.to_csv(OUT / f"check_{key}.csv", index=False)
    for a in ARMS:
        q = R[R.arm == a]
        print(f"CHECK {key} {a}: n {len(q)}; |f(1.3x+0.4)-f(x)| max {q.d_affine.abs().max():.2e} (|f| median {q.f0.abs().median():.2f}); "
              f"cos(G,x) median {q.cos_G_x.median():+.4f} max|.| {q.cos_G_x.abs().max():.4f}; cos(G,1) median {q.cos_G_1.median():+.4f}; "
              f"global <G,hp> vs FD: r {np.corrcoef(q.dd_grad, q.fd_005)[0, 1]:.3f} (eps .05) {np.corrcoef(q.dd_grad, q.fd_01)[0, 1]:.3f} (eps .1), median |grad-FD|/|FD| {(np.abs(q.dd_grad - q.fd_005) / (np.abs(q.fd_005) + 1e-9)).median():.3f}; "
              f"local: r {np.corrcoef(q.dd_grad_local, q.fd_local_005)[0, 1]:.3f} / {np.corrcoef(q.dd_grad_local, q.fd_local_01)[0, 1]:.3f}, sign agreement {(np.sign(q.dd_grad_local) == np.sign(q.fd_local_005)).mean():.2f}, "
              f"median local grad {q.dd_grad_local.median():+.4f} FD {q.fd_local_005.median():+.4f}", flush=True)
    print("DIRDERIV_CHECK_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", nargs="*", default=[]); ap.add_argument("--case-slice", default=None); ap.add_argument("--report", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(); return
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k in a.cells:
        check(k, dev, a.case_slice) if a.check else run_cell(k, dev, a.case_slice)


if __name__ == "__main__":
    main()
