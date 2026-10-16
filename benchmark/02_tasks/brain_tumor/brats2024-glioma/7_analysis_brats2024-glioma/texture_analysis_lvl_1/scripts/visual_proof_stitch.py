#!/usr/bin/env python
"""Visual proof v13 (FINDINGS 40, design pre-registered in outputs/plots/stitch/PREREG.md): model stitching, "where does real fill make the difference".

Flat-fill (rung 4.5) and real-fill (rung 5) networks have near-identical representations yet differ in OOD Dice. Hypothesis: the advantage is a late re-wiring
(decoder tail / readout) over a shared feature set. Test: map one network's activation at location L into the other network with a label-free affine
(ridge) stitch and let the TARGET network finish the prediction. Natural test images only, same v10 plan patch per case as the switch experiment (v12).
  --calib : first N_CAL (env ST_N_CAL, default 20) sorted cases of the cell, arms flat / real, folds 0-2. For every location L (skips s0..s5, decoder stages
            d0..d4) and four pairings per fold k (FR: flat_k -> real_k, RF: real_k -> flat_k, FF: flat_k -> flat_k+1, RR: real_k -> real_k+1, k+1 mod 3)
            accumulate on GPU (float64) over body voxels (body mask nearest-downsampled): n, sum x, sum y, X X', X Y', sum y^2. Closed form at the end:
            centred scatters Sxx, Sxy; W = (Sxx + lam I)^-1 Sxy, lam = 1e-3 trace(Sxx) / C; b = muY - W' muX; y_hat = W' x + b (applied voxel-wise
            everywhere in the patch). Also stores the calibration R^2 = 1 - SSE / SST of the stitch.
  run     : every case of the --case-slice (calibration cases flagged calib=True). Hybrids per fold k, then fold-mean softmax over the 3 pairs:
            E   = all six encoder skips of the source mapped, target decoder;
            D_s = source run up to decoder stage s, d_s mapped, target decoder continues from stage s+1 with ITS OWN skips (decoder_from);
            D_4 = source d4 mapped -> target seg layer (readout swap). Parents = flat_k / real_k natural predictions.
            Option ST_BODY_VARIANT=1 (not pre-registered, tests whether mapping background voxels costs Dice): kinds FRb / RFb / FFb / RRb = same hybrids
            with the map applied inside the body only and the TARGET network's own activation kept outside. Every case also gets held-out stitch-fit rows
            (kind r2: per pairing and location, SSE of the mapped source vs the target activation and SST around the calibration mean muY, body voxels, summed
            over folds) so the report can give held-out R^2 = 1 - sum SSE / sum SST next to the in-sample calibration R^2.
  --report: stitch.md (tables + P1/P2/P3 verdicts), stitch_cells.csv, stitch_summary.png/pdf (non-calibration cases only).
Outputs: <OUT>/<cell>/{stitch_<pairing>_f<k>_all.npz, <case>_stitch.csv}; OUT = env VP_STITCH_OUT or <THIS.parent>/outputs/plots/stitch.
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS = Path(__file__).resolve().parent; sys.path.insert(0, str(THIS))
import visual_proof_switch as sw  # noqa: E402  (plumbing: load_case, activations, decoder_fwd, dice_and_j, dice_prob, models_for, wp, LOCS, FOLDS)
from visual_proof_switch import vt, vo, v2, load_case, activations, decoder_fwd, dice_and_j, dice_prob, models_for, wp, LOCS, FOLDS  # noqa: E402

OUT = Path(os.environ.get("VP_STITCH_OUT") or (THIS.parent / "outputs/plots/stitch")); OUT.mkdir(parents=True, exist_ok=True)
N_CAL = int(os.environ.get("ST_N_CAL", 20)); LAM = 1e-3
PAIRS = {"FR": ("flat", 0, "real", 0), "RF": ("real", 0, "flat", 0), "FF": ("flat", 0, "flat", 1), "RR": ("real", 0, "real", 1)}  # src arm, dk, tgt arm, dk
HLOCS = ["E", "D0", "D1", "D2", "D3", "D4"]; GAP_CASE = 0.02; GAP_CELL = 0.05
PAL = {"FR": "#d62728", "RF": "#1f77b4", "FF": "#888888", "RR": "#888888"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def src_tgt(pair, k):
    sa, sd, ta, td = PAIRS[pair]; return (sa, (k + sd) % len(FOLDS)), (ta, (k + td) % len(FOLDS))


# ----------------------------------------------------------------------------------------------------------------------------------- plumbing
def decoder_from(net, skips, s_start, h_start):
    """Continue net's decoder from stage s_start, given h_start as the low-res input of that stage (= output of stage s_start - 1), with net's own skips.
    s_start = len(stages) applies the seg layer only (readout swap)."""
    dec = net.decoder; h = h_start
    for s in range(s_start, len(dec.stages)):
        h = dec.transpconvs[s](h); h = sw.torch_cat(h, skips[-(s + 2)]); h = dec.stages[s](h)
    return dec.seg_layers[-1](h)


def apply_map(h, W, b):
    """h (1, C, ...) -> W' h + b voxel-wise."""
    return (h.movedim(1, -1) @ W + b).movedim(-1, 1).contiguous()


def load_nets(key, dev):
    prs = {a: v2.load_predictor(d, dev) for a, d in models_for(key).items() if a in ("flat", "real")}
    for pr in prs.values():
        if hasattr(pr.network.decoder, "deep_supervision"):
            pr.network.decoder.deep_supervision = False
    return prs


def all_acts(prs, x):
    """Activations + logits of all 6 networks (arm, fold) on x."""
    A = {}
    for arm, pr in prs.items():
        net = pr.network
        for k in FOLDS:
            net.load_state_dict(pr.list_of_parameters[k]); net.eval(); A[(arm, k)] = activations(net, x)
    return A


def set_fold(prs, arm, k):
    net = prs[arm].network; net.load_state_dict(prs[arm].list_of_parameters[k]); net.eval(); return net


# ----------------------------------------------------------------------------------------------------------------------------------- calibration
def calib(key, dev):
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; od = OUT / key; od.mkdir(parents=True, exist_ok=True); prs = load_nets(key, dev); pr0 = prs["flat"]
    cases = sorted(cell["cases"])[:N_CAL]; ax = {}; axy = {}   # ax[(arm, k, loc)] = [n, sx, sxx]; axy[(pair, k, loc)] = [sy, sxy, syy]
    for case in cases:
        t0 = time.time(); ip, bp, gp, _ = load_case(pr0, cell, case, T); x = torch.from_numpy(ip)[None, None].to(dev)
        with torch.no_grad():
            A = all_acts(prs, x); masks = {}; cache = {}
            def flat_(arm, k, loc):
                if (arm, k, loc) in cache:
                    return cache[(arm, k, loc)]
                a = A[(arm, k)][0][loc][0]; sh = tuple(a.shape[1:])
                if sh not in masks:
                    m = torch.from_numpy(bp).to(dev)[None, None].float(); masks[sh] = torch.nn.functional.interpolate(m, size=sh, mode="nearest")[0, 0].reshape(-1) > 0.5
                cache[(arm, k, loc)] = a.reshape(a.shape[0], -1)[:, masks[sh]].double(); return cache[(arm, k, loc)]
            for loc in LOCS:
                cache.clear()
                for arm in ("flat", "real"):
                    for k in FOLDS:
                        X = flat_(arm, k, loc); st = ax.setdefault((arm, k, loc), [0, 0.0, 0.0]); st[0] += X.shape[1]; st[1] = st[1] + X.sum(1); st[2] = st[2] + X @ X.T
                for pair in PAIRS:
                    for k in FOLDS:
                        (sa, sk), (ta, tk) = src_tgt(pair, k); X = flat_(sa, sk, loc); Y = flat_(ta, tk, loc)
                        st = axy.setdefault((pair, k, loc), [0.0, 0.0, 0.0]); st[0] = st[0] + Y.sum(1); st[1] = st[1] + X @ Y.T; st[2] = st[2] + (Y * Y).sum(1)
        A = cache = None; torch.cuda.empty_cache(); log(f"[{key}] calib {case}: {time.time() - t0:.0f} s")
    for pair in PAIRS:
        for k in FOLDS:
            out = {}
            for loc in LOCS:
                (sa, sk), _ = src_tgt(pair, k); n, sx, sxx = ax[(sa, sk, loc)]; sy, sxy, syy = axy[(pair, k, loc)]
                muX, muY = sx / n, sy / n; Sxx = sxx - n * torch.outer(muX, muX); Sxy = sxy - n * torch.outer(muX, muY); C = Sxx.shape[0]
                lam = LAM * torch.trace(Sxx) / C; W = torch.linalg.solve(Sxx + lam * torch.eye(C, dtype=Sxx.dtype, device=Sxx.device), Sxy); b = muY - W.T @ muX
                sst = (syy - n * muY ** 2).sum(); sse = sst - 2 * torch.trace(W.T @ Sxy) + torch.trace(W.T @ Sxx @ W)
                out[f"{loc}_W"] = W.float().cpu().numpy(); out[f"{loc}_b"] = b.float().cpu().numpy(); out[f"{loc}_r2"] = float((1 - sse / sst).cpu()); out[f"{loc}_n"] = int(n); out[f"{loc}_muY"] = muY.float().cpu().numpy()
            np.savez_compressed(od / f"stitch_{pair}_f{k}_all.npz", **out)
    for pair in PAIRS:
        Z = np.load(od / f"stitch_{pair}_f0_all.npz"); log(f"R2 {pair} f0 " + " ".join(f"{loc}:{float(Z[loc + '_r2']):.3f}" for loc in LOCS))
    print("CALIB_DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- run
def run_cell(key, dev, sl=None):
    import torch
    cell = vt.CELLS[key]; T = cell["target"]; od = OUT / key; od.mkdir(parents=True, exist_ok=True); prs = load_nets(key, dev); pr0 = prs["flat"]
    allc = sorted(cell["cases"]); calset = set(allc[:N_CAL]); cases = allc[slice(*[int(v) for v in sl.split(":")])] if sl else allc
    body_var = bool(os.environ.get("ST_BODY_VARIANT")); maps = {}; muy = {}
    for pair in PAIRS:
        for k in FOLDS:
            Z = np.load(od / f"stitch_{pair}_f{k}_all.npz")
            for loc in LOCS:
                maps[(pair, k, loc)] = (torch.from_numpy(Z[f"{loc}_W"]).to(dev), torch.from_numpy(Z[f"{loc}_b"]).to(dev))
                muy[(pair, k, loc)] = torch.from_numpy(Z[f"{loc}_muY"]).to(dev) if f"{loc}_muY" in Z else None
    if os.environ.get("ST_SELFCHECK") and cases:   # plumbing check: own activations through decoder_fwd / decoder_from must reproduce the logits
        ip, bp, gp, _ = load_case(pr0, cell, cases[0], T); x = torch.from_numpy(ip)[None, None].to(dev)
        with torch.no_grad():
            for arm in ("flat", "real"):
                net = set_fold(prs, arm, 0); acts, lg = activations(net, x); tsk = [acts[f"s{j}"] for j in range(6)]
                dE = float((decoder_fwd(net, tsk) - lg).abs().max()); dI = float((decoder_fwd(net, [apply_map(s, torch.eye(s.shape[1], device=dev), torch.zeros(s.shape[1], device=dev)) for s in tsk]) - lg).abs().max())
                dD = [float((decoder_from(net, tsk, s + 1, acts[f"d{s}"]) - lg).abs().max()) for s in range(5)]
                log(f"SELFCHECK {arm} f0: max|logit diff| E {dE:.2e}, E identity-map {dI:.2e}, " + ", ".join(f"D{s} {v:.2e}" for s, v in enumerate(dD)) + f" (logit scale {float(lg.abs().max()):.1f})")
    for case in cases:
        f_out = od / f"{case}_stitch.csv"
        if f_out.exists():
            continue
        t0 = time.time(); ip, bp, gp, _ = load_case(pr0, cell, case, T); x = torch.from_numpy(ip)[None, None].to(dev)
        base = dict(cell=key, task=cell["task"], case=case, calib=case in calset, n_target=int(np.isin(gp, T).sum())); acc = {}; r2acc = {}; masks = {}
        bt = torch.from_numpy(bp).to(dev)[None, None].float()

        def bmask(h):   # body mask at h's resolution (nearest), shape (1, 1, ...) bool
            sh = tuple(h.shape[2:])
            if sh not in masks:
                masks[sh] = torch.nn.functional.interpolate(bt, size=sh, mode="nearest") > 0.5
            return masks[sh]
        with torch.no_grad():
            A = all_acts(prs, x)
            for arm in ("flat", "real"):
                for k in FOLDS:
                    d_, j_, p_ = dice_and_j(A[(arm, k)][1], gp, T); st = acc.setdefault(("parent", arm, ""), [0, []]); st[0] = st[0] + p_.cpu().numpy() / len(FOLDS); st[1].append(j_)
            for k in FOLDS:
                for pair in PAIRS:
                    (sa, sk), (ta, tk) = src_tgt(pair, k); src = A[(sa, sk)][0]; tgt = A[(ta, tk)][0]; net = set_fold(prs, ta, tk)
                    mp = {loc: apply_map(src[loc], *maps[(pair, k, loc)]) for loc in LOCS}
                    for loc in LOCS:   # held-out stitch fit on body voxels (SST around the calibration mean)
                        m = bmask(mp[loc])[0, 0]; yh = mp[loc][0][:, m].double(); y = tgt[loc][0][:, m].double(); mu = muy[(pair, k, loc)]
                        st = r2acc.setdefault((pair, loc), [0.0, 0.0]); st[0] += float(((yh - y) ** 2).sum())
                        st[1] += float(((y - (mu.double()[:, None] if mu is not None else y.mean(1, keepdim=True))) ** 2).sum())
                    tsk = [tgt[f"s{j}"] for j in range(6)]
                    for v in (("", "b") if body_var else ("",)):
                        mix = (lambda loc: mp[loc]) if v == "" else (lambda loc: torch.where(bmask(mp[loc]), mp[loc], tgt[loc]))
                        for hl in HLOCS:
                            if hl == "E":
                                lg = decoder_fwd(net, [mix(f"s{j}") for j in range(6)])
                            else:
                                s = int(hl[1]); lg = decoder_from(net, tsk, s + 1, mix(f"d{s}"))
                            d_, j_, p_ = dice_and_j(lg, gp, T); st = acc.setdefault((pair + v, ta, hl), [0, []]); st[0] = st[0] + p_.cpu().numpy() / len(FOLDS); st[1].append(j_)
                    mp = None   # free; rebinding instead of del keeps the lambdas' name defined
        del A; torch.cuda.empty_cache()
        rows = [dict(base, kind=kd, arm=arm, loc=hl, dice=dice_prob(P, gp, T), J=float(np.mean(js))) for (kd, arm, hl), (P, js) in acc.items()]
        rows += [dict(base, kind="r2", arm=pair, loc=loc, sse=v[0], sst=v[1]) for (pair, loc), v in r2acc.items()]
        pd.DataFrame(rows).to_csv(f_out, index=False); log(f"[{key}] {case}: {len(rows)} rows, {time.time() - t0:.0f} s")
    print("STITCH_DONE", flush=True)


# ----------------------------------------------------------------------------------------------------------------------------------- report
def report():
    cells = [k for k in vo.ORDER if (OUT / k).exists() and list((OUT / k).glob("*_stitch.csv"))]
    S = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=[""]) for k in cells for f in sorted((OUT / k).glob("*_stitch.csv"))], ignore_index=True)
    S["loc"] = S["loc"].fillna(""); S = S[S["calib"].astype(str) != "True"]; R2h = S[S["kind"] == "r2"]; S = S[S["kind"] != "r2"]
    VARS = [("", "map applied to the whole patch (pre-registered)")] + ([("b", "map applied INSIDE THE BODY only, target's own activation outside (exploratory variant)")] if (S["kind"] == "FRb").any() else [])
    L = ["# Model stitching flat <-> real (FINDINGS 40, pre-registered: PREREG.md)", "",
         f"Non-calibration cases per cell: " + ", ".join(f"{k} {S[S.cell == k].case.nunique()}" for k in cells) + f". Stitch = ridge W'x + b fitted on body voxels of the first "
         f"{N_CAL} cases (label-free, lambda = 1e-3 tr(Sxx)/C). Dice = target Dice on the plan patch, fold-mean softmax over the 3 fold pairs. FR = flat source -> real "
         "target, RF = real -> flat, FF / RR = cross-fold self-stitch (k -> k+1); pen = parent - self-stitch Dice (median over cases). g = (FR - flat)/(real - flat): per-case "
         f"median over cases with |real - flat| > {GAP_CASE}, and cell-level (medians) when |median gap| > {GAP_CELL}. g_RF = (real - RF)/(real - flat) = fraction of the gap LOST by R->F. "
         "g_corr = g with the stitch penalty of the target network added back (FR + (real - RR), RF - (flat - FF)). R2 columns = in-sample calibration R^2 (mean over folds; E = mean over s0..s5).", ""]
    rows = []; verd = []
    for k in cells:
        s = S[S.cell == k]; par = {a: s[(s["kind"] == "parent") & (s["arm"] == a)].set_index("case").dice for a in ("flat", "real")}; fl, re_ = par["flat"], par["real"].reindex(par["flat"].index)
        gap = re_ - fl; gm = re_.median() - fl.median(); sel = gap.abs() > GAP_CASE
        r2 = {}; r2loc = {}
        for pair in PAIRS:
            if (OUT / k / f"stitch_{pair}_f0_all.npz").exists():
                Zs = [np.load(OUT / k / f"stitch_{pair}_f{kk}_all.npz") for kk in FOLDS]
                r2loc[pair] = {l: float(np.mean([Z[f"{l}_r2"] for Z in Zs])) for l in LOCS}
                r2[pair] = {hl: float(np.mean([r2loc[pair][l] for l in ([f"s{j}" for j in range(6)] if hl == "E" else [f"d{hl[1]}"])])) for hl in HLOCS}
        L += [f"## {k}: parents flat {fl.median():.3f} / real {re_.median():.3f} (median gap {gm:+.3f}, n = {len(fl)}, cases with |gap| > {GAP_CASE}: {int(sel.sum())})", ""]
        for v, vdesc in VARS:
            L += [f"### {vdesc}", "", "| loc | FR | RF | FF | RR | pen FF | pen RR | g FR (case) | g FR (cell) | g_corr FR (cell) | g_RF loss (cell) | p FR-flat | p FR-real | p FF-flat | p RR-real | p RF-real | R2 FR / RF / FF / RR |", "|" + "---|" * 17]
            for hl in HLOCS:
                H = {p: s[(s["kind"] == p + v) & (s["loc"] == hl)].set_index("case").dice.reindex(fl.index) for p in PAIRS}
                gc = ((H["FR"] - fl) / gap)[sel].median() if sel.any() else np.nan
                ok = abs(gm) > GAP_CELL
                gcell = (H["FR"].median() - fl.median()) / gm if ok else np.nan; gcorr = (H["FR"].median() + (re_.median() - H["RR"].median()) - fl.median()) / gm if ok else np.nan
                grf = (re_.median() - H["RF"].median()) / gm if ok else np.nan; grf_corr = (re_.median() - (H["RF"].median() - (fl.median() - H["FF"].median()))) / gm if ok else np.nan
                r = dict(cell=k, variant=v or "full", loc=hl, n=len(fl), flat=fl.median(), real=re_.median(), gap=gm, **{p: H[p].median() for p in PAIRS},
                         pen_FF=(fl - H["FF"]).median(), pen_RR=(re_ - H["RR"]).median(), g_case=gc, g_cell=gcell, g_corr=gcorr, g_RF=grf, g_RF_corr=grf_corr,
                         p_FR_flat=wp(H["FR"] - fl), p_FR_real=wp(H["FR"] - re_), p_FF_flat=wp(H["FF"] - fl), p_RR_real=wp(H["RR"] - re_), p_RF_real=wp(H["RF"] - re_),
                         **{f"r2_{p}": r2.get(p, {}).get(hl, np.nan) for p in PAIRS})
                rows.append(r)
                L.append(f"| {hl} | {r['FR']:.3f} | {r['RF']:.3f} | {r['FF']:.3f} | {r['RR']:.3f} | {r['pen_FF']:+.3f} | {r['pen_RR']:+.3f} | {gc:.2f} | {gcell:.2f} | {gcorr:.2f} | {grf:.2f} | {r['p_FR_flat']:.1g} | {r['p_FR_real']:.1g} | {r['p_FF_flat']:.1g} | {r['p_RR_real']:.1g} | {r['p_RF_real']:.1g} | "
                         + " / ".join(f"{r[f'r2_{p}']:.2f}" for p in PAIRS) + " |")
            R = pd.DataFrame([r for r in rows if r["cell"] == k and r["variant"] == (v or "full")]).set_index("loc")
            p1 = bool((R["pen_FF"] < 0.03).all() and (R["pen_RR"] < 0.03).all()); worst = max(R["pen_FF"].max(), R["pen_RR"].max())
            if abs(gm) <= GAP_CELL:
                p2 = p3 = f"NOT EVALUABLE (|median gap| {abs(gm):.3f} <= {GAP_CELL}; raw Dice above)"
            else:
                g4, g3 = R.loc["D4", "g_cell"], R.loc["D3", "g_cell"]; near = abs(R.loc["D4", "FR"] - re_.median()) < 0.03
                p2 = ("HOLDS" if (g4 >= 0.7 and g3 >= 0.7 and near) else "KILLED" if g4 < 0.3 else "PARTIAL") + f" (g D4 {g4:.2f}, D3 {g3:.2f}, FR_D4 - real {R.loc['D4', 'FR'] - re_.median():+.3f}, p {R.loc['D4', 'p_FR_real']:.1g}; penalty-corrected g D4 {R.loc['D4', 'g_corr']:.2f})"
                p3 = ("HOLDS" if R.loc["D4", "g_RF"] >= 0.7 else "FAILS") + f" (R->F at D4 loses {R.loc['D4', 'g_RF']:.2f} of the gap; penalty-corrected {R.loc['D4', 'g_RF_corr']:.2f})"
                gE = R.loc["E", "g_cell"]; p2 += f"; secondary: E recovers {gE:.2f}" + (" (>= 0.7: advantage already in the encoder, hypothesis revised)" if gE >= 0.7 else "")
            L += ["", f"- P1 (self-stitch penalty < 0.03 at every location): {'HOLDS' if p1 else 'FAILS'} (worst median penalty {worst:.3f})", f"- P2 (late re-wiring): {p2}", f"- P3 (symmetry): {p3}", ""]
            verd.append(f"| {k} | {v or 'full'} | {'HOLDS' if p1 else 'FAILS'} | {p2.split(' (')[0]} | {p3.split(' (')[0]} |")
        h = R2h[R2h.cell == k]
        if len(h) and r2loc:
            L += ["### Stitch fit R^2 per location: in-sample (calibration, mean over folds) / held-out (non-calibration cases, 1 - sum SSE / sum SST pooled over cases and folds)", "",
                  "| pair | " + " | ".join(LOCS) + " |", "|" + "---|" * (1 + len(LOCS))]
            for pair in PAIRS:
                hp_ = h[h["arm"] == pair].groupby("loc")[["sse", "sst"]].sum()
                L.append(f"| {pair} | " + " | ".join(f"{r2loc[pair][l]:.2f} / {1 - hp_.loc[l, 'sse'] / hp_.loc[l, 'sst']:.2f}" if l in hp_.index else f"{r2loc[pair][l]:.2f} / -" for l in LOCS) + " |")
                for l in LOCS:
                    if l in hp_.index:
                        rows.append(dict(cell=k, variant="r2", loc=l, pair=pair, r2_in=r2loc[pair][l], r2_heldout=1 - hp_.loc[l, "sse"] / hp_.loc[l, "sst"]))
            L.append("")
    L[4:4] = ["## Verdicts (Wilcoxon over cases reported; thresholds on medians as pre-registered; variant b is exploratory)", "", "| cell | variant | P1 | P2 | P3 |", "|---|---|---|---|---|"] + verd + [""]
    Rall = pd.DataFrame(rows); Rall.to_csv(OUT / "stitch_cells.csv", index=False); (OUT / "stitch.md").write_text("\n".join(L)); print("\n".join(L[:14]))
    Rf = Rall[Rall["variant"] == "full"]; Rb = Rall[Rall["variant"] == "b"]
    fig, axs = plt.subplots(1, len(cells), figsize=(3.8 * len(cells), 3.6), squeeze=False)
    for c, k in enumerate(cells):
        ax = axs[0, c]; R = Rf[Rf.cell == k].set_index("loc").reindex(HLOCS); xs = np.arange(len(HLOCS))
        for p in PAIRS:
            ax.plot(xs, R[p], "--" if p in ("FF", "RR") else "-", marker="o" if p in ("FR", "RF") else ("s" if p == "FF" else "^"), ms=3.5, color=PAL[p], lw=1.2 if p in ("FR", "RF") else 0.9,
                    label={"FR": "flat -> real", "RF": "real -> flat", "FF": "flat -> flat (fold)", "RR": "real -> real (fold)"}[p])
        if len(Rb):
            Q = Rb[Rb.cell == k].set_index("loc").reindex(HLOCS)
            for p in ("FR", "RF"):
                ax.plot(xs, Q[p], ":", color=PAL[p], lw=0.9, label=f"{p} body-only map")
        ax.axhline(R["flat"].iloc[0], color="#1f77b4", lw=0.8, ls=":", label="flat parent"); ax.axhline(R["real"].iloc[0], color="#d62728", lw=0.8, ls=":", label="real parent")
        ax.set_xticks(xs); ax.set_xticklabels(HLOCS); ax.set_title(f"{k} (n = {int(R['n'].iloc[0])})", fontsize=8); ax.set_xlabel("stitch location")
        if c == 0:
            ax.set_ylabel("target Dice (median)"); ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "stitch_summary.png", dpi=150); fig.savefig(OUT / "stitch_summary.pdf"); plt.close(fig); print("STITCH_REPORT_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cells", nargs="*", default=[]); ap.add_argument("--case-slice", default=None); ap.add_argument("--calib", action="store_true"); ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report(); return
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for key in args.cells:
        calib(key, dev) if args.calib else run_cell(key, dev, args.case_slice)


if __name__ == "__main__":
    main()
