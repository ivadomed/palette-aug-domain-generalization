#!/usr/bin/env python
"""Scratch aggregation configs of the PA variant screens (Vulcan, after `aggregate.sh fetch`). Nothing in the repo results tree
is written: per setting, a SCRATCH metrics dir of symlinks (each arm's run restricted to the table's folds, + the old headline PA
val000 run with all its folds + the same headline run restricted to the table's folds, key <run>_f<folds>), and configs:
  vs_ctrl/          sig_ref = the TamIA control (arm `ctrl`): `sig. vs ref` = one-sided "control better than this row", all contrasts
  vs_ctrl_ho/       same, HELD-OUT contrasts only (contrast_groups = every non-training contrast): `all` = the decision metric
  vs_<arm>/         sig_ref = that arm, all contrasts: the control row's p-value = one-sided "arm better than control"
  vs_<arm>_ho/      same, held-out contrasts only
Arms = the run table's `arm` column (round 2: ctrl p075 edge2x alphawide; round 1 re-read: ctrl texlite keepres edgeonly;
round 3: ctrl + the weakening arms). The control row (`ctrl`) is required.
Each arm row is restricted to the table's folds (a round-1 arm evaluated on folds 0-2 enters a fold-0 table with fold 0 only).
usage: python write_configs.py <run_ids.tsv> <out root> [<title prefix>]   (run_ids.tsv: setting, fold, run_id, pack, arm)"""
import collections, csv, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / "benchmark/02_tasks"
REF = "auglabAug_v26_6_2_train050_val000"
S = {  # setting -> (dataset dir, headline config, metrics subdir, label)
    "onh_t1w": ("brain_healthy/on-harmony", "on-harmony_T1w.yaml", "on_harmony_model/T1w", "Brain (on-harmony T1w)"),
    "openms_flair": ("brain_ms/open-ms", "open-ms_flair_01_results.yaml", "open_ms_model/flair", "MS (open-ms FLAIR)"),
    "chaos_t2spir": ("abdomen_healthy/chaos", "chaos_t2spir_01_results.yaml", "chaos_model/t2spir", "Abdomen (chaos T2spir)"),
}


def restricted(L, name, src, folds):
    """L/<name>/fold<k> -> src/fold<k> for the table's folds only (a real dir of symlinks)."""
    d = L / name
    if d.is_symlink():
        d.unlink()
    d.mkdir(exist_ok=True)
    for x in d.iterdir():
        if x.is_symlink():
            x.unlink()
    for k in folds:
        (d / f"fold{k}").symlink_to(src / f"fold{k}")


def main():
    tsv, out = Path(sys.argv[1]), Path(sys.argv[2])
    prefix = sys.argv[3] if len(sys.argv) > 3 else "PA variants round 2"
    rid = collections.defaultdict(dict); folds = collections.defaultdict(set); arms = []
    for l in tsv.read_text().splitlines()[1:]:
        if not l.strip() or l.startswith("#"):
            continue
        s, f, r, p, a = l.split("\t")[:5]
        assert rid[s].get(a, r) == r, (s, a); rid[s][a] = r; folds[s].add(int(f))
        if a not in arms:
            arms.append(a)
    arms = ["ctrl"] + [a for a in arms if a != "ctrl"]
    for s, (ds, hcfg, sub, label) in S.items():
        if s not in rid:
            print(f"[skip] {s}"); continue
        name = ds.split("/")[1]
        real = T / ds / f"8_results_{name}" / "02_metrics" / sub
        cfg0 = yaml.safe_load((T / ds / f"5_scripts_{name}/06_evaluate/configs" / hcfg).read_text())
        head = [r for r in cfg0["runs"] if REF in r and "/" not in r]
        assert len(head) == 1, (s, head); head = head[0]
        L = out / s / "metrics"; L.mkdir(parents=True, exist_ok=True)
        for x in L.iterdir():
            if x.is_symlink(): x.unlink()
        fl = sorted(folds[s]); ftag = "_f" + "".join(map(str, fl))
        (L / f"auglab_{head}").symlink_to(real / f"auglab_{head}")
        restricted(L, f"auglab_{head}{ftag}", real / f"auglab_{head}", fl)
        runs = [head, head + ftag]
        for a in arms:
            if a in rid[s]:
                r = rid[s][a]; d = real / f"auglab_{r}"
                miss = [k for k in fl if not (d / f"fold{k}" / "eval_all.csv").is_file()]
                if miss:
                    print(f"[warn] {s} {a}: {r} has no eval_all.csv for folds {miss} -- row dropped"); continue
                restricted(L, f"auglab_{r}", d, fl); runs.append(r)
        if rid[s].get("ctrl") not in runs:
            print(f"[skip] {s}: control not evaluated yet"); continue
        ind = cfg0.get("in_domain_contrast")
        contrasts = sorted({row["group"] for k in fl
                            for row in csv.DictReader(open(real / f"auglab_{rid[s]['ctrl']}" / f"fold{k}" / "eval_all.csv"))})
        held = [c for c in contrasts if c != ind]
        assert ind in contrasts and held, (s, ind, contrasts)
        for refarm in [a for a in arms if rid[s].get(a) in runs]:
            for ho in (False, True):
                cfg = {k: v for k, v in cfg0.items()
                       if k not in ("runs", "output_dir", "sig_ref", "title", "metrics_dir", "contrast_groups", "use_contrast_groups")}
                tag = f"vs_{refarm}" + ("_ho" if ho else "")
                what = f"HELD-OUT contrasts only ({', '.join(held)}; `all` = held-out mean)" if ho else "all contrasts"
                cfg.update(title=f"{prefix}, {label}, folds {fl}, {what}: sig. vs ref = {refarm} (one-sided '{refarm} better')",
                           metrics_dir=str(L), runs=runs, sig_ref=rid[s][refarm], output_dir=str(out / s / tag))
                if ho:
                    cfg.pop("in_domain_contrast", None); cfg["contrast_groups"] = {c: c for c in held}
                p = out / s / f"config_{tag}.yaml"
                p.write_text(yaml.safe_dump(cfg, sort_keys=False, width=200, allow_unicode=True))
                print(f"[write] {p}  runs={len(runs)}")


if __name__ == "__main__":
    main()
