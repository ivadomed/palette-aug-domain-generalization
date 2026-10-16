#!/usr/bin/env python
"""Scratch aggregation configs of the PA variants round 2 (Vulcan, after `aggregate.sh fetch`). Nothing in the repo results tree
is written: per setting, a SCRATCH metrics dir of symlinks (the round's runs, fetched into the real metrics dir, + the old
headline PA val000 run with all its folds + the same headline run restricted to the round's folds, key <run>_f<folds>), and
one config per reference:
  vs_ctrl/     sig_ref = the TamIA control (pa_ctrl): `sig. vs ref` = one-sided "control better than this row"
  vs_<arm>/    sig_ref = that variant: the control row's p-value = one-sided "variant better than control"
Rows: old headline PA val000 (3 folds, reference only), the same restricted to the round's folds (reference), ctrl, variants.
usage: python write_configs.py <run_ids.tsv> <out root>     (run_ids.tsv: setting, fold, run_id, pack, arm)"""
import collections, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]
T = ROOT / "benchmark/02_tasks"
REF = "auglabAug_v26_6_2_train050_val000"
ARMS = ["ctrl", "p075", "edge2x", "alphawide"]
S = {  # setting -> (dataset dir, headline config, metrics subdir, label)
    "onh_t1w": ("brain_healthy/on-harmony", "on-harmony_T1w.yaml", "on_harmony_model/T1w", "Brain (on-harmony T1w)"),
    "openms_flair": ("brain_ms/open-ms", "open-ms_flair_01_results.yaml", "open_ms_model/flair", "MS (open-ms FLAIR)"),
    "chaos_t2spir": ("abdomen_healthy/chaos", "chaos_t2spir_01_results.yaml", "chaos_model/t2spir", "Abdomen (chaos T2spir)"),
}


def main():
    tsv, out = Path(sys.argv[1]), Path(sys.argv[2])
    rid = collections.defaultdict(dict); folds = collections.defaultdict(set)
    for l in tsv.read_text().splitlines()[1:]:
        s, f, r, p, a = l.split("\t")
        assert rid[s].get(a, r) == r, (s, a); rid[s][a] = r; folds[s].add(int(f))
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
        hf = L / f"auglab_{head}{ftag}"; hf.mkdir(exist_ok=True)
        for k in fl:
            (hf / f"fold{k}").unlink(missing_ok=True); (hf / f"fold{k}").symlink_to(real / f"auglab_{head}" / f"fold{k}")
        runs = [head, head + ftag]
        for a in ARMS:
            if a in rid[s]:
                r = rid[s][a]; d = real / f"auglab_{r}"
                miss = [k for k in fl if not (d / f"fold{k}" / "eval_all.csv").is_file()]
                if miss:
                    print(f"[warn] {s} {a}: {r} has no eval_all.csv for folds {miss} -- row dropped"); continue
                (L / f"auglab_{r}").symlink_to(d); runs.append(r)
        if rid[s].get("ctrl") not in runs:
            print(f"[skip] {s}: control not evaluated yet"); continue
        for refarm in ["ctrl"] + [a for a in ARMS[1:] if rid[s].get(a) in runs]:
            cfg = {k: v for k, v in cfg0.items() if k not in ("runs", "output_dir", "sig_ref", "title", "metrics_dir")}
            cfg.update(title=f"PA variants round 2, {label}, folds {fl}: sig. vs ref = {refarm} (one-sided '{refarm} better')",
                       metrics_dir=str(L), runs=runs, sig_ref=rid[s][refarm], output_dir=str(out / s / f"vs_{refarm}"))
            p = out / s / f"config_vs_{refarm}.yaml"
            p.write_text(yaml.safe_dump(cfg, sort_keys=False, width=200, allow_unicode=True))
            print(f"[write] {p}  runs={len(runs)}")


if __name__ == "__main__":
    main()
