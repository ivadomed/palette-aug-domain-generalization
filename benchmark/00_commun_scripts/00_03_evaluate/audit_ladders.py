"""Read-only audit of every causal-ablation ladder (benchmark/02_tasks/*/*/8_results_*/02_metrics/*/*/ablations[/<item>]/ladder_series.json).
Per ladder: rung count, flat fill -> real fill at rungs 4/5 (required when its noise-fill RUN_ID is in ladder_flat_rung.yaml), val100 rung,
and per (source, item) the case count of every rung x fold (a missing run dir or a fold with fewer cases than the max is a PROBLEM).
STALE = some eval_all.csv the ladder reads is newer than its ladder_series.json (rerun that ladder's own 06_1X/06_3X script).
Usage: .venv/bin/python audit_ladders.py [-v]   (exit 1 if any ladder has a problem; run through scripts/evaluate/run_audit_ladders.sh)"""
import csv
import glob
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from ladder_ood_common import resolve_run_dir, _src_key, _src_root  # noqa: E402  (the engine's own key resolution)

PATTERNS = ["benchmark/02_tasks/*/*/8_results_*/02_metrics/*/*/ablations/ladder_series.json",
            "benchmark/02_tasks/*/*/8_results_*/02_metrics/*/*/ablations/*/ladder_series.json"]


def registered_noise_rungs():
    reg = HERE / "ladder_flat_rung.yaml"
    if not reg.exists():
        return set()
    return {ln.split(":")[0].strip() for ln in reg.read_text().splitlines()
            if ln.startswith("  ") and ":" in ln and not ln.strip().startswith("#")}


def fold_counts(run_dir):
    """{fold: ({group: n_cases}, mtime) | None} for folds 0-2."""
    out = {}
    for k in range(3):
        f = Path(run_dir) / f"fold{k}" / "eval_all.csv"
        if not f.exists():
            out[k] = None
            continue
        cases = defaultdict(set)
        with open(f) as fh:
            for r in csv.DictReader(fh):
                cases[r["group"]].add(r["case"])
        out[k] = ({g: len(v) for g, v in cases.items()}, f.stat().st_mtime)
    return out


def audit(p: Path, noise_ids):
    d = json.loads(p.read_text())
    labs, keys, jm = d["labels"], d["run_keys"], p.stat().st_mtime
    flat_ok = len(labs) > 5 and labs[4].startswith("flat fill") and "real fill" in labs[5]
    needs_flat = any(k.rpartition("/")[2] in noise_ids for k in keys)
    root = p.parent.parent if p.parent.name == "ablations" else p.parent.parent.parent
    sources = [root] + [({"metrics_root": Path(s["metrics_root"]), "run_subdir": s.get("run_subdir")}
                         if isinstance(s, dict) else Path(s))
                        for s in (d.get("extra_ood_sources") or []) + (d.get("ood_sources") or [])]
    probs, newer = [], 0
    if needs_flat and not flat_ok:
        probs.append("noise-fill rung is registered in ladder_flat_rung.yaml but the ladder has no flat fill -> real fill step")
    for s in sources:
        sroot = _src_root(s)
        name = sroot.parts[sroot.parts.index("02_metrics") - 1].replace("8_results_", "")
        sub = s.get("run_subdir") if isinstance(s, dict) else None
        per = {}
        for i, key in enumerate(keys):
            rd = resolve_run_dir(sroot, _src_key(s, key))
            per[i] = fold_counts(rd) if rd is not None and Path(rd).is_dir() else None
        groups = {g for v in per.values() for c in (v or {}).values() if c for g in c[0]}
        for v in per.values():
            newer += sum(1 for c in (v or {}).values() if c and c[1] > jm)
        for g in sorted(groups):
            vals = {i: None if v is None else tuple((v[k][0].get(g, 0) if v[k] else 0) for k in range(3))
                    for i, v in per.items()}
            mx = max(max(t) for t in vals.values() if t)
            bad = {i: t for i, t in vals.items() if t is None or min(t) < mx}
            if bad:
                probs.append(f"{name}{'/' + sub if sub else ''}:{g} max={mx} short="
                             + ",".join(f"r{i}{'NODIR' if t is None else t}" for i, t in bad.items()))
    if newer:
        probs.append(f"STALE: {newer} eval_all.csv newer than the json")
    return labs, flat_ok, probs, jm


def main():
    noise_ids = registered_noise_rungs()
    n_bad = 0
    for pat in PATTERNS:
        for f in sorted(glob.glob(str(REPO / pat))):
            p = Path(f)
            rel = p.relative_to(REPO / "benchmark/02_tasks")
            labs, flat_ok, probs, jm = audit(p, noise_ids)
            n_bad += bool(probs)
            v100 = any("val100" in l for l in labs)
            print(f"{time.strftime('%m-%d %H:%M', time.localtime(jm))} n={len(labs)} flat->real@4/5={'Y' if flat_ok else 'N'} "
                  f"val100={'Y' if v100 else 'N'} {'OK' if not probs else 'PROBLEMS: ' + ' | '.join(probs)}  {rel.parent}")
    print(f"audit_ladders: {n_bad} ladder(s) with problems")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
