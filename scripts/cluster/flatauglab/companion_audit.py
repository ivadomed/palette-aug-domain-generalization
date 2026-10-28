#!/usr/bin/env python
"""Audit one companion run's metrics (flatauglab arm; cloned from sem50, 2026-10-18): every fold dir must hold an eval_all.csv whose number of DISTINCT
cases per item equals the expected count (no stale / partial / mixed case sets, see the eval stale-skip gotcha).

  python companion_audit.py <run metrics dir> --ref <reference run metrics dir>     # counts taken from the reference run (same test set)
  python companion_audit.py <run metrics dir> --expect item:n [item:n ...]          # explicit counts
  python companion_audit.py <run metrics dir> --gt <raw dir> --items a b            # n = number of <raw dir>/labelsTs_<item>/*.nii.gz
Options: --folds "0 1 2" (default), --item <name> when the eval_all.csv has no item column (one item per run dir).
The item column is the first of group / contrast / modality present in the csv. Exit 1 on any mismatch or missing file.
"""
import argparse, csv, glob, os, sys, collections


def counts(path, item=None):
    c = collections.defaultdict(set)
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return c
    col = next((k for k in ("group", "contrast", "modality") if k in rows[0]), None)
    for r in rows:
        c[item or (r[col] if col else "?")].add(r["case"])
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--ref"); ap.add_argument("--expect", nargs="*"); ap.add_argument("--gt"); ap.add_argument("--items", nargs="*")
    ap.add_argument("--folds", default="0 1 2"); ap.add_argument("--item")
    a = ap.parse_args()
    bad = 0
    for k in a.folds.split():
        p = os.path.join(a.run, f"fold{k}", "eval_all.csv")
        if not os.path.isfile(p):
            print(f"AUDIT FAIL fold{k}: missing {p}"); bad = 1; continue
        got = counts(p, a.item)
        if a.ref:
            rp = os.path.join(a.ref, f"fold{k}", "eval_all.csv")
            if not os.path.isfile(rp):
                print(f"AUDIT FAIL fold{k}: missing reference {rp}"); bad = 1; continue
            exp = {i: len(s) for i, s in counts(rp, a.item).items()}
        elif a.expect:
            exp = {x.rsplit(":", 1)[0]: int(x.rsplit(":", 1)[1]) for x in a.expect}
        else:
            exp = {i: len(glob.glob(os.path.join(a.gt, f"labelsTs_{i}", "*.nii.gz"))) for i in a.items}
        for i, n in sorted(exp.items()):
            g = len(got.get(i, ()))
            ok = g == n and n > 0
            print(f"{'ok  ' if ok else 'FAIL'} fold{k} {i}: {g}/{n} cases")
            bad |= not ok
        extra = set(got) - set(exp)
        if extra:
            print(f"AUDIT FAIL fold{k}: unexpected items {sorted(extra)}"); bad = 1
    print("AUDIT OK" if not bad else "AUDIT FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
