#!/usr/bin/env python3
"""Join the READ-ONLY TamIA scan (tamia_scan_variants.sh -> tamia_variant_scan.tsv) with the Vulcan copies of each run's metrics.
For every candidate dir: metrics dirs -> each fold's eval_all.csv md5 vs the Vulcan copy (dataset 02_metrics trees + the review dirs);
run / prediction dirs -> whether that run id's metrics exist on Vulcan; pack roots / throwaways -> n/a. Writes tamia_deletion_list.tsv.
  python3 checks/deletion_join.py <dir holding tamia_variant_scan.tsv>      (Vulcan login node: md5 of a few dozen small CSVs)
"""
import csv, glob, hashlib, os, re, sys
W = sys.argv[1]
R = "/project/aip-jcohen/paulh/mri_synthesis_project/benchmark/02_tasks"
ROOTS = [R + "/brain_healthy/on-harmony/8_results_on-harmony/02_metrics", R + "/brain_ms/open-ms/8_results_open-ms/02_metrics",
         R + "/abdomen_healthy/chaos/8_results_chaos/02_metrics", "/scratch/paulh/review_20261015", "/scratch/paulh/review_20261016",
         "/scratch/paulh/review_20261017"]
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()


def vulcan_evals(rid):
    out = {}
    for root in ROOTS:
        for c in glob.glob(f"{root}/**/auglab_{rid}/fold*/eval_all.csv", recursive=True):
            out.setdefault(os.path.basename(os.path.dirname(c)), set()).add((md5(c), c))
    return out


rows = list(csv.DictReader(open(f"{W}/tamia_variant_scan.tsv"), delimiter="\t"))
for r in rows:
    tag, kind = r["run_id_or_tag"], r["kind"]
    if kind == "pack_root_or_throwaway" or not re.search(r"_2026\d{4}_\d{6}$", tag):
        st = "n/a (pack root / throwaway: no metrics of its own)"
    else:
        ve = vulcan_evals(tag)
        tam = dict(x.split(":") for x in r["eval_all_md5"].strip(";").split(";") if ":" in x) if r["eval_all_md5"] != "-" else {}
        if kind == "metrics":
            st = " ".join(f"{f}:" + ("IDENTICAL" if h in {x[0] for x in ve.get(f, ())} else ("DIFFERS" if ve.get(f) else "ABSENT_ON_VULCAN"))
                          for f, h in sorted(tam.items())) or "TamIA metrics dir has no eval_all.csv"
        elif ve:
            where = sorted({os.path.dirname(os.path.dirname(x[1])).split("/02_metrics")[0] for v in ve.values() for x in v})
            st = "metrics on Vulcan: " + " ".join(f"{f}(n={len(v)})" for f, v in sorted(ve.items())) + " in " + ";".join(where)
        else:
            st = "NO metrics on Vulcan for this run id"
    r["vulcan_metrics_check"] = st
    r.pop("eval_all_md5", None)
with open(f"{W}/tamia_deletion_list.tsv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} rows, {sum(int(r['size_KiB']) for r in rows) / 1048576:.2f} GiB, {sum(int(r['n_files']) for r in rows)} files")
for r in rows:
    if any(k in r["vulcan_metrics_check"] for k in ("NO metrics", "DIFFERS", "ABSENT", "no eval_all")):
        print("ATTENTION", r["path"], "|", r["vulcan_metrics_check"])
