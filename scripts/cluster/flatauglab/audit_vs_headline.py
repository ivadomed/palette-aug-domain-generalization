#!/usr/bin/env python3
"""Audit the flatauglab arm's metrics against the HEADLINE PA+Auglab run, source by source, exactly where the paper tables read
them (2026-10-29). For a setting, the task's *_combined_01_results.yaml gives the training-contrast modality, its sources
(metrics_dir: own test set + cross-dataset companions) and the headline run (runs[auglabAug_v26_6_2_train050_val000]). For every
source and fold 0-2: <metrics_dir>/auglab_<flatauglab RID>/fold<k>/eval_all.csv must exist and match the headline run's
<metrics_dir>/<category>_<headline RID>/fold<k>/eval_all.csv in total row count AND distinct cases per item (group / contrast /
modality column).

  .venv/bin/python scripts/cluster/flatauglab/audit_vs_headline.py <setting> [<setting> ...] [--only ds1,ds2] [--run-table tsv]
  setting = a run_ids.tsv setting (chaos_t1in, tf2_cbct, ...) or "all". --only: restrict to sources whose path contains
  /<ds>/8_results_ (e.g. the datasets one controller produced). Exit 0 = every audited source OK; 1 = mismatch or missing.
"""
import argparse, collections, csv, glob, os, sys
import yaml

REPO = "/project/aip-jcohen/paulh/mri_synthesis_project"
TASK = {"brats": "brain_tumor/brats2024-glioma", "chaos": "abdomen_healthy/chaos", "onh": "brain_healthy/on-harmony",
        "openms": "brain_ms/open-ms", "tf2": "mandible_healthy/toothfairy2", "ispy2": "breast_cancer/ispy2",
        "pelvic": "pelvis_healthy/totalseg-pelvic", "atlas": "liver_hcc/atlas-liver-hcc", "panseg": "pancreas_disease/pansegdata"}
REF = "auglabAug_v26_6_2_train050_val000"


def stats(p):
    rows = list(csv.DictReader(open(p)))
    col = next((k for k in ("group", "contrast", "modality") if rows and k in rows[0]), None)
    c = collections.defaultdict(set)
    for r in rows:
        c[r[col] if col else "?"].add(r["case"])
    return len(rows), {k: len(v) for k, v in c.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("settings", nargs="+")
    ap.add_argument("--only", default="")
    ap.add_argument("--run-table", default="/scratch/paulh/flatauglab_20261028/run_ids.tsv")
    a = ap.parse_args()
    rt = {}
    for r in csv.DictReader(open(a.run_table), delimiter="\t"):
        rt[r["setting"]] = r["run_id"]
    sel = list(dict.fromkeys(rt)) if a.settings == ["all"] else a.settings
    only = [x for x in a.only.split(",") if x]
    bad = 0
    for s in sel:
        rid = rt[s]; k, tc = s.split("_", 1); tdir = TASK[k]; ds = tdir.split("/")[1]
        cfgp = glob.glob(f"{REPO}/benchmark/02_tasks/{tdir}/5_scripts_{ds}/06_evaluate/configs/*combined_01_results.yaml")
        assert len(cfgp) == 1, cfgp
        cfg = yaml.safe_load(open(cfgp[0]))
        mod = [m for m in cfg["modalities"] if m["name"] == tc]
        assert len(mod) == 1, (s, [m["name"] for m in cfg["modalities"]])
        mod = mod[0]; href = mod["runs"][REF]
        env = {"METRICS_ROOT": f"{REPO}/benchmark/02_tasks/{tdir}/8_results_{ds}/02_metrics", "PROJECT_ROOT": REPO}
        for src in mod.get("sources") or [{"metrics_dir": mod["metrics_dir"]}]:
            md = src["metrics_dir"]
            for v, x in env.items():
                md = md.replace("${%s}" % v, x)
            if only and not any(f"/{o}/8_results_" in md for o in only):
                continue
            tag = md.replace(REPO + "/benchmark/02_tasks/", "")
            hd = [d for d in glob.glob(f"{md}/*_{href}") if os.path.basename(d) in (f"auglab_{href}", f"nnUNet_{href}")]
            if len(hd) != 1:
                print(f"FAIL {s} {tag}: headline dir for {href} not unique: {hd}"); bad = 1; continue
            for f in (0, 1, 2):
                p = f"{md}/auglab_{rid}/fold{f}/eval_all.csv"; hp = f"{hd[0]}/fold{f}/eval_all.csv"
                if not os.path.isfile(p):
                    print(f"MISSING {s} {tag} fold{f}: {p}"); bad = 1; continue
                n, c = stats(p); hn, hc = stats(hp)
                ok = n == hn and c == hc and n > 0
                bad |= not ok
                print(f"{'ok  ' if ok else 'FAIL'} {s} {tag} fold{f}: rows {n}/{hn} cases "
                      + " ".join(f"{i}={c.get(i, 0)}/{hc.get(i, 0)}" for i in sorted(set(c) | set(hc))))
    print("AUDIT OK" if not bad else "AUDIT FAILED")
    sys.exit(bad)


if __name__ == "__main__":
    main()
