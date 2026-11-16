#!/usr/bin/env python3
"""T11 completion block (plans/2026-11-16-repredict-indomain.md "Use"; T11 rules: plans/2026-11-16-PREDICTION_T11_T13_T27.md,
sha256 e435ef58...). Scores the settings whose SynthSeg-noEM in-domain predictions were missing at the first T11 scoring and
passed the acceptance QC (qc_indomain.tsv), with T11's own code, unchanged:
  1. t11_noem_errors/t11_cases.py main() with t04_arms.STAGE pointed at the in-domain stage (the only change: where
     t04_arms.pred_dir looks when the repo tree has no hit), --settings = the admitted settings, into OUT/cases/;
  2. OUT/merged/ = the first scoring's per-case tables (read only, FIRST dir) with the admitted settings' rows replaced by
     OUT/cases/ rows; then t11_noem_errors/t11_score.py --dir OUT/merged (registered classification, outcome over 13 P14
     members, prediction 2);
  3. OUT/completion_block.md: the newly scored settings, the outcome before / after, and the non-admitted settings.
The first scoring (FIRST/*) is never written.
  t11_completion.py --qc QC.tsv --stage DIR --first T11 --out T11/completion --tex TEX [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
T11DIR = HERE.parent / "t11_noem_errors"
sys.path.insert(0, str(T11DIR))
import t11_cases as T11C  # noqa: E402

SCOPE = ["Glioma_t1c", "Glioma_t2f", "Pelvis_ct", "Pelvis_mri", "Mandible_cbct"]
TABLES = ["case_rows.parquet", "label_rows.parquet", "fp_components.parquet", "cf14_cases.parquet", "stored_label_dice.parquet"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", required=True)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--first", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tex", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    first, out = Path(a.first), Path(a.out)
    if out.resolve() == first.resolve():
        raise SystemExit("refusing: --out is the first scoring dir")
    out.mkdir(parents=True, exist_ok=True)
    qc = pd.read_csv(a.qc, sep="\t")
    q = qc[(qc.arm == "synthseg_noEM") & qc.setting.isin(SCOPE)].set_index("setting")
    admitted = [s for s in SCOPE if s in q.index and bool(q.loc[s, "passed"])]
    refused = {s: ("QC failed" if s in q.index else "no QC row") for s in SCOPE if s not in admitted}
    print(f"[t11_completion] admitted {admitted}; not admitted {refused}", flush=True)
    (out / "admitted.json").write_text(json.dumps(dict(admitted=admitted, not_admitted=refused), indent=1))
    if not admitted:
        (out / "completion_block.md").write_text("# T11 completion block\n\nNo setting passed the acceptance QC; nothing scored.\n")
        return
    # 1. per-case tables of the admitted settings (T11 code unchanged; stage = in-domain stage)
    T11C.TA.STAGE = Path(a.stage)
    cdir = out / "cases"
    sys.argv = ["t11_cases.py", "--out", str(cdir), "--workers", str(a.workers), "--settings", ",".join(admitted)]
    try:
        T11C.main()
    except SystemExit as e:
        if e.code not in (0, None):
            raise SystemExit(f"t11_cases exited {e.code} (errors in the completion cases; see log)")
    newR = pd.read_csv(cdir / "roster.tsv", sep="\t")
    print("[t11_completion] prediction sources", sorted(newR.how.astype(str).unique()), flush=True)
    # 2. merged copy: first scoring with the admitted settings replaced
    mdir = out / "merged"
    mdir.mkdir(exist_ok=True)
    oldR = pd.read_csv(first / "roster.tsv", sep="\t")
    pd.concat([oldR[~oldR.setting.isin(admitted)], newR], ignore_index=True).to_csv(mdir / "roster.tsv", sep="\t", index=False)
    checks = {}
    for t in TABLES:
        o, n = pd.read_parquet(first / t), pd.read_parquet(cdir / t)
        pd.concat([o[~o.setting.isin(admitted)], n], ignore_index=True).to_parquet(mdir / t)
        if t == "cf14_cases.parquet":  # CF14 needs no prediction: the first scoring already had it; must be identical
            j = o[o.setting.isin(admitted)].merge(n, on=["setting", "case"], suffixes=("_first", "_new"))
            checks["cf14_cases_compared"] = int(len(j))
            checks["cf14_max_abs_diff"] = float((j.cf14_first - j.cf14_new).abs().max()) if len(j) else None
        if t == "stored_label_dice.parquet":
            j = o[o.setting.isin(admitted)].merge(n, on=["setting", "case", "fold", "label"], suffixes=("_first", "_new"))
            checks["stored_rows_compared"] = int(len(j))
            checks["stored_max_abs_diff"] = float((j.stored_dice_first - j.stored_dice_new).abs().max()) if len(j) else None
    (out / "merge_checks.json").write_text(json.dumps(checks, indent=1))
    print("[t11_completion] merge checks", checks, flush=True)
    res = subprocess.run([sys.executable, str(T11DIR / "t11_score.py"), "--dir", str(mdir), "--tex", a.tex],
                         capture_output=True, text=True)
    (mdir / "score_console.txt").write_text(res.stdout + res.stderr)
    if res.returncode:
        raise SystemExit(f"t11_score failed ({res.returncode}); see {mdir / 'score_console.txt'}")
    # 3. completion block
    P = pd.read_csv(mdir / "per_setting.tsv", sep="\t")
    R1 = json.loads((first / "rules.json").read_text())
    R2 = json.loads((mdir / "rules.json").read_text())
    cols = ["label", "p14", "pred_complete", "paper_ind_noEM", "recomputed_own_item_ind", "gate_diff", "gate", "qc_label_mismatch_1e-4",
            "n_case_folds", "n_empty", "mean_dice_fg", "E", "F", "F_all", "n_fp_comp", "median_fp_dist_big", "class"]
    newT = P[P.setting.isin(admitted)][[c for c in cols if c in P]]
    p14 = P[P.p14 == True]  # noqa: E712
    lines = ["# T11 completion block (in-domain re-prediction, plans/2026-11-16-repredict-indomain.md)", "",
             "Automatically written by repredict_indomain/t11_completion.py; the first scoring (T11/*) is unchanged.",
             f"Admitted by the acceptance QC (SynthSeg-noEM): {', '.join(admitted)}. Not admitted: "
             + (", ".join(f"{k} ({v})" for k, v in refused.items()) or "none") + ".", "",
             "## Newly scored settings (registered T11 rules, T11 code unchanged)", "",
             "```", newT.round(4).to_string(index=False), "```", "",
             "## P14 outcome over 13 members", "",
             f"- First scoring: {R1['p14_counts']} -> {R1['outcome']}",
             f"- With the completion: {R2['p14_counts']} -> {R2['outcome']}", "",
             "```", p14[["label", "E", "F", "class"]].round(4).to_string(index=False), "```", "",
             "## M08 prediction 2 (CF14 needs no prediction; expected unchanged)", "",
             f"- First: rho {R1['prediction2']['rho']:.4f} over {R1['prediction2']['n_settings']} settings, passes {R1['prediction2']['passes']}",
             f"- Merged: rho {R2['prediction2']['rho']:.4f} over {R2['prediction2']['n_settings']} settings, passes {R2['prediction2']['passes']}",
             f"- Merge checks: {checks}", "",
             f"Files: {cdir} (per-case tables of the admitted settings), {mdir} (merged tables, per_setting.tsv, rules.json, "
             "score_console.txt)."]
    (out / "completion_block.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
