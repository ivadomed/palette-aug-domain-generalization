#!/usr/bin/env python3
"""
Download the MALIGNANT NON-HCC subset of LLD-MMRI-MedSAM2 (wanglab, HuggingFace):
intrahepatic cholangiocarcinoma (category 1, n=58) + hepatic metastasis
(category 3, n=51) = 109 patients, all 8 phases, images + MedSAM2 masks.

Why this dataset exists (2026-09-01): atlas-liver-hcc's causal-ablation ladder
draws EVERY out-of-domain contrast from a single evaluation cohort
(lld-mmri-hcc's 157 HCC patients), and its conclusion flips sign depending on
which of those contrasts are counted (real-fill vs noise-fill: -1.85 p=0.006 on
4 contrasts, -0.71 n.s. on 3, +2.21 p=0.006 on 2). More cases on the
unambiguously-OOD contrasts (T2WI, DWI) is the available data fix. These are the
same source repo, same 8 phases, same MedSAM2 human-in-the-loop mask provenance
and the same acquisition cohort as lld-mmri-hcc -- so this buys CASE COUNT and
disease diversity, NOT cohort independence. Do not describe it as an independent
cohort.

Benign categories (hemangioma 79, abscess 54, cyst 53, FNH 46) are deliberately
EXCLUDED: the atlas model segments HCC tumour, and benign focal lesions are a
different task (fluid-bright cysts, markedly T2-hyperintense hemangiomas), not a
harder version of the same one.

Filenames on the hub are `{PID}_{category}_{PHASE}_0000.nii.gz` (images) and
`{PID}_{category}_{PHASE}.nii.gz` (labels) -- the middle token is the lesion
category, so exact paths are constructible and we can use per-file
hf_hub_download. Do NOT switch to snapshot_download with allow_patterns: it must
first list all 7971 files in the repo and that call rate-limits (429) hard.

Usage:  python 00_00_download.py [--workers 8] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "wanglab/LLD-MMRI-MedSAM2"
DATASET_ROOT = Path(__file__).resolve().parents[2]           # benchmark/02_tasks/liver_hcc/lld-mmri-malignant
RAW = DATASET_ROOT / "0_raw_lld-mmri-malignant"
DEST = RAW / "LLD-MMRI-MedSAM2"

PHASES = ["T2WI", "DWI", "C-pre", "C+A", "C+V", "C+Delay", "InPhase", "OutPhase"]


def targets() -> list[tuple[str, Path]]:
    cats = json.loads((RAW / "malignant_patient_categories.json").read_text())
    out = []
    for pid, cat in sorted(cats.items()):
        for ph in PHASES:
            stem = f"{pid}_{cat}_{ph}"
            out.append((f"images/{stem}_0000.nii.gz", DEST / "images" / f"{stem}_0000.nii.gz"))
            out.append((f"labels/{stem}.nii.gz", DEST / "labels" / f"{stem}.nii.gz"))
    return out


def fetch(repo_path: str, dest: Path) -> tuple[str, str]:
    if dest.exists() and dest.stat().st_size > 0:
        return ("skip", repo_path)
    try:
        got = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=repo_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        try:
            dest.hardlink_to(got)
        except OSError:
            import shutil
            shutil.copyfile(got, dest)
        return ("ok", repo_path)
    except Exception as e:                        # missing phase for this patient is normal
        return (f"fail:{type(e).__name__}", repo_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = targets()
    print(f"{len(jobs)} files ({len(jobs)//16} patients x 8 phases x image+label)", flush=True)
    if args.dry_run:
        for r, d in jobs[:8]:
            print(f"  {r} -> {d}")
        return

    counts: dict[str, int] = {}
    missing: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, r, d): r for r, d in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            status, repo_path = f.result()
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if key == "fail":
                missing.append(repo_path)
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}  {counts}", flush=True)

    print(f"\ndone: {counts}")
    if missing:
        (RAW / "missing_files.txt").write_text("\n".join(sorted(missing)) + "\n")
        print(f"{len(missing)} missing -> {RAW/'missing_files.txt'}")
        print("(some patients genuinely lack some phases; verify counts before assuming a bug)")
    sys.exit(0)


if __name__ == "__main__":
    main()
