"""Move CT files to the draft-BIDS CT layout (BEP024): sub-X/ct/sub-X[_acq-..]_ct.nii.gz (+ .json),
derivatives/labels/sub-X/ct/sub-X[_acq-..]_ct_label-<name>_(seg|dseg).nii.gz. Pure renames (os.rename on
the same filesystem): no file content changes. Anat folders left empty by the move are removed.

  python scripts/git_annex/rename_ct_bids.py <bids_leaf> [<bids_leaf> ...]           # dry run: print the plan
  python scripts/git_annex/rename_ct_bids.py --apply <bids_leaf> [<bids_leaf> ...]   # do it

Matched: files in sub-*/anat/ and derivatives/labels/sub-*/anat/ whose entity part ends in _CT or _ct
(image `..._CT.nii.gz`/`.json`, mask `..._CT_label-..`). MR files are untouched. Refuses on any collision,
and checks file count and total bytes per leaf before/after.
"""
import os
import re
import sys
from pathlib import Path

PAT = re.compile(r"^(sub-[A-Za-z0-9]+(?:_[a-z]+-[A-Za-z0-9]+)*)_(CT|ct)((?:_label-[A-Za-z0-9]+_(?:seg|dseg))?\.(?:nii\.gz|json))$")


def plan(leaf: Path):
    moves = []
    for anat in list(leaf.glob("sub-*/anat")) + list(leaf.glob("derivatives/labels/sub-*/anat")):
        for f in sorted(anat.iterdir()):
            m = PAT.match(f.name)
            if m:
                moves.append((f, anat.parent / "ct" / f"{m.group(1)}_ct{m.group(3)}"))
    return moves


def tree_stats(leaf: Path):
    n = b = 0
    for root, _, files in os.walk(leaf):
        if "/.git" in root:
            continue
        for x in files:
            n += 1
            b += os.lstat(os.path.join(root, x)).st_size
    return n, b


def main():
    args = sys.argv[1:]
    apply = "--apply" in args
    leaves = [Path(a) for a in args if a != "--apply"]
    for leaf in leaves:
        moves = plan(leaf)
        dsts = [d for _, d in moves]
        problems = [f"exists: {d}" for d in dsts if d.exists()]
        if len(set(dsts)) != len(dsts):
            problems.append("two sources map to one destination")
        kinds = {}
        for s, d in moves:
            k = ("deriv " if "derivatives" in s.parts else "raw ") + re.sub(r"sub-[A-Za-z0-9]+", "sub-X", d.name)
            kinds[k] = kinds.get(k, 0) + 1
        print(f"== {leaf.name}: {len(moves)} files to move")
        for k, v in sorted(kinds.items()):
            print(f"   {v:5d}  {k}")
        if moves:
            print(f"   e.g. {moves[0][0].relative_to(leaf)} -> {moves[0][1].relative_to(leaf)}")
        if problems:
            sys.exit(f"REFUSING {leaf}: {problems[:5]}")
        if not apply:
            continue
        before = tree_stats(leaf)
        for s, d in moves:
            d.parent.mkdir(exist_ok=True)
            os.rename(s, d)
        removed = 0
        for anat in list(leaf.glob("sub-*/anat")) + list(leaf.glob("derivatives/labels/sub-*/anat")):
            if not any(anat.iterdir()):
                anat.rmdir()
                removed += 1
        after = tree_stats(leaf)
        print(f"   applied: files/bytes before {before} after {after}; empty anat dirs removed: {removed}")
        if before != after:
            sys.exit(f"COUNT/BYTES CHANGED in {leaf}")
    if not apply:
        print("dry run only; pass --apply to rename")


if __name__ == "__main__":
    main()
