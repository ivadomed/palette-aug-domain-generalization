#!/usr/bin/env python
"""Create and prove the AugLab config of the "SynthSeg-EM (+Auglab; 50%)" competitor arm (ssem50, 2026-10-18).

New config = transform_params_gpu_default01-23_auglabAug_Synthseg_EM.json (SynthSeg-EM + the full Auglab operator set of PA)
with ONLY SynthSeg.probability 1.0 -> 0.5. Written by a one-line TEXT substitution (the first "probability" inside the
"SynthSeg" block) so the file keeps its siblings' hand formatting ([3,5,7], 0.50, blank lines); json.dump would reformat
every line. The result is then checked semantically:
  (a) new vs auglabAug_Synthseg_EM.json: the only difference is SynthSeg.probability 1.0 -> 0.5
  (b) every non-synthesis key of the new config == PA's headline config
      (synthesis blocks: SynthSeg here, ImageContrastV26_6_2GPUTransform in PA)
  (c) headline SynthSeg-EM config (transform_params_gpu_default01-23_Synthseg_EM.json, md5 59f7e8c6 = the file saved in
      every trained headline run) vs new: differs exactly in the six operator probabilities (Scharr, UnsharpMask,
      RandomConv, RedistributeSeg, Inverse, HistogramEqualization) and SynthSeg.probability
  python scripts/cluster/ssem50/make_config.py [--check]     (--check: do not write, only prove the existing file)
"""
import difflib, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CFG = ROOT / "sub-workspaces/auglab_workspace/AugLab/auglab/configs"
SRC = CFG / "transform_params_gpu_default01-23_auglabAug_Synthseg_EM.json"
NEW = CFG / "transform_params_gpu_default01-23_auglabAug_Synthseg_EM_train050.json"
PA = CFG / "transform_params_gpu_default01-23_auglabAug_ImageContrastV26_6_2GPUTransform_train050.json"
HEAD = CFG / "transform_params_gpu_default01-23_Synthseg_EM.json"
SYN = {"SynthSeg", "ImageContrastV26_6_2GPUTransform"}
SIX = {"ScharrTransform", "UnsharpMaskTransform", "RandomConvTransform", "RedistributeSegTransform", "InverseTransform",
       "HistogramEqualizationTransform"}


def flat(d, p=()):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, p + (k,))
        else:
            yield p + (k,), v


def diff(a, b):
    fa, fb = dict(flat(a)), dict(flat(b))
    return sorted((".".join(k), fa.get(k, "<absent>"), fb.get(k, "<absent>")) for k in set(fa) | set(fb) if fa.get(k, "<absent>") != fb.get(k, "<absent>"))


def build():
    text = SRC.read_text()
    m = re.search(r'"SynthSeg"\s*:\s*\{', text)
    head, tail = text[:m.end()], text[m.end():]
    tail2, n = re.subn(r'("probability"\s*:\s*)1\.0\b', r'\g<1>0.5', tail, count=1)
    assert n == 1, "SynthSeg probability 1.0 not found"
    out = head + tail2
    want = json.loads(text); want["SynthSeg"]["probability"] = 0.5
    assert json.loads(out) == want, "text substitution does not equal the json-level edit"
    # the substituted line must sit inside the SynthSeg block (before its closing brace)
    blk_end = out.index("\n    },", m.end())
    assert out.index('"probability": 0.5', m.end()) < blk_end
    return out


def main():
    check = "--check" in sys.argv
    out = build()
    if check:
        assert NEW.read_text() == out, f"{NEW.name} on disk differs from the generated content"
        print(f"[check] {NEW.name} == generated content")
    else:
        if NEW.exists() and NEW.read_text() != out:
            raise SystemExit(f"{NEW} exists with different content -- refusing to overwrite")
        NEW.write_text(out); print(f"wrote {NEW.relative_to(ROOT)}")
    src, new, pa, hd = (json.loads(p.read_text()) for p in (SRC, NEW, PA, HEAD))
    ok = True
    print("\n== unified text diff auglabAug_Synthseg_EM.json -> new ==")
    sys.stdout.writelines(difflib.unified_diff(SRC.read_text().splitlines(True), NEW.read_text().splitlines(True), SRC.name, NEW.name, n=1))
    d = diff(src, new); print(f"\n(a) semantic diff vs {SRC.name}: {d}")
    ok &= d == [("SynthSeg.probability", 1.0, 0.5)]
    pa_ops = {k: v for k, v in pa.items() if k not in SYN}; new_ops = {k: v for k, v in new.items() if k not in SYN}
    d = diff(pa_ops, new_ops)
    print(f"(b) non-synthesis keys: PA {len(pa_ops)} blocks, new {len(new_ops)} blocks, same block set: {set(pa_ops) == set(new_ops)}; "
          f"semantic diff: {d}")
    print(f"    synthesis blocks: PA {sorted(set(pa) & SYN)} (probability {pa['ImageContrastV26_6_2GPUTransform']['probability']}), "
          f"new {sorted(set(new) & SYN)} (probability {new['SynthSeg']['probability']})")
    ok &= d == [] and set(pa) - SYN == set(new) - SYN and set(pa) & SYN == {"ImageContrastV26_6_2GPUTransform"} and set(new) & SYN == {"SynthSeg"}
    d = diff(hd, new); print(f"(c) semantic diff {HEAD.name} -> new:")
    for row in d:
        print("    ", row)
    blocks = {k.split(".")[0] for k, _, _ in d}; keys = {k.split(".", 1)[1] for k, _, _ in d}
    ok &= blocks == SIX | {"SynthSeg"} and keys == {"probability"} and len(d) == 7
    ok &= dict((k, b) for k, a, b in d)["SynthSeg.probability"] == 0.5
    # the SynthSeg generator block itself (everything but probability) is unchanged vs the headline config
    ok &= {k: v for k, v in hd["SynthSeg"].items() if k != "probability"} == {k: v for k, v in new["SynthSeg"].items() if k != "probability"}
    print("ALL PROOFS PASSED" if ok else "PROOF FAILED"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
