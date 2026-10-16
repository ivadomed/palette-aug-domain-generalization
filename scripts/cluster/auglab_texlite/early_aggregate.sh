#!/usr/bin/env bash
# Vulcan side of the EARLY READ (early_read.sh on TamIA): fetch the throwaway early metrics (md5-verified) to Vulcan scratch, build
# per-setting SCRATCH metrics dirs = symlinks to every run of the real metrics dir (headline table, PA alone, finished arms) + the
# early runs, write one-off configs from the setting's *_texlite_01_results.yaml (metrics_dir + output_dir on scratch, runs without
# metrics dropped) and aggregate them in a CPU run_job. Nothing in the repo results tree is written.
#   bash scripts/cluster/auglab_texlite/early_aggregate.sh fetch
#   bash scripts/cluster/auglab_texlite/early_aggregate.sh aggregate     # -> ${E}/agg/<setting>/out/01_results_summary.md
set -euo pipefail
cd /project/aip-jcohen/paulh/mri_synthesis_project
E=/scratch/paulh/review_20261016/arms_early; TE=/scratch/p/paulh/review_20261016/arms_early; mkdir -p "${E}"
case "${1:?fetch|aggregate|_job}" in
  fetch)
    for tag in $(ssh tamia.alliancecan.ca "cd ${TE} && ls -d */02_metrics | cut -d/ -f1"); do
      mkdir -p "${E}/${tag}"; rm -rf "${E}/${tag}/02_metrics"
      ssh tamia.alliancecan.ca "cd ${TE}/${tag} && tar cf - 02_metrics best_epochs.txt" | tar xf - -C "${E}/${tag}"
      a=$(cd "${E}/${tag}/02_metrics" && find . -type f | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-32)
      b=$(ssh tamia.alliancecan.ca "cd ${TE}/${tag}/02_metrics && find . -type f | LC_ALL=C sort | xargs md5sum | md5sum | cut -c1-32")
      [ "$a" = "$b" ] && echo "[fetch] OK ${tag}" || { echo "[fetch] MISMATCH ${tag}"; exit 1; }
    done ;;
  aggregate)
    source scripts/job_runner/run_job.sh
    RUN_JOB_ACCOUNT=aip-jcohen run_job --name arms_early_agg --gpus 0 --cpus 8 --mem 32G --time 01:00:00 \
        --log "${E}/agg_%j.log" --wait -- bash scripts/cluster/auglab_texlite/early_aggregate.sh _job ;;
  _job)
    .venv/bin/python - "${E}" <<'PY'
import glob, os, sys, yaml
from pathlib import Path
E = Path(sys.argv[1]); T = Path("benchmark/02_tasks")
S = {"onh_t1w": ("brain_healthy/on-harmony", "on-harmony_T1w_texlite_01_results.yaml", "on_harmony_model/T1w"),
     "openms_flair": ("brain_ms/open-ms", "open-ms_flair_texlite_01_results.yaml", "open_ms_model/flair"),
     "chaos_t2spir": ("abdomen_healthy/chaos", "chaos_t2spir_texlite_01_results.yaml", "chaos_model/t2spir")}
for setting, (ds, cfgname, sub) in S.items():
    name = ds.split("/")[1]
    real = T / ds / f"8_results_{name}" / "02_metrics" / sub
    L = E / "agg" / setting / "metrics"; L.mkdir(parents=True, exist_ok=True)
    for x in L.iterdir():
        if x.is_symlink(): x.unlink()
    for x in real.iterdir():
        (L / x.name).symlink_to(x.resolve())
    early = {}
    for d in glob.glob(str(E / f"*_{setting}" / "02_metrics" / sub / "auglab_*")):
        (L / Path(d).name).unlink(missing_ok=True); (L / Path(d).name).symlink_to(d)
        early[Path(d).name[len("auglab_"):]] = open(Path(d).parents[len(sub.split("/")) + 1] / "best_epochs.txt").read().split("\n")
    cfg = yaml.safe_load(open(T / ds / f"5_scripts_{name}/06_evaluate/configs" / cfgname))
    keep = [r for r in cfg["runs"] if any((L / f"{p}{r}").is_dir() for p in ("nnUNet_", "auglab_", ""))]
    cfg["runs"] = keep; cfg["metrics_dir"] = str(L); cfg["output_dir"] = str(E / "agg" / setting / "out")
    cfg["title"] = "EARLY READ (checkpoint_best snapshot; see best_epochs) -- " + cfg["title"]
    out = E / "agg" / setting / "config.yaml"; out.write_text(yaml.safe_dump(cfg, sort_keys=False, width=200, allow_unicode=True))
    print(f"{setting}: {len(keep)} runs (dropped {len(cfg['runs']) and 0})"); [print("  early", k, " | ".join(v)) for k, v in early.items()]
PY
    for s in onh_t1w openms_flair chaos_t2spir; do
      .venv/bin/python benchmark/00_commun_scripts/00_03_evaluate/aggregate_from_config.py "${E}/agg/${s}/config.yaml" > "${E}/agg/${s}/aggregate.log" 2>&1 \
        && echo "[agg] ${s} OK" || { echo "[agg] ${s} FAILED"; tail -5 "${E}/agg/${s}/aggregate.log"; }
    done ;;
esac
