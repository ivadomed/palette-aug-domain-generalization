# dg_mechanisms: analyses of why domain generalization works or fails

Cross-task analysis scripts for the DG-mechanisms project, a follow-up to PALETTE. They read existing benchmark outputs (per-case `eval_all.csv`, `ladder_series.json`, predictions) through the shared loaders in `../../00_03_evaluate/` and the run-ID rosters, and produce per-cell readout tables (task, training contrast, test contrast): plain-model retention, method gaps, fill-ladder steps, error-type splits.

Rules: scripts are launched through `run_job` wrappers (no inline compute on login nodes); every output records the main-repo and AugLab commits it was produced with; no results, case IDs or patient-level tables are committed here (the repo is public).
