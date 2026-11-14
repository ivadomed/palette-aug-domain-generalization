"""SLAug arm (slaug) SIZING-PROBE diagnostics -- PROBE ONLY, never on a real run's PYTHONPATH.

`slaug.sh record` (PROBE=1) puts this directory right after the pin on each recorded fold command's PYTHONPATH and the probe job body
exports SLAUG_PROBE_DIAG=<unit dir>/diag. Python imports `sitecustomize` at start-up, so in every fold process this module wraps two
torch functions without touching the training code (AugLab pin / nnU-Net):
  * torch.amp.GradScaler.update  -> counts optimizer steps the AMP scaler SKIPPED (found_inf > 0 in any optimizer state) and tracks the
                                    loss scale (a falling scale = repeated skips, the GIN-IPA failure mode of 2026-11-02);
  * torch.nn.utils.clip_grad_norm_ -> counts non-finite total gradient norms (nnU-Net and the SLAug trainer both call it on the
                                    unscaled, accumulated gradient right before the step).
Counters are written every 25 scaler updates (the probe ends by `timeout`, SIGTERM skips atexit) to
$SLAUG_PROBE_DIAG/diag_<host>_<pid>.json. Without SLAUG_PROBE_DIAG it does nothing at all.
"""
import os

_DIAG = os.environ.get("SLAUG_PROBE_DIAG")
if _DIAG:
    try:
        import atexit
        import json
        import math
        import socket
        import sys
        import time

        import torch
        from torch.amp import grad_scaler as _gs

        _PID0 = os.getpid()
        _ST = {"updates": 0, "skipped": 0, "first_skip_at_update": None, "nonfinite_gradnorm": 0, "clip_calls": 0,
               "max_gradnorm": 0.0, "min_scale": None, "last_scale": None, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "argv": sys.argv, "t_start": time.time()}
        os.makedirs(_DIAG, exist_ok=True)
        _PATH = os.path.join(_DIAG, f"diag_{socket.gethostname()}_{_PID0}.json")

        def _dump():
            if os.getpid() != _PID0:  # forked data-loader workers never report
                return
            _ST["t_dump"] = time.time()
            tmp = _PATH + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(_ST, fh)
            os.replace(tmp, _PATH)

        _orig_update = _gs.GradScaler.update

        def _update(self, new_scale=None):
            try:
                found = 0.0
                for state in self._per_optimizer_states.values():
                    for v in state.get("found_inf_per_device", {}).values():
                        found += float(v.item())
                if found > 0:
                    _ST["skipped"] += 1
                    if _ST["first_skip_at_update"] is None:
                        _ST["first_skip_at_update"] = _ST["updates"]
            except Exception:
                pass
            r = _orig_update(self, new_scale)
            _ST["updates"] += 1
            try:
                sc = float(self.get_scale())
                _ST["last_scale"] = sc
                _ST["min_scale"] = sc if _ST["min_scale"] is None else min(_ST["min_scale"], sc)
            except Exception:
                pass
            if _ST["updates"] % 25 == 0:
                _dump()
            return r

        _gs.GradScaler.update = _update
        _orig_clip = torch.nn.utils.clip_grad_norm_

        def _clip(*a, **k):
            n = _orig_clip(*a, **k)
            try:
                _ST["clip_calls"] += 1
                v = float(n)
                if math.isfinite(v):
                    _ST["max_gradnorm"] = max(_ST["max_gradnorm"], v)
                else:
                    _ST["nonfinite_gradnorm"] += 1
            except Exception:
                pass
            return n

        torch.nn.utils.clip_grad_norm_ = _clip
        atexit.register(_dump)
        _dump()
        print(f"[slaug probe diag] GradScaler.update + clip_grad_norm_ wrapped -> {_PATH}", flush=True)
    except Exception as _e:  # diagnostics must never stop a probe fold
        print(f"[slaug probe diag] disabled: {_e!r}", flush=True)
