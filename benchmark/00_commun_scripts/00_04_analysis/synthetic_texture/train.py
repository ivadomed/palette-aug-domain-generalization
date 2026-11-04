"""Train + evaluate one cell (lam, seed, arm) of the synthetic texture experiment.

Writes <out>/lam{lam}_seed{seed}_{arm}.json with per-image Dice (in-domain val, 8 held-out domains).
Within one seed the three arms share the training/val/test images, the network initialisation, the
batch order and the spatial augmentation (separate numpy RNG), so cells are paired across arms.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generator as G  # noqa: E402
import pa_arms  # noqa: E402

N_TRAIN, N_VAL, N_TEST = 300, 50, 100


def conv_block(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.InstanceNorm2d(co, affine=True), nn.LeakyReLU(0.01),
                         nn.Conv2d(co, co, 3, padding=1), nn.InstanceNorm2d(co, affine=True), nn.LeakyReLU(0.01))


class UNet(nn.Module):
    def __init__(self, ch=(16, 32, 64, 128), n_cls=2):
        super().__init__()
        self.enc = nn.ModuleList()
        ci = 1
        for c in ch:
            self.enc.append(conv_block(ci, c))
            ci = c
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(len(ch) - 1, 0, -1)])
        self.dec = nn.ModuleList([conv_block(2 * ch[i - 1], ch[i - 1]) for i in range(len(ch) - 1, 0, -1)])
        self.head = nn.Conv2d(ch[0], n_cls, 1)

    def forward(self, x):
        skips = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for u, d in zip(self.up, self.dec):
            x = u(x)
            x = d(torch.cat([x, skips.pop()], 1))
        return self.head(x)


def dice_loss(logits, y):
    p = logits.softmax(1)[:, 1]
    t = (y[:, 0] == 1).float()
    inter = (p * t).sum((1, 2))
    return 1 - ((2 * inter + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)).mean()


def spatial_aug(rng, x, y):
    """Random flips + 90-degree rotations, identical across arms (numpy RNG)."""
    x, y = x.copy(), y.copy()
    for i in range(x.shape[0]):
        k = rng.integers(4)
        fl = rng.random() < 0.5
        xi, yi = np.rot90(x[i], k), np.rot90(y[i], k)
        if fl:
            xi, yi = xi[:, ::-1], yi[:, ::-1]
        x[i], y[i] = xi, yi
    return x, y


@torch.no_grad()
def predict_dice(model, X, Y, dev, bs=50):
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i:i + bs][:, None]).to(dev)
        pred = model(pa_arms.zscore_fg(x)).argmax(1).cpu().numpy() == 1
        gt = Y[i:i + bs] == 1
        inter = (pred & gt).sum((1, 2))
        denom = pred.sum((1, 2)) + gt.sum((1, 2))
        out.extend(np.where(denom > 0, 2 * inter / np.maximum(denom, 1), 1.0).tolist())
    return out


def git_head():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        return subprocess.check_output(["git", "-C", here, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return os.environ.get("SYNTEX_COMMIT", "unknown")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--arm", choices=["plain", "flat", "texture"], required=True)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-test", action="store_true", help="timing probe: skip evaluation")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    dtr = G.train_domain(a.seed)
    Xtr, Ytr = G.make_set(G.TRAIN_ANAT_SEED(a.seed), N_TRAIN, [dtr], a.lam)
    Xva, Yva = G.make_set(G.VAL_ANAT_SEED(a.seed), N_VAL, [dtr], a.lam)
    tds = G.test_domains()
    Xte, Yte = G.make_set(G.TEST_ANAT_SEED, N_TEST, tds, a.lam)
    Xtr, Xva = Xtr[0], Xva[0]
    t_data = time.time() - t0

    torch.manual_seed(a.seed)
    np_rng = np.random.default_rng(a.seed)  # batch order + spatial aug: identical across arms
    model = UNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: (1 - it / a.iters) ** 0.9)
    tf = pa_arms.make_transform(a.arm)
    torch.manual_seed(10_000 + a.seed)  # transform RNG stream (diverges across arms after use)

    t1 = time.time()
    t_tf = 0.0
    for it in range(a.iters):
        idx = np_rng.choice(N_TRAIN, a.batch, replace=False)
        xb, yb = spatial_aug(np_rng, Xtr[idx], Ytr[idx])
        x = torch.from_numpy(np.ascontiguousarray(xb)[:, None]).to(dev)
        y = torch.from_numpy(np.ascontiguousarray(yb)[:, None].astype(np.int64)).to(dev)
        ts = time.time()
        xin = pa_arms.apply_arm(tf, x, y)
        t_tf += time.time() - ts
        logits = model(xin)
        loss = F.cross_entropy(logits, y[:, 0]) + dice_loss(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if (it + 1) % 500 == 0:
            print(f"it {it + 1} loss {loss.item():.4f} {(time.time() - t1) / (it + 1):.4f}s/it", flush=True)
    t_train = time.time() - t1

    model.eval()
    res = dict(lam=a.lam, seed=a.seed, arm=a.arm, iters=a.iters, batch=a.batch, lr=a.lr,
               commit=git_head(), auglab_pin=open(os.path.join(pa_arms.AUGLAB_PIN, "AUGLAB_COMMIT")).read().strip(),
               t_data=t_data, t_train=t_train, t_transform=t_tf, device=str(dev), final_loss=float(loss.item()))
    if not a.no_test:
        res["val"] = predict_dice(model, Xva, Yva, dev)
        res["test"] = [predict_dice(model, Xte[i], Yte, dev) for i in range(len(tds))]
        res["val_mean"] = float(np.mean(res["val"]))
        res["test_mean"] = float(np.mean([np.mean(t) for t in res["test"]]))
        print(f"val {res['val_mean']:.4f} test {res['test_mean']:.4f}", flush=True)
    fn = os.path.join(a.out, f"lam{a.lam:.1f}_seed{a.seed}_{a.arm}.json")
    with open(fn + ".tmp", "w") as f:
        json.dump(res, f)
    os.replace(fn + ".tmp", fn)
    print("wrote", fn, f"train {t_train:.0f}s transform {t_tf:.0f}s", flush=True)


if __name__ == "__main__":
    main()
