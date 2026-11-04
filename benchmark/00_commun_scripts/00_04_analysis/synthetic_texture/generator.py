"""Procedural 2D 'tissue' images with one target class, rendered in random contrast domains.

Design (see README.md):
  * anatomy: an elliptical body on a zero background, split into K=4 smooth tissue regions
    (argmax of smoothed Gaussian random fields); 1-2 star-shaped targets, each surrounded by
    a ring of the HOST tissue (tissue 0), so the target's only neighbour is the host.
  * every tissue k has a base level b_k and a unit-variance band-pass texture field with its own
    spectrum; the target has its own texture spectrum L (high frequency, oriented).
  * sweep parameter lam in [0, 1] ("texture fraction"):
        target base level = b_host + (1 - lam) * M        (mean step, vanishes at lam = 1)
        target texture    = sqrt(1 - lam) * H + sqrt(lam) * L
    where H is the host texture field itself (same realisation: no seam at lam = 0) and L an
    independent field with the target spectrum. Both fields have unit variance, so the target's
    texture variance equals the host's at every lam: at lam = 1 the target differs from the
    host only by the spatial pattern of its texture (same mean, same marginal variance).
  * a contrast domain is an INTENSITY -> INTENSITY function g_d (piecewise linear through
    knots at the base levels, non-monotone), applied to the base level of every pixel, plus a
    texture gain (signed) per base level, an acquisition noise level and a smooth bias field.
    The target's texture gain is the host's gain, and the target's level is interpolated on
    the same function, so at lam = 1 no domain can create a mean or variance cue for the target;
    at lam = 0 every domain keeps a mean step of |0.15..0.30| (sign random) between target and
    host.
All randomness comes from numpy Generators seeded explicitly, so every image is reproducible.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

SIZE = 128
K_TISSUES = 4
HOST = 0
BASE_LEVELS = np.array([0.2, 0.4, 0.6, 0.8])  # tissue base levels (source "physics")
M_STEP = 0.1  # target base level at lam=0 is b_host + M_STEP (a knot of g_d)
TEX_STD = 0.05  # base texture amplitude (std) before domain gain

# Texture spectra (radial centre frequency in cycles/px, bandwidth, anisotropy, orientation).
# Fixed "world" constants, shared by all seeds and domains.
TISSUE_SPECTRA = [
    dict(f0=0.05, bw=0.02, aniso=0.0, theta=0.0),          # host: coarse isotropic blobs
    dict(f0=0.10, bw=0.03, aniso=1.5, theta=0.0),          # tissue 1
    dict(f0=0.08, bw=0.03, aniso=0.0, theta=0.0),          # tissue 2
    dict(f0=0.15, bw=0.04, aniso=1.5, theta=np.pi / 3),    # tissue 3 (texture close to target's)
]
TARGET_SPECTRUM = dict(f0=0.20, bw=0.04, aniso=2.0, theta=None)  # theta random per image


def bandpass_field(rng, f0, bw, aniso, theta, size=SIZE):
    """Unit-std band-pass Gaussian noise; anisotropy via a von-Mises-like angular term."""
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    fr = np.sqrt(fx ** 2 + fy ** 2)
    env = np.exp(-0.5 * ((fr - f0) / bw) ** 2)
    if aniso > 0:
        phi = np.arctan2(fy, fx)
        env = env * np.exp(aniso * (np.cos(2 * (phi - theta)) - 1))
    env[0, 0] = 0.0
    w = rng.standard_normal((size, size))
    f = np.real(np.fft.ifft2(np.fft.fft2(w) * env))
    return (f - f.mean()) / (f.std() + 1e-8)


def sample_anatomy(rng, size=SIZE):
    """Body mask, tissue map (0..K-1, -1 = background), target mask, texture fields."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    cy, cx = size / 2 + rng.uniform(-4, 4, 2)
    ay, ax = rng.uniform(0.40, 0.47, 2) * size
    body = ((yy - cy) / ay) ** 2 + ((xx - cx) / ax) ** 2 <= 1.0

    fields = np.stack([gaussian_filter(rng.standard_normal((size, size)), 9.0) for _ in range(K_TISSUES)])
    fields[HOST] += 0.01  # tiny prior so the host is not rare
    tissue = np.argmax(fields, axis=0)

    target = np.zeros((size, size), bool)
    ring = np.zeros((size, size), bool)
    n_t = rng.integers(1, 3)
    placed = 0
    for _ in range(50):
        if placed == n_t:
            break
        r0 = rng.uniform(7, 14)
        ty, tx = rng.uniform(0.25, 0.75, 2) * size
        dy, dx = yy - ty, xx - tx
        rr = np.sqrt(dy ** 2 + dx ** 2)
        ang = np.arctan2(dy, dx)
        rad = np.full_like(rr, r0)
        for k in (2, 3, 5):
            rad = rad * (1 + rng.uniform(0, 0.12) * np.cos(k * ang + rng.uniform(0, 2 * np.pi)))
        blob = rr <= rad
        halo = rr <= rad + 6
        if not body[halo].all() or (ring & halo).any():
            continue
        target |= blob
        ring |= halo
        placed += 1
    tissue[ring] = HOST
    tissue[~body] = -1
    target &= body

    tex = np.stack([bandpass_field(rng, **s) for s in TISSUE_SPECTRA])
    ts = dict(TARGET_SPECTRUM)
    ts["theta"] = rng.uniform(0, np.pi)
    tex_target = bandpass_field(rng, **ts)
    bias_field = gaussian_filter(rng.standard_normal((size, size)), 24.0)
    bias_field = bias_field / (np.abs(bias_field).max() + 1e-8)
    noise = rng.standard_normal((size, size))
    return dict(body=body, tissue=tissue, target=target, tex=tex, tex_target=tex_target,
                bias=bias_field, noise=noise)


def sample_domain(rng):
    """Contrast domain: intensity->intensity knots, signed texture gains, noise, bias."""
    g = rng.uniform(0.15, 1.0, K_TISSUES)  # g_d(b_k): random, non-monotone
    step = rng.uniform(0.15, 0.30) * rng.choice([-1.0, 1.0])
    g_t = g[HOST] + step
    if not (0.08 <= g_t <= 1.1):
        g_t = g[HOST] - step
    gains = rng.uniform(0.6, 1.5, K_TISSUES) * np.where(rng.random(K_TISSUES) < 0.3, -1.0, 1.0)
    return dict(g=g, g_target=float(g_t), gains=gains,
                noise_sd=float(rng.uniform(0.005, 0.015)), bias_amp=float(rng.uniform(0.0, 0.1)))


def render(anat, dom, lam):
    """Render one anatomy in one domain at texture fraction lam. Returns (image float32, label uint8)."""
    tissue, target, body = anat["tissue"], anat["target"], anat["body"]
    img = np.zeros(tissue.shape, np.float64)
    for k in range(K_TISSUES):
        m = tissue == k
        img[m] = dom["g"][k] + TEX_STD * dom["gains"][k] * anat["tex"][k][m]
    # target: level interpolated on g_d between the host knot and the target knot (piecewise linear)
    mean_t = dom["g"][HOST] + (1.0 - lam) * (dom["g_target"] - dom["g"][HOST])
    tex_t = np.sqrt(1.0 - lam) * anat["tex"][HOST] + np.sqrt(lam) * anat["tex_target"]
    img[target] = mean_t + TEX_STD * dom["gains"][HOST] * tex_t[target]
    img = img * (1.0 + dom["bias_amp"] * anat["bias"]) + dom["noise_sd"] * anat["noise"]
    img = np.clip(img, 0.03, 1.5)
    img[~body] = 0.0
    return img.astype(np.float32), target.astype(np.uint8)


def make_set(anat_seed, n, domains, lam):
    """n anatomies (seeded by anat_seed) rendered in each domain. Returns X [D,n,H,W], Y [n,H,W]."""
    rng = np.random.default_rng(anat_seed)
    anats = [sample_anatomy(rng) for _ in range(n)]
    X = np.zeros((len(domains), n, SIZE, SIZE), np.float32)
    Y = np.zeros((n, SIZE, SIZE), np.uint8)
    for j, a in enumerate(anats):
        for i, d in enumerate(domains):
            X[i, j], Y[j] = render(a, d, lam)
    return X, Y


# Seed conventions (fixed before any result; README "Seeds").
def train_domain(seed):
    return sample_domain(np.random.default_rng(1000 + seed))


def test_domains(n=8):
    return [sample_domain(np.random.default_rng(9001 + i)) for i in range(n)]


TRAIN_ANAT_SEED = lambda seed: 20000 + seed   # noqa: E731
VAL_ANAT_SEED = lambda seed: 30000 + seed     # noqa: E731
TEST_ANAT_SEED = 40000
