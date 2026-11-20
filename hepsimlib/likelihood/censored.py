"""Censored per-crossing likelihood: seen or silent, with fitted widths (docs/calibration.md, 2.3).

For every expected crossing of a channel (a track through a scintillator bar, a source event in a crystal) the data
say either *seen*, with a selected pulse of charge ``q`` and time ``t``, or *silent*. A likelihood that uses only
the seen pulses is biased wherever the selection cuts into the response. Here a silent crossing contributes the
probability of missing it, and the widths are parameters.

Response model of one crossing, given five predictors (from any forward: an analytic surrogate, or the expected
response of a table library, :mod:`hepsimlib.expected`)::

    log q      ~ Normal(mu_logq, sigma_q)
    t | log q  ~ Normal(mu_t - walk * (log q - logq_ref), sigma_t)

with ``sigma = exp(log_sigma)``. The selection keeps ``log q >= log_threshold`` and ``t_lo <= t <= t_hi``.

- seen: ``-log`` of the joint density of ``(log q, t)``;
- silent: ``-log P(miss)``. ``(log q, t)`` is bivariate normal: with ``X = (log q - mu_logq) / sigma_q`` and
  ``Y = (t - c) / s``, where ``c = mu_t - walk (mu_logq - logq_ref)`` and ``s^2 = walk^2 sigma_q^2 + sigma_t^2``,
  the correlation is ``rho = -walk sigma_q / s``. Then
  ``P(miss) = Phi(a) + P(X >= a, Y < L) + P(X >= a, Y > U)``, three non-negative terms with no cancellation. The
  two orthant probabilities come from Genz's bivariate-normal algorithm (A. Genz, Statistics and Computing 14
  (2004) 251-260), reimplemented here for NumPy and JAX. They are accurate to about 1e-15 absolute, so the miss
  probability is exact at double precision: no quadrature over the charge.

The value runs in NumPy (float64, SciPy) or JAX (the input dtype, differentiable). ``grad=True`` (NumPy) returns the
analytic gradient with respect to the five predictors, ``[n, 5]`` in the order of :class:`CrossingPrediction`. It
comes from the closed-form derivatives of the bivariate normal and is the exact derivative of the value. The
caller chains it to its own parameters, for example per-channel gains on ``mu_logq`` and offsets on ``mu_t``.
"""
from typing import NamedTuple

import numpy as np

from ._special import backend, floating

LOG_SQRT_2PI = float(0.5 * np.log(2 * np.pi))   # Python floats: weakly typed in JAX
PREDICTORS = ("mu_logq", "mu_t", "log_sigma_q", "log_sigma_t", "walk")
_GL_X, _GL_W = np.polynomial.legendre.leggauss(20)
CLIP = 38.0     # |h|, |k| beyond this are infinite for every probability here


class CrossingPrediction(NamedTuple):
    """The five predictors per crossing (arrays or scalars, broadcast) and the walk reference log charge."""
    mu_logq: object
    mu_t: object
    log_sigma_q: object
    log_sigma_t: object
    walk: object = 0.0
    logq_ref: object = 0.0


class Selection(NamedTuple):
    """The frozen pulse selection per crossing: log charge threshold and time window (infinite bounds allowed)."""
    log_threshold: object
    t_lo: object
    t_hi: object


def bvn_upper(h, k, r):
    """P(X > h, Y > k) for a standard bivariate normal with correlation ``r`` (|r| < 1), NumPy or JAX.

    Genz's algorithm: a 20-point Gauss-Legendre rule over the correlation for |r| < 0.925 (Drezner-Wesolowsky), and
    the expansion about |r| = 1 with a Gauss-Legendre correction otherwise; both are evaluated and selected
    elementwise, so the function is jit- and grad-safe."""
    b = backend(h, k, r)
    h, k, r = floating(b, h, k, r)
    xp = b.xp
    h, k = xp.clip(h, -CLIP, CLIP), xp.clip(k, -CLIP, CLIP)
    dt = h.dtype
    x = xp.asarray(1 + _GL_X, dt)          # nodes on (0, 2)
    w = xp.asarray(_GL_W, dt)
    tp = float(2 * np.pi)
    hk = h * k
    # |r| < 0.925
    hs = (h * h + k * k) / 2
    asr = xp.arcsin(r) / 2
    sn = xp.sin(asr[..., None] * x)
    low = xp.sum(w * xp.exp((sn * hk[..., None] - hs[..., None]) / (1 - sn * sn)), axis=-1) * asr / tp \
        + b.ndtr(-h) * b.ndtr(-k)
    # |r| >= 0.925 (computed with safe values everywhere, selected below)
    big = xp.abs(r) >= 0.925
    rr = xp.where(big, r, 0.95)
    kk = xp.where(rr < 0, -k, k)
    hk2 = h * kk
    as_ = (1 - rr) * (1 + rr)
    a = xp.sqrt(as_)
    bs = (h - kk) ** 2
    asr2 = -(bs / as_ + hk2) / 2
    c = (4 - hk2) / 8
    d = (12 - hk2) / 80
    t1 = xp.where(asr2 > -100, a * xp.exp(xp.maximum(asr2, -100.0)) * (1 - c * (bs - as_) * (1 - d * bs) / 3
                                                                       + c * d * as_ * as_), 0.0)
    bpos = bs > 0
    bb = xp.sqrt(xp.where(bpos, bs, 1.0))
    bb = xp.where(bpos, bb, 0.0)
    sp = float(np.sqrt(tp)) * b.ndtr(-bb / a)
    t2 = xp.where(hk2 > -100, xp.exp(-xp.clip(hk2, -100.0, 100.0) / 2) * sp * bb * (1 - c * bs * (1 - d * bs) / 3), 0.0)
    bvn = t1 - t2
    ah = a / 2
    xs = (ah[..., None] * x) ** 2
    asr3 = -(bs[..., None] / xs + hk2[..., None]) / 2
    sp3 = 1 + c[..., None] * xs * (1 + 5 * d[..., None] * xs)
    rs = xp.sqrt(1 - xs)
    ep = xp.exp(-(hk2[..., None] / 2) * xs / (1 + rs) ** 2) / rs
    term = xp.where(asr3 > -100, xp.exp(xp.maximum(asr3, -100.0)) * (sp3 - ep), 0.0)
    bvn = (ah * xp.sum(w * term, axis=-1) - bvn) / tp
    pos = bvn + b.ndtr(-xp.maximum(h, kk))
    lneg = xp.where(h < 0, b.ndtr(kk) - b.ndtr(h), b.ndtr(-h) - b.ndtr(-kk))
    neg = xp.where(h >= kk, -bvn, lneg - bvn)
    high = xp.where(rr > 0, pos, neg)
    return xp.clip(xp.where(big, high, low), 0.0, 1.0)


def _bvn_upper_grad(h, k, r):
    """Partial derivatives of :func:`bvn_upper` in h, k and r (NumPy, closed form)."""
    from scipy.special import ndtr
    q = np.sqrt((1 - r) * (1 + r))
    ph = np.exp(-0.5 * h * h - LOG_SQRT_2PI)
    pk = np.exp(-0.5 * k * k - LOG_SQRT_2PI)
    dh = -ph * ndtr((r * h - k) / q)
    dk = -pk * ndtr((r * k - h) / q)
    dr = np.exp(-(h * h - 2 * r * h * k + k * k) / (2 * q * q)) / (2 * np.pi * q)
    return dh, dk, dr


def _prepare(pred, sel, seen, logq, t):
    b = backend(*pred, *sel, logq, t)
    arrs = floating(b, *pred, *sel, logq, t)
    shape = np.broadcast_shapes(*(tuple(np.shape(a)) for a in arrs), np.shape(seen))
    arrs = [b.xp.broadcast_to(a, shape) for a in arrs]
    s = b.xp.broadcast_to(b.xp.asarray(seen), shape)
    if s.dtype != bool:
        raise ValueError("seen must be boolean")
    return b, arrs[:6], arrs[6:9], s, arrs[9], arrs[10]


def _miss_parts(b, mq, mt, sq, st, walk, ref, lthr, lo, hi):
    xp = b.xp
    a = (lthr - mq) / sq
    c = mt - walk * (mq - ref)
    s = xp.sqrt((walk * sq) ** 2 + st ** 2)
    rho = -walk * sq / s
    Lp = xp.clip((lo - c) / s, -CLIP, CLIP)
    Up = xp.clip((hi - c) / s, -CLIP, CLIP)
    B1 = bvn_upper(a, -Lp, -rho)       # P(X >= a, Y < L)
    B2 = bvn_upper(a, Up, rho)         # P(X >= a, Y > U)
    p_miss = b.ndtr(a) + B1 + B2
    return p_miss, dict(a=a, c=c, s=s, rho=rho, L=Lp, U=Up)


def censored_nll(pred, sel, seen, logq=None, t=None, *, grad=False):
    """Per-crossing NLL (module docstring). ``seen`` boolean; ``logq`` and ``t`` are read on seen crossings only
    (anything, NaN included, elsewhere). ``grad=True`` (NumPy): also ``d nll / d predictors``, ``[..., 5]``."""
    logq = np.nan if logq is None else logq
    t = np.nan if t is None else t
    b, (mq, mt, lsq, lst, walk, ref), (lthr, lo, hi), s, lq, tm = _prepare(pred, sel, seen, logq, t)
    xp = b.xp
    sq, st = xp.exp(lsq), xp.exp(lst)
    # seen: joint density (safe values on silent rows)
    lq_s = xp.where(s, lq, mq)
    tm_s = xp.where(s, tm, mt)
    z = (lq_s - mq) / sq
    r = (tm_s - mt + walk * (lq_s - ref)) / st
    seen_term = 0.5 * (z * z + r * r) + lsq + lst + 2 * LOG_SQRT_2PI
    # silent: probability of a miss (a floor keeps the log finite where it rounds to zero)
    p_miss, p = _miss_parts(b, mq, mt, sq, st, walk, ref, lthr, lo, hi)
    nll = xp.where(s, seen_term, -xp.log(xp.maximum(p_miss, xp.finfo(p_miss.dtype).tiny)))
    if not grad:
        return nll
    if b.jax:
        raise ValueError("grad=True is the NumPy analytic gradient; use jax.grad with JAX inputs")
    g = np.zeros(nll.shape + (5,))
    g[..., 0] = np.where(s, -z / sq, 0.0)
    g[..., 1] = np.where(s, -r / st, 0.0)
    g[..., 2] = np.where(s, 1 - z * z, 0.0)
    g[..., 3] = np.where(s, 1 - r * r, 0.0)
    g[..., 4] = np.where(s, r * (lq_s - ref) / st, 0.0)
    a, c, sd, rho, Lp, Up = (p[k] for k in ("a", "c", "s", "rho", "L", "U"))
    # derivatives of the pieces in the predictors (mu_logq, mu_t, log_sigma_q, log_sigma_t, walk)
    zero = np.zeros_like(a)
    da = [-1 / sq, zero, -a, zero, zero]
    dc = [-walk, np.ones_like(a), zero, zero, -(mq - ref)]
    ds = [zero, zero, (walk * sq) ** 2 / sd, st ** 2 / sd, walk * sq ** 2 / sd]
    drho = [zero, zero, rho * (1 - (walk * sq / sd) ** 2), -rho * (st / sd) ** 2, -sq / sd - rho * walk * sq ** 2 / sd ** 2]
    inL = np.abs((lo - c) / sd) < CLIP
    inU = np.abs((hi - c) / sd) < CLIP
    phi_a = np.exp(-0.5 * a * a - LOG_SQRT_2PI)
    h1, k1, r1 = _bvn_upper_grad(a, -Lp, -rho)
    h2, k2, r2 = _bvn_upper_grad(a, Up, rho)
    pm = np.maximum(p_miss, np.finfo(np.float64).tiny)
    for j in range(5):
        dL = np.where(inL, (-dc[j] - Lp * ds[j]) / sd, 0.0)
        dU = np.where(inU, (-dc[j] - Up * ds[j]) / sd, 0.0)
        dp = phi_a * da[j] + h1 * da[j] - k1 * dL - r1 * drho[j] + h2 * da[j] + k2 * dU + r2 * drho[j]
        g[..., j] = np.where(s, g[..., j], -dp / pm)
    return nll, g


def p_selected(pred, sel):
    """Probability that a crossing gives a selected pulse, ``1 - P(miss)`` (either backend)."""
    b = backend(*pred, *sel)
    xp = b.xp
    _, (mq, mt, lsq, lst, walk, ref), (lthr, lo, hi), _, _, _ = _prepare(pred, sel, False, np.nan, np.nan)
    p_miss, _ = _miss_parts(b, mq, mt, xp.exp(lsq), xp.exp(lst), walk, ref, lthr, lo, hi)
    return 1 - p_miss
