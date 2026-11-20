"""Order-statistic (first-arrival) timing with any delay distribution (docs/calibration.md, 2.3).

A sensor's pulse time is set by its earliest photons, not by their mean. For a Poisson number of photons with mean
``mu``, each arriving at ``t0 + delay`` with delay CDF ``F`` and density ``f`` (scintillation decay, wavelength
shifting, fibre transit, transit-time spread, all folded into one distribution), the arrivals before ``t`` are
Poisson with mean ``mu F(t - t0)``. The ``k``-th arrival therefore has

    P(T_k <= t) = P(Pois(mu F(t - t0)) >= k)
    density     = mu f(t - t0) Pois(k - 1; mu F(t - t0))

``k = 1`` is the first photon; ``k > 1`` is a discriminator that fires on the ``k``-th photoelectron. The density
integrates to ``P(N >= k)``. A silent sensor (no ``k``-th arrival inside the readout window) contributes
``1 - P(T_k in window)``. Gaussian timing on the mean arrival is biased by the delay shape and by ``mu``; this is
not.

Delays are objects with ``cdf(d)`` and ``pdf(d)`` (and optionally ``log_pdf(d)``) in the backend of ``d``:
:class:`TabulatedDelay` (a histogram: piecewise-constant density, piecewise-linear CDF), :class:`ExponentialDelay`,
:class:`ExpGaussDelay` (exponential decay convolved with Gaussian jitter). The per-channel arrival-time
distribution of :mod:`hepsimlib.expected` has the same interface. Functions run in NumPy or JAX (differentiable
in ``t``, ``t0``, ``mu`` and the delay parameters); ``k`` is a NumPy integer.
"""
import numpy as np

from ._special import backend, floating
from .detection import _log_pois, _threshold, log_p_detect



class TabulatedDelay:
    """A delay histogram: ``edges [T + 1]``, non-negative ``pdf`` (or counts) ``[T]``, normalised here. The CDF is
    linear inside each bin, 0 below the first edge and 1 above the last."""

    def __init__(self, edges, pdf):
        e = np.asarray(edges, np.float64)
        w = np.asarray(pdf, np.float64)
        if e.ndim != 1 or e.size < 2 or np.any(np.diff(e) <= 0) or w.shape != (e.size - 1,) or np.any(w < 0):
            raise ValueError("edges must increase and pdf must be non-negative, one value per bin")
        mass = w * np.diff(e)
        if mass.sum() <= 0:
            raise ValueError("the delay histogram is empty")
        self.edges = e
        self.density = w / mass.sum()
        self.cdf_nodes = np.r_[0.0, np.cumsum(mass) / mass.sum()]

    def cdf(self, d):
        b = backend(d)
        (d,) = floating(b, d)
        xp = b.xp
        return xp.interp(d, xp.asarray(self.edges, d.dtype), xp.asarray(self.cdf_nodes, d.dtype), left=0.0, right=1.0)

    def pdf(self, d):
        b = backend(d)
        (d,) = floating(b, d)
        xp = b.xp
        e = xp.asarray(self.edges, d.dtype)
        i = xp.clip(xp.searchsorted(e, d, side="right") - 1, 0, self.density.size - 1)
        inside = (d >= e[0]) & (d < e[-1])
        return xp.where(inside, xp.asarray(self.density, d.dtype)[i], 0.0)


class ExponentialDelay:
    """``f(d) = exp(-d / tau) / tau`` for ``d >= 0``."""

    def __init__(self, tau):
        self.tau = tau

    def _args(self, d):
        b = backend(d, self.tau)
        return b, floating(b, d, self.tau)

    def cdf(self, d):
        b, (d, tau) = self._args(d)
        return b.xp.where(d > 0, -b.xp.expm1(-b.xp.maximum(d, 0.0) / tau), 0.0)

    def log_pdf(self, d):
        b, (d, tau) = self._args(d)
        return b.xp.where(d >= 0, -b.xp.maximum(d, 0.0) / tau - b.xp.log(tau), -b.xp.inf)

    def pdf(self, d):
        b = backend(d, self.tau)
        return b.xp.exp(self.log_pdf(d))


class ExpGaussDelay:
    """Exponential decay ``tau`` convolved with Gaussian jitter ``sigma`` (exponentially modified Gaussian)::

        f(d) = exp(s^2 / (2 tau^2) - d / tau) Phi(d / s - s / tau) / tau
        F(d) = Phi(d / s) - tau f(d)
    """

    def __init__(self, tau, sigma):
        self.tau, self.sigma = tau, sigma

    def _args(self, d):
        b = backend(d, self.tau, self.sigma)
        return b, floating(b, d, self.tau, self.sigma)

    def _log_tail(self, b, d, tau, s):
        return 0.5 * (s / tau) ** 2 - d / tau + b.log_ndtr(d / s - s / tau)

    def log_pdf(self, d):
        b, (d, tau, s) = self._args(d)
        return self._log_tail(b, d, tau, s) - b.xp.log(tau)

    def pdf(self, d):
        b = backend(d, self.tau, self.sigma)
        return b.xp.exp(self.log_pdf(d))

    def cdf(self, d):
        b, (d, tau, s) = self._args(d)
        return b.ndtr(d / s) - b.xp.exp(self._log_tail(b, d, tau, s))


def _log_pdf(delay, d):
    if hasattr(delay, "log_pdf"):
        return delay.log_pdf(d)
    b = backend(d)
    p = delay.pdf(d)
    pos = p > 0
    return b.xp.where(pos, b.xp.log(b.xp.where(pos, p, 1.0)), -b.xp.inf)


def arrival_log_cdf(t, t0, mu, delay, k=1):
    """log P(T_k <= t)."""
    b = backend(t, t0, mu)
    t, t0, mu = floating(b, t, t0, mu)
    return log_p_detect(mu * delay.cdf(t - t0), k)


def first_arrival_logpdf(t, t0, mu, delay, k=1):
    """log density of the ``k``-th arrival at ``t`` (unconditional: it integrates to P(N >= k))."""
    b = backend(t, t0, mu)
    t, t0, mu = floating(b, t, t0, mu)
    n = _threshold(k)
    d = t - t0
    m = mu * delay.cdf(d)
    xp = b.xp
    return xp.log(mu) + _log_pdf(delay, d) + _log_pois(b, xp.asarray(n - 1, m.dtype), m)


def _log_p_window(t0, mu, delay, k, lo, hi):
    """log P(lo <= T_k <= hi)."""
    b = backend(t0, mu, lo, hi)
    xp = b.xp
    lhi = arrival_log_cdf(hi, t0, mu, delay, k)
    llo = arrival_log_cdf(lo, t0, mu, delay, k)
    return lhi + xp.log(-xp.expm1(xp.minimum(llo - lhi, -xp.finfo(lhi.dtype).tiny)))


def first_arrival_nll(t, t0, mu, delay, k=1, *, window=None, seen=None, conditional=False):
    """Per-sensor NLL of the ``k``-th arrival time.

    ``window=(lo, hi)`` is the readout window (absolute times, like ``t``). ``seen`` (boolean) marks sensors with an
    arrival in it; ``t`` is read only where seen. Unconditional (default): seen ``-log density``, silent
    ``-log(1 - P(T_k in window))``, so the count of seen sensors constrains ``mu``. ``conditional=True``: seen
    sensors only, ``-log(density / P(T_k in window))``, so timing alone, with silent sensors contributing 0.
    Without a window every sensor is seen and the conditional window is the whole line (``P(N >= k)``).
    """
    b = backend(t, t0, mu)
    t, t0, mu = floating(b, t, t0, mu)
    xp = b.xp
    if window is None:
        lp = first_arrival_logpdf(t, t0, mu, delay, k)
        if conditional:
            lp = lp - log_p_detect(mu, k)
        return -lp
    lo, hi = floating(b, *window)
    s = xp.ones(xp.broadcast_shapes(t.shape, t0.shape, mu.shape), bool) if seen is None else xp.asarray(seen)
    t_safe = xp.where(s, t, 0.5 * (lo + hi))
    lp = first_arrival_logpdf(t_safe, t0, mu, delay, k)
    lw = _log_p_window(t0, mu, delay, k, lo, hi)
    if conditional:
        return xp.where(s, -(lp - lw), 0.0)
    return xp.where(s, -lp, -xp.log(-xp.expm1(xp.minimum(lw, -xp.finfo(lw.dtype).tiny))))
