"""Detection as a probability (docs/calibration.md, 2.3).

A fit that stops before digitization does not differentiate a pulse finder or a discriminator. It models
detection as the probability that the number of photoelectrons ``N ~ Poisson(mu)`` reaches an integer threshold
``n``, a smooth function of the expected count ``mu``:

    P(N >= n) = 1 - Q(n, mu),   Q(n, mu) = P(N < n) = sum_{k < n} exp(-mu) mu^k / k!
    d P / d mu = Pois(n - 1; mu)

Both logarithms are computed without cancellation: ``log Q`` as an exact finite log-sum-exp, and ``log P`` from the
Poisson tail series where ``mu`` is small next to ``n`` (where ``1 - Q`` loses digits), else as
``log(-expm1(log Q))``. NumPy (float64, with the analytic derivatives :func:`d_log_p_detect`,
:func:`d_log_p_miss`) or JAX (input dtype, with the same analytic derivative as a custom JVP). Thresholds are
integers >= 1, given as NumPy values (they fix the loop sizes); they broadcast against ``mu``.
"""
import numpy as np

from ._special import backend, floating

TAIL_TERMS = 100       # terms of the tail series, used where mu / (n + 1) < TAIL_RATIO
TAIL_RATIO = 0.7       # 0.7**100 ~ 3e-16


def _threshold(threshold):
    n = np.asarray(threshold)
    if n.dtype.kind == "f":
        if not np.all(n == np.round(n)):
            raise ValueError("thresholds are integer photoelectron counts")
        n = n.astype(np.int64)
    if n.dtype.kind not in "iu" or np.any(n < 1):
        raise ValueError("thresholds must be integers >= 1 (NumPy values, not traced)")
    return n


def _xlogy(b, k, mu):
    """k log mu with 0 log 0 = 0 and k log 0 = -inf for k > 0."""
    xp = b.xp
    pos = mu > 0
    lm = xp.log(xp.where(pos, mu, 1.0))
    return xp.where(pos, k * lm, xp.where(k == 0, 0.0, -xp.inf))


def _log_pois(b, k, mu):
    """log Pois(k; mu), k >= 0 (array, broadcast with mu)."""
    return -mu + _xlogy(b, k, mu) - b.gammaln(k + 1.0)


def _logs(b, mu, n):
    """(log P(N >= n), log P(N < n)): the tail series where mu is small next to n, the exact finite log-sum-exp
    elsewhere; each one computed from the side where it has no cancellation."""
    xp = b.xp
    nf = xp.asarray(n, mu.dtype)
    K = int(np.max(n))
    k = xp.arange(K, dtype=mu.dtype)
    terms = xp.where(k < nf[..., None], _log_pois(b, k, mu[..., None]), -xp.inf)
    lse = xp.minimum(b.logsumexp(terms, axis=-1), 0.0)
    j = xp.arange(TAIL_TERMS, dtype=mu.dtype)
    lmu = xp.log(xp.where(mu > 0, mu, 1.0))[..., None]
    series = b.logsumexp(j * lmu - (b.gammaln(nf[..., None] + j + 1) - b.gammaln(nf[..., None] + 1)), axis=-1)
    tail = xp.minimum(_log_pois(b, nf, mu) + series, 0.0)
    small = mu < TAIL_RATIO * (nf + 1)
    tiny = xp.finfo(mu.dtype).tiny
    log_p = xp.where(small, tail, xp.log(xp.maximum(-xp.expm1(xp.minimum(lse, -tiny)), tiny)))
    tail_safe = xp.where(small, xp.minimum(tail, -tiny), -1.0)      # no log(0) in the unused branch
    log_q = xp.where(small, xp.log1p(-xp.exp(tail_safe)), lse)
    return log_p, log_q


def _log_p(b, mu, n):
    return _logs(b, mu, n)[0]


def _log_q(b, mu, n):
    return _logs(b, mu, n)[1]


def _prep(mu, threshold):
    b = backend(mu)
    (mu,) = floating(b, mu)
    n = np.broadcast_to(_threshold(threshold), np.broadcast_shapes(np.shape(threshold), tuple(mu.shape)))
    mu = b.xp.broadcast_to(mu, n.shape)
    if not b.jax and np.any(mu < 0):
        raise ValueError("expected counts must be non-negative")
    return b, mu, n


def log_p_detect(mu, threshold):
    """log P(N >= threshold) for N ~ Poisson(mu)."""
    b, mu, n = _prep(mu, threshold)
    if not b.jax:
        return _log_p(b, mu, n)
    import jax

    @jax.custom_jvp
    def f(m):
        return _log_p(b, m, n)

    @f.defjvp
    def _(primals, tangents):
        (m,), (dm,) = primals, tangents
        lp = f(m)
        return lp, dm * b.xp.exp(_log_pois(b, b.xp.asarray(n - 1, m.dtype), m) - lp)

    return f(mu)


def log_p_miss(mu, threshold):
    """log P(N < threshold) for N ~ Poisson(mu)."""
    b, mu, n = _prep(mu, threshold)
    if not b.jax:
        return _log_q(b, mu, n)
    import jax

    @jax.custom_jvp
    def f(m):
        return _log_q(b, m, n)

    @f.defjvp
    def _(primals, tangents):
        (m,), (dm,) = primals, tangents
        lq = f(m)
        return lq, -dm * b.xp.exp(_log_pois(b, b.xp.asarray(n - 1, m.dtype), m) - lq)

    return f(mu)


def p_detect(mu, threshold):
    b = backend(mu)
    return b.xp.exp(log_p_detect(mu, threshold))


def d_log_p_detect(mu, threshold):
    """d log P(N >= n) / d mu = Pois(n - 1; mu) / P(N >= n) (NumPy, float64)."""
    b, mu, n = _prep(np.asarray(mu, np.float64), threshold)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return np.exp(_log_pois(b, (n - 1).astype(np.float64), mu) - _log_p(b, mu, n))


def d_log_p_miss(mu, threshold):
    """d log P(N < n) / d mu = -Pois(n - 1; mu) / P(N < n) (NumPy, float64)."""
    b, mu, n = _prep(np.asarray(mu, np.float64), threshold)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return -np.exp(_log_pois(b, (n - 1).astype(np.float64), mu) - _log_q(b, mu, n))


def detection_nll(mu, threshold, seen, *, grad=False):
    """Per-row Bernoulli NLL of seen (1) / silent (0) with ``p = P(N >= threshold)``; ``seen`` may be a fraction
    (expected, Asimov data). ``grad=True`` (NumPy) also returns ``d nll / d mu``."""
    lp, lq = log_p_detect(mu, threshold), log_p_miss(mu, threshold)
    b = backend(lp)
    s = b.xp.asarray(seen, lp.dtype)
    xp = b.xp
    nll = -(xp.where(s > 0, s * lp, 0.0) + xp.where(s < 1, (1 - s) * lq, 0.0))
    if not grad:
        return nll
    if b.jax:
        raise ValueError("grad=True is the NumPy analytic derivative; use jax.grad with JAX inputs")
    with np.errstate(invalid="ignore"):
        g = -(np.where(s > 0, s * d_log_p_detect(mu, threshold), 0.0)
              + np.where(s < 1, (1 - s) * d_log_p_miss(mu, threshold), 0.0))
    return nll, g
