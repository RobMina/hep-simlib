"""Per-channel gains and time offsets: closed-form profiling, Schur elimination, gauge and Cramér-Rao errors.

The calibration problem shared by scintillator, crystal, PMT and SiPM detectors (docs/calibration.md, 2.1): rows
``i`` (hits, crossings, source events) carry a channel ``c(i)``, an expected charge ``M_i(theta)`` and/or an
expected time ``T_i(theta)`` from a forward model with a few global parameters ``theta``. Each channel adds

    charge_i = exp(g_c) * M_i(theta)      (multiplicative gain, log parametrised)
    time_i   = T_i(theta) + o_c           (additive time offset)

with ``C`` channels (thousands). The nuisances ``nu_c = (g_c, o_c)`` enter only their own channel's rows, so the
Gauss-Newton (Fisher) matrix is block-arrow: a dense ``theta`` block, per-channel ``m x m`` blocks (``m`` = 1 or 2)
and their couplings. Every step and every covariance eliminates the channel blocks by the Schur complement
``S = H_tt - sum_c H_tc H_cc^-1 H_ct``, in ``O(n p m + C m^3 + p^3)``.

- **Separable models** (the default): the library applies the gain and offset itself, so they are profiled in
  closed form after each ``theta`` step (variable projection): Poisson ``exp(g_c) = sum Q / sum M``, Gaussian
  ``sum w Q M / sum w M^2``, offset ``sum v (t - T) / sum v``.
- **Coupled models** (``coupled=True``): the model applies the nuisances itself (time walk on the charge, for
  example) as ``model(theta, aux, log_gain_rows, offset_rows)``. Each iteration is then the full damped
  Gauss-Newton step on ``(theta, nu)``, with the per-channel blocks eliminated.
- **Gauge.** A global scale (time) in ``theta`` that enters exactly like a common gain (offset) is degenerate with
  them. Name its index in ``gauge``: it is held during the fit, and at the end the weighted mean of the channel
  terms is moved into it, so the reported channel terms sum to zero (weights: equal, or the channel information to
  avoid the Jensen bias of dim channels). The NLL change of the move is recorded; it is zero when the declared
  parameter really is degenerate.
- **Errors** (:func:`channel_crb`): the ``theta`` covariance with the nuisances marginalised, ``S^-1``, and the
  per-channel errors from the block formulas, gauge included, without forming the ``(p + C m)^2`` matrix.
"""
import warnings
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from . import _backend as B
from .optimize import GNTerms, _fixed_mask, _quiet, lbfgs

FAMILIES = ("poisson", "gaussian")


def _xp(x):
    if B.is_jax(x):
        import jax.numpy as jnp
        return jnp
    return np


# ------------------------------------------------------------------------------------------------ closed forms
def _channel_index(channel, n_channels):
    ch = np.asarray(channel)
    if ch.ndim != 1 or ch.dtype.kind not in "iu":
        raise ValueError("channel must be a 1D integer array")
    if ch.size and (ch.min() < 0 or ch.max() >= n_channels):
        raise ValueError(f"channel indices must be in [0, {n_channels})")
    return ch.astype(np.int64)


def _sigma(sigma, shape):
    s = np.broadcast_to(np.asarray(1.0 if sigma is None else sigma, np.float64), shape)
    if not np.all(np.isfinite(s) & (s > 0)):
        raise ValueError("sigma must be finite and positive")
    return s


def profile_log_gain(channel, observed, expected, n_channels, family="poisson", sigma=None):
    """Closed-form log gain per channel for ``observed ~ exp(g_c) * expected``; NaN for a channel without rows.

    Poisson: ``log(sum Q / sum M)`` (the exact maximum-likelihood gain). Gaussian with errors ``sigma``:
    ``log(sum w Q M / sum w M^2)``, ``w = sigma**-2`` (NaN if not positive).
    """
    ch = _channel_index(channel, n_channels)
    Q, M = np.asarray(observed, np.float64), np.asarray(expected, np.float64)
    if family == "poisson":
        num = np.bincount(ch, Q, minlength=n_channels)
        den = np.bincount(ch, M, minlength=n_channels)
    elif family == "gaussian":
        w = _sigma(sigma, Q.shape) ** -2
        num = np.bincount(ch, w * Q * M, minlength=n_channels)
        den = np.bincount(ch, w * M * M, minlength=n_channels)
    else:
        raise ValueError(f"family must be one of {FAMILIES}")
    with np.errstate(divide="ignore", invalid="ignore"):
        k = num / den
        return np.where((den > 0) & (k > 0), np.log(np.where(k > 0, k, 1.0)), np.nan)


def profile_offset(channel, observed_t, expected_t, n_channels, sigma=None):
    """Closed-form additive offset per channel, the weighted mean of ``t - T`` (``w = sigma**-2``); NaN without rows."""
    ch = _channel_index(channel, n_channels)
    t, T = np.asarray(observed_t, np.float64), np.asarray(expected_t, np.float64)
    v = _sigma(sigma, t.shape) ** -2
    den = np.bincount(ch, v, minlength=n_channels)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, np.bincount(ch, v * (t - T), minlength=n_channels) / den, np.nan)


def gauge(values, weights=None):
    """Sum-to-zero gauge: ``(shift, values - shift)`` with ``shift`` the (weighted) mean, which a degenerate global
    parameter absorbs. NaN values (channels without data) are ignored in the mean."""
    v = np.asarray(values, np.float64)
    w = np.ones_like(v) if weights is None else np.asarray(weights, np.float64)
    ok = np.isfinite(v) & (w > 0)
    if not ok.any():
        raise ValueError("no finite value with positive weight")
    shift = float(np.sum(w[ok] * v[ok]) / np.sum(w[ok]))
    return shift, v - shift


# ------------------------------------------------------------------------------------------------ the problem
@dataclass
class Observed:
    """An observable per row: ``values``, a likelihood ``family`` (``"poisson"`` counts or ``"gaussian"``) and, for
    Gaussian, ``sigma`` (scalar or per row)."""
    values: np.ndarray
    family: str = "gaussian"
    sigma: object = None

    def __post_init__(self):
        self.values = np.asarray(self.values, np.float64)
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("observed values must be finite")
        if self.family == "poisson":
            if np.any(self.values < 0):
                raise ValueError("Poisson counts must be non-negative")
            self.sigma = None
        else:
            self.sigma = _sigma(self.sigma, self.values.shape)

    def terms(self, mu):
        """Per-row NLL, d NLL / d mu and the Fisher weight (float64)."""
        y = self.values
        if self.family == "gaussian":
            s2 = self.sigma ** 2
            r = mu - y
            return 0.5 * r * r / s2, r / s2, 1 / s2
        with np.errstate(divide="ignore", invalid="ignore"):
            pos = mu > 0
            nll = np.where(pos, mu - np.where(y > 0, y * np.log(np.where(pos, mu, 1.0)), 0.0),
                           np.where((mu == 0) & (y == 0), 0.0, np.inf))
            w = np.where(pos, 1 / np.where(pos, mu, 1.0), 0.0)
            return nll, np.where(pos, 1 - y * w, 1.0), w


@dataclass
class ChannelProblem:
    """Rows with a channel index and a forward model (module docstring).

    ``model(theta, aux) -> {"charge": M [n], "time": T [n]}`` (separable), or, with ``coupled=True``,
    ``model(theta, aux, log_gain_rows, offset_rows)`` returning the predictions with the nuisances applied (an
    absent nuisance is passed as None). ``jac``: ``"fwd"``, ``"rev"``, ``"fd"`` or, for a separable model, a
    callable ``jac(theta, aux) -> {"charge": [n, p], "time": [n, p]}`` of the raw outputs. ``dtype``: the dtype
    ``theta`` and the nuisance rows are cast to before the model is called (float32 or float64).
    """
    channel: np.ndarray
    n_channels: int
    model: object
    charge: Observed = None
    time: Observed = None
    gain: bool = True
    offset: bool = True
    coupled: bool = False
    aux: object = None
    jac: object = None
    dtype: object = None
    kinds: tuple = field(init=False)

    def __post_init__(self):
        self.n_channels = int(self.n_channels)
        self.channel = _channel_index(self.channel, self.n_channels)
        n = self.channel.size
        if self.charge is None and self.time is None:
            raise ValueError("a channel problem needs a charge or a time observable")
        if self.gain and self.charge is None:
            raise ValueError("gain=True needs a charge observable")
        if self.offset and self.time is None:
            raise ValueError("offset=True needs a time observable")
        for name in ("charge", "time"):
            o = getattr(self, name)
            if o is not None and o.values.shape != (n,):
                raise ValueError(f"{name} values must have one entry per row")
        self.kinds = tuple(k for k, on in (("log_gain", self.gain), ("offset", self.offset)) if on)
        if self.jac is None:
            self.jac = "fwd" if B.have_jax() else "fd"
        if callable(self.jac) and self.coupled:
            raise ValueError("a callable jac is for separable models; use 'fwd' or 'fd' for coupled ones")
        self.dtype = B.check_dtype(self.dtype, self.jac if isinstance(self.jac, str) else None)
        self.observables = tuple(k for k in ("charge", "time") if getattr(self, k) is not None)
        if self.kinds:
            counts = np.bincount(self.channel, minlength=self.n_channels)
            if np.any(counts == 0):
                bad = np.flatnonzero(counts == 0)
                raise ValueError(f"{bad.size} channel(s) have no rows (first: {bad[:5].tolist()}); renumber the "
                                 "channels or drop them from n_channels")

    @property
    def m(self):
        return len(self.kinds)

    def kind_index(self, kind):
        return self.kinds.index(kind) if kind in self.kinds else None

    def zeros(self):
        return np.zeros((self.n_channels, self.m))


class ChannelBlocks(NamedTuple):
    """Gauss-Newton blocks at one point (float64): NLL, gradients and the block-arrow information."""
    nll: float
    g_t: np.ndarray        # [p]
    g_n: np.ndarray        # [C, m]
    H_tt: np.ndarray       # [p, p]
    H_tn: np.ndarray       # [C, p, m]
    H_nn: np.ndarray       # [C, m, m]

    def dense(self, free=None):
        """The full information over (theta[free], nu flattened channel-major) and the gradient: for tests."""
        p = self.g_t.size
        f = np.ones(p, bool) if free is None else np.asarray(free, bool)
        C, m = self.g_n.shape
        pf = int(f.sum())
        H = np.zeros((pf + C * m, pf + C * m))
        H[:pf, :pf] = self.H_tt[np.ix_(f, f)]
        for c in range(C):
            sl = slice(pf + c * m, pf + (c + 1) * m)
            H[:pf, sl] = self.H_tn[c][f]
            H[sl, :pf] = self.H_tn[c][f].T
            H[sl, sl] = self.H_nn[c]
        return H, np.r_[self.g_t[f], self.g_n.ravel()]


# ------------------------------------------------------------------------------------------------ evaluation
def _rows(prob, nu):
    out = {}
    for k in ("log_gain", "offset"):
        j = prob.kind_index(k)
        out[k] = None if j is None else nu[prob.channel, j]
    return out


def _apply(prob, raw, rows):
    """Final predictions (dict) from the raw separable outputs and the nuisance rows (same backend as raw)."""
    out = {}
    if "charge" in prob.observables:
        q = raw["charge"]
        if rows["log_gain"] is not None:
            xp = _xp(q)
            q = q * xp.exp(xp.asarray(rows["log_gain"], q.dtype))
        out["charge"] = q
    if "time" in prob.observables:
        t = raw["time"]
        if rows["offset"] is not None:
            xp = _xp(t)
            t = t + xp.asarray(rows["offset"], t.dtype)
        out["time"] = t
    return out


def _cast(prob, theta):
    return np.asarray(theta, np.float64).astype(prob.dtype)


def _rows_cast(prob, rows):
    return {k: None if v is None else v.astype(prob.dtype) for k, v in rows.items()}


def _predict(prob, theta, nu):
    """(predictions, raw) as float64 NumPy dicts; raw is None for coupled models."""
    rows = _rows_cast(prob, _rows(prob, nu))
    with B.highest_precision():
        if prob.coupled:
            pred = prob.model(_cast(prob, theta), prob.aux, rows["log_gain"], rows["offset"])
            raw = None
        else:
            raw = prob.model(_cast(prob, theta), prob.aux)
            pred = _apply(prob, raw, rows)
    pred = {k: B.host64(pred[k]) for k in prob.observables}
    raw = None if raw is None else {k: B.host64(raw[k]) for k in prob.observables}
    return pred, raw


def nll(prob, theta, nu):
    pred, _ = _predict(prob, theta, nu)
    total = 0.0
    for k in prob.observables:
        total += float(np.sum(getattr(prob, k).terms(pred[k])[0]))
    return total


def _flat(prob, out):
    xp = _xp(out[prob.observables[0]])
    return xp.concatenate([xp.reshape(out[k], (-1,)) for k in prob.observables])


def _jacobians(prob, theta, nu, pred):
    """d pred / d theta [n_obs * n, p] and d pred / d nu_row [n_obs * n, m] (float64), observables stacked."""
    n = prob.channel.size
    rows = _rows_cast(prob, _rows(prob, nu))
    if callable(prob.jac):
        rj = prob.jac(_cast(prob, theta), prob.aux)
        cols = []
        for k in prob.observables:
            J = B.host64(rj[k])
            if k == "charge" and rows["log_gain"] is not None:
                J = J * np.exp(rows["log_gain"].astype(np.float64))[:, None]
            cols.append(J)
        Jt = np.concatenate(cols)
    else:
        if prob.coupled:
            def fn(th):
                return _flat(prob, prob.model(th, prob.aux, rows["log_gain"], rows["offset"]))
        else:
            def fn(th):
                return _flat(prob, _apply(prob, prob.model(th, prob.aux), rows))
        Jt = B.host64(B.jacobian(fn, theta, prob.jac, dtype=prob.dtype))
    Jn = np.zeros((len(prob.observables) * n, prob.m))
    if not prob.coupled:
        for a, kind in enumerate(prob.kinds):
            obs = "charge" if kind == "log_gain" else "time"
            o = prob.observables.index(obs)
            Jn[o * n:(o + 1) * n, a] = pred["charge"] if kind == "log_gain" else 1.0
        return Jt, Jn
    th = _cast(prob, theta)
    for a, kind in enumerate(prob.kinds):
        other = "offset" if kind == "log_gain" else "log_gain"

        def fk(v, kind=kind, other=other):
            args = {kind: v, other: rows[other]}
            return _flat(prob, prob.model(th, prob.aux, args["log_gain"], args["offset"]))

        if prob.jac in ("fwd", "rev"):
            import jax
            import jax.numpy as jnp
            th = jnp.asarray(_cast(prob, theta))      # models dispatch on theta's backend
            v0 = jnp.asarray(rows[kind])
            with B.highest_precision():
                Jn[:, a] = B.host64(jax.jvp(fk, (v0,), (jnp.ones_like(v0),))[1])
        else:   # rows are independent: perturb every row's nuisance at once
            h = B.fd_step(np.zeros(1), prob.dtype)[0]
            v = rows[kind].astype(np.float64)
            Jn[:, a] = (B.host64(fk((v + h).astype(prob.dtype))) - B.host64(fk((v - h).astype(prob.dtype)))) / (2 * h)
    return Jt, Jn


def blocks(prob, theta, nu):
    """:class:`ChannelBlocks` at ``(theta, nu)``; ``nu`` is ``[C, m]`` in the order of ``prob.kinds``."""
    theta = np.asarray(theta, np.float64)
    nu = np.asarray(nu, np.float64).reshape(prob.n_channels, prob.m)
    pred, _ = _predict(prob, theta, nu)
    n, C, m, p = prob.channel.size, prob.n_channels, prob.m, theta.size
    nll_parts, d, w = [], [], []
    for k in prob.observables:
        a, b, c = getattr(prob, k).terms(pred[k])
        nll_parts.append(np.sum(a))
        d.append(b)
        w.append(c)
    d, w = np.concatenate(d), np.concatenate(w)
    total = float(np.sum(nll_parts))
    Jt, Jn = _jacobians(prob, theta, nu, pred)
    ch = np.tile(prob.channel, len(prob.observables))
    H_tt, _ = B.normal_equations(Jt, None, w)
    g_t = Jt.T @ d
    g_n = np.stack([np.bincount(ch, Jn[:, a] * d, minlength=C) for a in range(m)], axis=1)
    H_tn = np.zeros((C, p, m))
    for j in range(p):
        for a in range(m):
            H_tn[:, j, a] = np.bincount(ch, Jt[:, j] * w * Jn[:, a], minlength=C)
    H_nn = np.zeros((C, m, m))
    for a in range(m):
        for b in range(a, m):
            H_nn[:, a, b] = H_nn[:, b, a] = np.bincount(ch, Jn[:, a] * w * Jn[:, b], minlength=C)
    return ChannelBlocks(total, g_t, g_n, H_tt, H_tn, H_nn)


def _profile(prob, theta, raw=None):
    """Closed-form nuisances at theta (separable models)."""
    if raw is None:
        with B.highest_precision():
            raw = prob.model(_cast(prob, theta), prob.aux)
        raw = {k: B.host64(raw[k]) for k in prob.observables}
    nu = prob.zeros()
    for a, kind in enumerate(prob.kinds):
        if kind == "log_gain":
            nu[:, a] = profile_log_gain(prob.channel, prob.charge.values, raw["charge"], prob.n_channels,
                                        prob.charge.family, prob.charge.sigma)
        else:
            nu[:, a] = profile_offset(prob.channel, prob.time.values, raw["time"], prob.n_channels, prob.time.sigma)
    return nu


# ------------------------------------------------------------------------------------------------ Schur algebra
def _schur(bl, free, lam=0.0):
    """Damped Schur pieces on the free theta: (S, reduced gradient, H_nn^-1, B = H_nn^-1 H_nt, H_tn_free)."""
    Htt = bl.H_tt[np.ix_(free, free)]
    Htn = bl.H_tn[:, free, :]
    Hnn = bl.H_nn
    if lam:
        dt = np.diag(Htt).copy()
        dt[~(dt > 0)] = 1.0
        Htt = Htt + lam * np.diag(dt)
        dn = np.einsum("caa->ca", Hnn).copy()
        dn[~(dn > 0)] = 1.0
        Hnn = Hnn + lam * dn[:, :, None] * np.eye(Hnn.shape[1])[None]
    Hinv = np.linalg.inv(Hnn)
    Bm = Hinv @ Htn.transpose(0, 2, 1)                          # [C, m, pf]
    S = Htt - np.einsum("cpa,caq->pq", Htn, Bm)
    gt = bl.g_t[free] - np.einsum("cpa,ca->p", Htn, np.einsum("cab,cb->ca", Hinv, bl.g_n))
    return S, gt, Hinv, Bm, Htn


def _step(bl, free, lam):
    S, gt, Hinv, Bm, Htn = _schur(bl, free, lam)
    dt = np.linalg.solve(S, -gt) if gt.size else gt
    dn = -np.einsum("cab,cb->ca", Hinv, bl.g_n + np.einsum("cpa,p->ca", Htn, dt))
    return dt, dn


def _decrement(bl, free):
    try:
        S, gt, Hinv, _, _ = _schur(bl, free)
        a = 0.5 * float(np.einsum("ca,cab,cb->", bl.g_n, Hinv, bl.g_n))
        b = 0.5 * float(gt @ np.linalg.solve(S, gt)) if gt.size else 0.0
    except np.linalg.LinAlgError:
        return np.inf
    v = a + b
    return v if np.isfinite(v) else np.inf


# ------------------------------------------------------------------------------------------------ fitting
@dataclass
class ChannelFitResult:
    theta: np.ndarray
    log_gain: np.ndarray          # [C] or None; centred when a gain gauge parameter is given
    offset: np.ndarray            # [C] or None; centred when an offset gauge parameter is given
    nll: float
    converged: bool
    iterations: int
    rejected_steps: int
    message: str
    fixed: np.ndarray
    decrement: float = np.nan
    cov_theta: np.ndarray = None   # nuisances marginalised, gauge applied
    theta_err: np.ndarray = None
    log_gain_err: np.ndarray = None
    offset_err: np.ndarray = None
    gauge_shift: dict = field(default_factory=dict)
    gauge_nll_change: float = 0.0
    history: list = field(default_factory=list)

    @property
    def nuisance(self):
        cols = [v for v in (self.log_gain, self.offset) if v is not None]
        return np.stack(cols, axis=1)


def _gauge_spec(prob, gauge, p):
    spec = {}
    for kind, j in (gauge or {}).items():
        if kind not in ("log_gain", "offset"):
            raise ValueError("gauge keys are 'log_gain' and 'offset'")
        if j is None:
            continue
        if kind not in prob.kinds:
            raise ValueError(f"gauge for {kind!r} but the problem has no such nuisance")
        if not 0 <= int(j) < p:
            raise ValueError(f"gauge parameter index {j} out of range")
        spec[kind] = int(j)
    if len(set(spec.values())) != len(spec):
        raise ValueError("the gain and offset gauges need different parameters")
    return spec


def _gauge_weights(prob, bl, gauge_weights):
    C = prob.n_channels
    if gauge_weights is None:
        return {k: np.full(C, 1.0 / C) for k in prob.kinds}
    if isinstance(gauge_weights, str):
        if gauge_weights != "information":
            raise ValueError("gauge_weights must be None, 'information' or an array")
        out = {}
        for a, k in enumerate(prob.kinds):
            w = bl.H_nn[:, a, a]
            out[k] = w / w.sum()
        return out
    w = np.asarray(gauge_weights, np.float64)
    if w.shape != (C,) or not np.all(np.isfinite(w) & (w >= 0)) or w.sum() <= 0:
        raise ValueError("gauge_weights must be [C], finite, non-negative, not all zero")
    return {k: w / w.sum() for k in prob.kinds}


def channel_crb(prob, theta, nu, *, fixed=None, gauge=None, gauge_weights=None, bl=None):
    """Cramér-Rao errors at ``(theta, nu)``: the ``theta`` covariance with the nuisances marginalised and the
    per-channel errors, with the gauge (as in :func:`fit_channels`) applied: centred channel terms and the gauge
    parameters' variance. Returns ``(cov_theta [p, p], nuisance_err [C, m])``; NaN when singular."""
    theta = np.asarray(theta, np.float64)
    p = theta.size
    spec = _gauge_spec(prob, gauge, p)
    held = _fixed_mask(fixed, p)
    for j in spec.values():
        held[j] = True
    free = ~held
    bl = blocks(prob, theta, nu) if bl is None else bl
    C, m = prob.n_channels, prob.m
    cov = np.zeros((p, p))
    try:
        S, _, Hinv, Bm, _ = _schur(bl, free)
        Sinv = np.linalg.inv(S) if S.size else S
        if S.size:
            np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return np.full((p, p), np.nan), np.full((C, m), np.nan)
    cov[np.ix_(free, free)] = Sinv
    # var(nu_ck) = Hinv_ckk + B_ck S^-1 B_ck^T
    var_n = np.einsum("caa->ca", Hinv) + np.einsum("cap,pq,caq->ca", Bm, Sinv, Bm)
    if spec:
        aw = _gauge_weights(prob, bl, gauge_weights)
        A = np.stack([aw[k] for k in prob.kinds], axis=1)                 # [C, m]
        U = np.einsum("ca,cap->ap", A, Bm)                                # [m, pf]
        V = np.einsum("ca,cb,cab->ab", A, A, Hinv) + U @ Sinv @ U.T       # cov of the weighted means [m, m]
        cov_aq = -U @ Sinv                                                # cov(weighted mean_a, theta_free)
        free_idx = np.flatnonzero(free)
        for kind, j in spec.items():
            a = prob.kinds.index(kind)
            cov_na = A[:, a] * Hinv[:, a, a] + np.einsum("cp,pq,q->c", Bm[:, a, :], Sinv, U[a])
            var_n[:, a] = var_n[:, a] - 2 * cov_na + V[a, a]
            cov[j, free_idx] = cov[free_idx, j] = cov_aq[a]
            for kind2, j2 in spec.items():
                cov[j, j2] = V[a, prob.kinds.index(kind2)]
    return cov, np.sqrt(np.clip(var_n, 0, None))


class _Profiled:
    """theta -> NLL with the nuisances profiled in closed form; gradient by the envelope theorem (for L-BFGS)."""

    def __init__(self, prob):
        self.prob = prob

    def value(self, theta):
        nu = _profile(self.prob, theta)
        if not np.all(np.isfinite(nu)):
            return np.inf
        return nll(self.prob, theta, nu)

    def value_and_grad(self, theta):
        nu = _profile(self.prob, theta)
        if not np.all(np.isfinite(nu)):
            return np.inf, np.full(np.size(theta), np.nan)
        bl = blocks(self.prob, theta, nu)
        return bl.nll, bl.g_t

    def terms(self, theta):
        """NLL, gradient and the information of the profiled objective: the Schur complement over all theta."""
        nu = _profile(self.prob, theta)
        bl = blocks(self.prob, theta, nu)
        S = _schur(bl, np.ones(np.size(theta), bool))[0]
        return GNTerms(bl.nll, bl.g_t, S)


def fit_channels(prob, theta0, *, nuisance0=None, method="gn", profile=None, gauge=None, gauge_weights=None,
                 fixed=None, scale=None, max_iter=200, tol=1e-10, damping=1e-3, damping_up=10.0, damping_down=0.3,
                 max_reject=40, gtol=1e-7, crb=True):
    """Fit ``theta`` and the per-channel nuisances of ``prob`` (module docstring).

    ``method``: ``"gn"`` (damped Gauss-Newton with Schur elimination) or ``"lbfgs"`` (on ``theta`` with the
    nuisances profiled in closed form; separable models only). ``profile``: ``"closed_form"`` (default for
    separable models: re-profile after every step) or ``"schur"`` (joint step, the default and only choice for
    coupled models). ``nuisance0``: start ``[C, m]`` (default: closed form, or zeros for coupled models).
    ``gauge``: ``{"log_gain": index or None, "offset": index or None}``; ``gauge_weights``: None (equal),
    ``"information"`` or ``[C]``. ``fixed`` and ``scale`` as in :mod:`hepsimlib.fit.optimize` (``scale`` applies to
    L-BFGS; the Marquardt damping of Gauss-Newton is already scale invariant). ``crb``: compute the errors.
    """
    theta = np.array(theta0, np.float64, copy=True)
    p = theta.size
    spec = _gauge_spec(prob, gauge, p)
    held = _fixed_mask(fixed, p)
    for j in spec.values():
        held[j] = True
    free = ~held
    if profile is None:
        profile = "schur" if prob.coupled else "closed_form"
    if profile not in ("closed_form", "schur"):
        raise ValueError("profile must be 'closed_form' or 'schur'")
    if prob.coupled and profile == "closed_form":
        raise ValueError("closed-form profiling needs a separable model (coupled=False)")
    history, rejected, it = [], 0, 0
    if method == "lbfgs":
        if prob.coupled:
            raise ValueError("method='lbfgs' profiles the nuisances in closed form: separable models only")
        res = lbfgs(_Profiled(prob), theta, fixed=held, scale=scale, max_iter=max_iter, gtol=gtol)
        theta, history, rejected, it = res.theta, res.history, res.rejected_steps, res.iterations
        converged, message = res.converged, res.message
        nu = _profile(prob, theta)
        bl = blocks(prob, theta, nu)
    elif method == "gn":
        if nuisance0 is not None:
            nu = np.asarray(nuisance0, np.float64).reshape(prob.n_channels, prob.m).copy()
        elif prob.coupled:
            nu = prob.zeros()
        else:
            nu = _profile(prob, theta)
        if not np.all(np.isfinite(nu)):
            raise ValueError("the start nuisances are not finite (a channel without signal?)")
        bl = blocks(prob, theta, nu)
        if not (np.isfinite(bl.nll) and np.all(np.isfinite(bl.g_t)) and np.all(np.isfinite(bl.H_tt))):
            raise ValueError("the start point has a non-finite NLL or gradient")
        lam = float(damping)
        consecutive = 0
        converged, message = False, "maximum iterations reached"
        for it in range(1, max_iter + 1):
            dec = _decrement(bl, free)
            if dec <= tol:
                converged, message = True, "Newton decrement below tol"
                it -= 1
                break
            accepted = False
            while consecutive < max_reject:
                try:
                    dt, dn = _step(bl, free, lam)
                    ok = np.all(np.isfinite(dt)) and np.all(np.isfinite(dn))
                except np.linalg.LinAlgError:
                    ok = False
                if ok:
                    trial = theta.copy()
                    trial[free] += dt
                    with _quiet():
                        tnu = _profile(prob, trial) if profile == "closed_form" else nu + dn
                    if np.all(np.isfinite(tnu)):
                        with _quiet():
                            ft = nll(prob, trial, tnu)
                        if np.isfinite(ft) and ft < bl.nll:
                            with _quiet():
                                tb = blocks(prob, trial, tnu)
                            if all(np.all(np.isfinite(x)) for x in tb):
                                accepted = True
                                break
                rejected += 1
                consecutive += 1
                lam *= damping_up
            if not accepted:
                message = f"no decrease after {max_reject} damped steps"
                converged = dec <= max(tol, 1e-6)
                break
            consecutive = 0
            history.append(dict(iteration=it, nll=tb.nll, decrease=bl.nll - tb.nll, damping=lam, decrement=dec))
            theta, nu, bl = trial, tnu, tb
            lam = max(lam * damping_down, 1e-12)
    else:
        raise ValueError("method must be 'gn' or 'lbfgs'")
    dec = _decrement(bl, free)
    # gauge: move the weighted mean of the channel terms into the declared global parameters
    shifts, change = {}, 0.0
    if spec:
        aw = _gauge_weights(prob, bl, gauge_weights)
        before = bl.nll
        for kind, j in spec.items():
            a = prob.kinds.index(kind)
            s = float(np.sum(aw[kind] * nu[:, a]))
            nu[:, a] -= s
            theta[j] += s
            shifts[kind] = s
        after = nll(prob, theta, nu)
        change = after - before
        if not abs(change) <= 1e-6 * max(1.0, abs(before)) * (1 if prob.dtype == np.float64 else 1e3):
            warnings.warn(f"the gauge move changed the NLL by {change:.3g}: the declared gauge parameter(s) "
                          f"{spec} do not enter like a common channel term", RuntimeWarning, stacklevel=2)
    out = ChannelFitResult(theta=theta, log_gain=None, offset=None, nll=float(bl.nll + change), converged=converged,
                           iterations=it, rejected_steps=rejected, message=message, fixed=held, decrement=dec,
                           gauge_shift=shifts, gauge_nll_change=float(change), history=history)
    for a, kind in enumerate(prob.kinds):
        setattr(out, kind, nu[:, a].copy())
    if crb:
        bl_end = blocks(prob, theta, nu) if spec else bl
        cov, nerr = channel_crb(prob, theta, nu, fixed=fixed, gauge=gauge, gauge_weights=gauge_weights,
                                bl=bl_end)
        out.cov_theta = cov
        out.theta_err = np.sqrt(np.clip(np.diag(cov), 0, None))
        for a, kind in enumerate(prob.kinds):
            setattr(out, kind + "_err", nerr[:, a])
    return out
