"""Finite-difference checks of gradients and Jacobians, for the tests of any differentiable path (docs/calibration.md).

A differentiable fast-simulation path (an expected response, a likelihood, a calibration objective) is only useful
if its gradient is right. The checks here compare the gradient a package computes (by autodiff or by hand) with
central finite differences along chosen directions, with a tolerance that accounts for the dtype:

    err = |ad - fd| / (atol + rtol * max(|ad|, |fd|))   must be <= 1 in every direction

with ``rtol`` the truncation tolerance (default 1e-6 in float64, 1e-2 in float32) and ``atol`` the rounding floor of
the difference quotient, ``10 eps max(1, |f(x)|) / h``. The step is ``h = rel_step * max(1, max|x|)`` with
``rel_step = eps**(1/3)`` (the optimum of central differences). Points where the function has a kink are not
differentiable, so a sweep (:func:`gradient_sweep`) shows where the comparison is meaningful.

NumPy only; with ``grad=None`` / ``jac=None`` the reference gradient is ``jax.grad`` / ``jax.jacfwd`` (needs JAX).
"""
from dataclasses import dataclass

import numpy as np

DEFAULT_RTOL = {"float32": 1e-2, "float64": 1e-6}


def _is_jax(x):
    return type(x).__module__.split(".")[0] in ("jax", "jaxlib")


def _dtype(x, dtype):
    if dtype is not None:
        d = np.dtype(dtype)
    else:
        xd = np.asarray(x).dtype
        d = xd if xd in (np.float32, np.float64) else np.dtype(np.float64)
    if d.name not in DEFAULT_RTOL:
        raise ValueError("dtype must be float32 or float64")
    return d


def _directions(directions, p, seed):
    if isinstance(directions, str):
        if directions != "axes":
            raise ValueError("directions must be 'axes', an integer count of random directions, or an array [k, p]")
        return np.eye(p)
    if np.isscalar(directions):
        u = np.random.default_rng(seed).normal(size=(int(directions), p))
    else:
        u = np.asarray(directions, np.float64).reshape(-1, p)
    n = np.linalg.norm(u, axis=1, keepdims=True)
    if np.any(n == 0):
        raise ValueError("a direction is zero")
    return u / n


def _input(x64, d, use_jax):
    if use_jax:
        import jax.numpy as jnp
        return jnp.asarray(x64.astype(d))
    return x64.astype(d)


def _caller(f, d, use_jax):
    if use_jax:
        import jax.numpy as jnp
        return lambda v: f(jnp.asarray(v, d))
    return lambda v: f(np.asarray(v, d))


@dataclass
class GradientReport:
    """Per-direction comparison of a gradient (or Jacobian-vector product) with central differences."""
    directions: np.ndarray       # [k, p] unit vectors
    ad: np.ndarray               # [k] or [k, m]: the checked derivative along each direction
    fd: np.ndarray               # same shape: central differences
    step: np.ndarray             # [k]
    rtol: float
    atol: np.ndarray             # [k]
    dtype: str
    value: float = np.nan

    @property
    def error(self):
        """Normalised error per direction (the maximum over outputs for a Jacobian); <= 1 passes."""
        den = self.atol.reshape((-1,) + (1,) * (self.ad.ndim - 1)) + self.rtol * np.maximum(np.abs(self.ad),
                                                                                           np.abs(self.fd))
        e = np.abs(self.ad - self.fd) / den
        return e.reshape(e.shape[0], -1).max(axis=1)

    @property
    def max_rel_error(self):
        """Largest |ad - fd| / max(|ad|, |fd|) over directions and outputs (0 where both vanish)."""
        den = np.maximum(np.abs(self.ad), np.abs(self.fd))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(den > 0, np.abs(self.ad - self.fd) / den, 0.0)
        return float(np.max(r))

    @property
    def ok(self):
        return bool(np.all(np.isfinite(self.ad)) and np.all(np.isfinite(self.fd)) and np.all(self.error <= 1))

    def __str__(self):
        lines = [f"gradient check ({self.dtype}, rtol {self.rtol:g}): {'ok' if self.ok else 'FAILED'}, "
                 f"max relative error {self.max_rel_error:.3g}",
                 f"{'dir':>4} {'ad':>14} {'fd':>14} {'step':>10} {'err/tol':>9}"]
        ad = self.ad.reshape(len(self.step), -1)
        fd = self.fd.reshape(len(self.step), -1)
        for i, e in enumerate(self.error):
            j = int(np.argmax(np.abs(ad[i] - fd[i])))
            lines.append(f"{i:>4} {ad[i, j]:>14.7g} {fd[i, j]:>14.7g} {self.step[i]:>10.3g} {e:>9.3g}"
                         + ("" if e <= 1 else "  <--"))
        return "\n".join(lines)

    def assert_ok(self):
        if not self.ok:
            raise AssertionError(str(self))
        return self


def _steps(x, u, d, rel_step):
    eps = np.finfo(d).eps
    rs = eps ** (1 / 3) if rel_step is None else float(rel_step)
    scale = np.array([max(1.0, float(np.max(np.abs(x[np.abs(ui) > 0])))) for ui in u])
    return rs * scale, eps


def check_gradient(f, x, grad=None, *, directions="axes", rel_step=None, rtol=None, atol=None, dtype=None, seed=0):
    """Compare the gradient of a scalar function with central differences along ``directions``.

    ``grad``: None (``jax.grad(f)``), a callable ``grad(x)``, or the gradient array. ``directions``: ``"axes"``
    (every coordinate), an integer (that many random unit directions, ``seed``) or an array ``[k, p]``. ``dtype``
    (default: x's, else float64) is the dtype ``f`` is evaluated in. Returns a :class:`GradientReport`.
    """
    d = _dtype(x, dtype)
    x64 = np.asarray(x, np.float64).ravel()
    use_jax = grad is None or _is_jax(x)
    call = _caller(f, d, use_jax)
    if grad is None:
        import jax
        g = np.asarray(jax.grad(f)(_input(x64, d, True)), np.float64)
    elif callable(grad):
        g = np.asarray(grad(_input(x64, d, use_jax)), np.float64)
    else:
        g = np.asarray(grad, np.float64)
    g = g.ravel()
    if g.size != x64.size:
        raise ValueError("the gradient must have one entry per parameter")
    u = _directions(directions, x64.size, seed)
    h, eps = _steps(x64, u, d, rel_step)
    f0 = float(call(x64.astype(d)))
    ad = u @ g
    fd = np.empty(len(u))
    for i, (ui, hi) in enumerate(zip(u, h)):
        up = (x64 + hi * ui).astype(d)
        dn = (x64 - hi * ui).astype(d)
        hh = float(np.linalg.norm(up.astype(np.float64) - dn.astype(np.float64))) / 2   # the step actually taken
        fd[i] = (float(call(up)) - float(call(dn))) / (2 * hh)
    floor = 10 * eps * max(1.0, abs(f0)) / h if atol is None else np.full(len(u), float(atol))
    return GradientReport(u, ad, fd, h, DEFAULT_RTOL[d.name] if rtol is None else float(rtol), floor, d.name, f0)


def assert_gradient(f, x, grad=None, **kw):
    """:func:`check_gradient`, raising an AssertionError with the table when it fails."""
    return check_gradient(f, x, grad, **kw).assert_ok()


def check_jacobian(F, x, jac=None, *, directions="axes", rel_step=None, rtol=None, atol=None, dtype=None, seed=0):
    """Compare a Jacobian ``[m, p]`` (or ``jac=None``: ``jax.jacfwd(F)``) with central differences of the vector
    function ``F`` along ``directions``; the error of a direction is the largest over the outputs."""
    d = _dtype(x, dtype)
    x64 = np.asarray(x, np.float64).ravel()
    use_jax = jac is None or _is_jax(x)
    call = _caller(F, d, use_jax)
    xin = _input(x64, d, use_jax)
    if jac is None:
        import jax
        J = np.asarray(jax.jacfwd(F)(xin), np.float64)
    elif callable(jac):
        J = np.asarray(jac(xin), np.float64)
    else:
        J = np.asarray(jac, np.float64)
    J = J.reshape(-1, x64.size)
    u = _directions(directions, x64.size, seed)
    h, eps = _steps(x64, u, d, rel_step)
    F0 = np.asarray(call(x64.astype(d)), np.float64).ravel()
    ad = u @ J.T                                    # [k, m]
    fd = np.empty_like(ad)
    for i, (ui, hi) in enumerate(zip(u, h)):
        up = (x64 + hi * ui).astype(d)
        dn = (x64 - hi * ui).astype(d)
        hh = float(np.linalg.norm(up.astype(np.float64) - dn.astype(np.float64))) / 2
        fd[i] = (np.asarray(call(up), np.float64).ravel() - np.asarray(call(dn), np.float64).ravel()) / (2 * hh)
    floor = (10 * eps * max(1.0, float(np.max(np.abs(F0), initial=0))) / h if atol is None
             else np.full(len(u), float(atol)))
    return GradientReport(u, ad, fd, h, DEFAULT_RTOL[d.name] if rtol is None else float(rtol), floor, d.name)


def assert_jacobian(F, x, jac=None, **kw):
    return check_jacobian(F, x, jac, **kw).assert_ok()


def gradient_sweep(f, x, direction, span, n=21, grad=None, *, dtype=None, rel_step=None):
    """Values and directional derivatives (autodiff or ``grad`` vs central differences) of a scalar function at
    ``n`` points ``x + t u``, ``t`` in ``[-span, span]``: where they disagree, look for kinks, plateaus or bins.
    ``grad`` is None (``jax.grad``) or a callable ``grad(x)``.

    Returns a dict of arrays: ``t``, ``value``, ``ad``, ``fd``, ``error`` (normalised as in :class:`GradientReport`).
    """
    u = _directions(np.asarray(direction, np.float64)[None], np.size(x), 0)[0]
    x64 = np.asarray(x, np.float64).ravel()
    ts = np.linspace(-span, span, n)
    out = dict(t=ts, value=np.empty(n), ad=np.empty(n), fd=np.empty(n), error=np.empty(n))
    for i, t in enumerate(ts):
        r = check_gradient(f, x64 + t * u, grad, directions=u[None], dtype=dtype, rel_step=rel_step)
        out["value"][i], out["ad"][i], out["fd"][i], out["error"][i] = r.value, r.ad[0], r.fd[0], r.error[0]
    return out
