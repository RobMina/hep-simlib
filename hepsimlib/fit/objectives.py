"""Objectives for the drivers in :mod:`hepsimlib.fit.optimize`: residuals, Gaussian and Poisson expectations, scalars.

Each builder returns an object with ``value``, ``value_and_grad`` and (except :func:`scalar`) ``terms``. The model
function may be NumPy or JAX: it receives ``theta`` cast to ``dtype`` (float32 or float64) and returns an array; the
NLL, its gradient and the information are accumulated in float64 on the host, with the Jacobian products done in the
model's backend at the highest precision (:func:`hepsimlib.fit._backend.normal_equations`).

``jac`` selects the Jacobian: ``"fwd"`` (``jax.jacfwd``, the default when JAX is installed), ``"rev"``, ``"fd"``
(central differences, NumPy; the default without JAX) or a callable ``jac(theta) -> [n, p]``.
"""
import numpy as np

from . import _backend as B
from .optimize import GNTerms


def _default_jac(jac):
    if jac is None:
        return "fwd" if B.have_jax() else "fd"
    return jac


class _Model:
    def __init__(self, fn, jac, dtype):
        self.fn = fn
        self.jac = _default_jac(jac)
        self.dtype = B.check_dtype(dtype, self.jac if isinstance(self.jac, str) else None)

    def _cast(self, theta):
        return np.asarray(theta, np.float64).astype(self.dtype)

    def eval(self, theta):
        with B.highest_precision():
            return self.fn(self._cast(theta))

    def jacobian(self, theta):
        if callable(self.jac):
            return self.jac(self._cast(theta))
        return B.jacobian(self.fn, theta, self.jac, dtype=self.dtype)


class LeastSquares:
    """``nll = 0.5 * sum r(theta)**2`` for whitened residuals ``r`` (already divided by their errors)."""

    def __init__(self, residual_fn, *, jac=None, dtype=None):
        self.model = _Model(residual_fn, jac, dtype)

    def value(self, theta):
        r = B.host64(self.model.eval(theta))
        return 0.5 * float(r @ r)

    def terms(self, theta):
        r = self.model.eval(theta)
        J = self.model.jacobian(theta)
        H, g = B.normal_equations(J, r)
        r64 = B.host64(r)
        return GNTerms(0.5 * float(r64 @ r64), g, H)

    def value_and_grad(self, theta):
        t = self.terms(theta)
        return t.nll, t.grad


def least_squares(residual_fn, *, jac=None, dtype=None):
    return LeastSquares(residual_fn, jac=jac, dtype=dtype)


def gaussian(expected_fn, observed, sigma, *, jac=None, dtype=None):
    """Gaussian measurements ``observed ~ Normal(expected(theta), sigma)``: least squares on ``(mu - y) / sigma``."""
    y = np.asarray(observed, np.float64)
    s = np.broadcast_to(np.asarray(sigma, np.float64), y.shape)
    if not np.all(np.isfinite(s) & (s > 0)):
        raise ValueError("sigma must be finite and positive")

    def residual(theta):
        mu = expected_fn(theta)
        xp = __import__("jax.numpy", fromlist=["asarray"]) if B.is_jax(mu) else np
        mu = xp.asarray(mu)
        return (mu - xp.asarray(y, mu.dtype)) / xp.asarray(s, mu.dtype)

    return LeastSquares(residual, jac=jac, dtype=dtype)


class Poisson:
    """Counts ``observed ~ Poisson(expected(theta))``; ``nll = sum(mu - k log mu)`` (the k! term dropped).

    The information is the Fisher matrix ``J^T diag(1/mu) J``, so Gauss-Newton on this objective is Fisher scoring.
    A non-positive expectation where a count is observed gives an infinite NLL (rolled back by the drivers); rows
    with ``mu = 0`` and ``k = 0`` contribute nothing.
    """

    def __init__(self, expected_fn, observed, *, jac=None, dtype=None):
        self.model = _Model(expected_fn, jac, dtype)
        self.k = np.asarray(observed, np.float64)
        if not np.all(np.isfinite(self.k) & (self.k >= 0)):
            raise ValueError("observed counts must be finite and non-negative")

    def _nll(self, mu):
        k = self.k
        with np.errstate(divide="ignore", invalid="ignore"):
            bad = np.any((mu <= 0) & (k > 0)) or np.any(mu < 0) or not np.all(np.isfinite(mu))
            if bad:
                return np.inf
            return float(np.sum(mu) - np.sum(np.where(k > 0, k * np.log(np.where(k > 0, mu, 1.0)), 0.0)))

    def value(self, theta):
        return self._nll(B.host64(self.model.eval(theta)))

    def terms(self, theta):
        mu_b = self.model.eval(theta)
        mu = B.host64(mu_b)
        nll = self._nll(mu)
        J = self.model.jacobian(theta)
        pos = mu > 0
        w = np.where(pos, 1 / np.where(pos, mu, 1.0), 0.0)
        r = np.where(pos, 1 - self.k * w, 1.0)        # d nll / d mu
        if B.is_jax(J):
            import jax.numpy as jnp
            H, _ = B.normal_equations(J, None, jnp.asarray(w, J.dtype))
            _, g = B.normal_equations(J, jnp.asarray(r, J.dtype))
        else:
            H, _ = B.normal_equations(J, None, w)
            _, g = B.normal_equations(J, r)
        return GNTerms(nll, g, H)

    def value_and_grad(self, theta):
        t = self.terms(theta)
        return t.nll, t.grad


def poisson(expected_fn, observed, *, jac=None, dtype=None):
    return Poisson(expected_fn, observed, jac=jac, dtype=dtype)


class Scalar:
    """A scalar NLL with its gradient: ``grad`` = ``"rev"`` (``jax.value_and_grad``), ``"fd"`` or a callable
    ``grad(theta)``. No information matrix: for L-BFGS (and ``fit.covariance`` of a separately computed Hessian)."""

    def __init__(self, fn, *, grad=None, dtype=None):
        self.fn = fn
        self.grad = ("rev" if B.have_jax() else "fd") if grad is None else grad
        self.dtype = B.check_dtype(dtype, "rev" if self.grad == "rev" else None)
        if self.grad == "rev":
            import jax
            self._vg = jax.value_and_grad(fn)

    def _cast(self, theta):
        return np.asarray(theta, np.float64).astype(self.dtype)

    def value(self, theta):
        with B.highest_precision():
            return float(self.fn(self._cast(theta)))

    def value_and_grad(self, theta):
        if self.grad == "rev":
            import jax.numpy as jnp
            with B.highest_precision():
                v, g = self._vg(jnp.asarray(self._cast(theta)))
            return float(v), B.host64(g)
        if callable(self.grad):
            return self.value(theta), B.host64(self.grad(self._cast(theta)))
        if self.grad != "fd":
            raise ValueError("grad must be 'rev', 'fd' or a callable")
        theta = np.asarray(theta, np.float64)
        h = B.fd_step(theta, self.dtype)
        g = np.empty_like(theta)
        for j in range(theta.size):
            e = np.zeros_like(theta)
            e[j] = h[j]
            g[j] = (self.value(theta + e) - self.value(theta - e)) / (2 * h[j])
        return self.value(theta), g


def scalar(fn, *, grad=None, dtype=None):
    return Scalar(fn, grad=grad, dtype=dtype)
