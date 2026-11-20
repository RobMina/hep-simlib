"""Damped Gauss-Newton (Fisher scoring) and L-BFGS drivers with the same hygiene (docs/calibration.md, 2.1).

Both drivers minimise a negative log likelihood ``nll(theta)`` over a small parameter vector (float64, host) and
share:

- ``fixed``: parameters held at their start values, bit for bit (a boolean array or a list of indices);
- ``scale``: per-parameter preconditioning; the drivers work in ``u = (theta - theta0) / scale`` on the free
  parameters. ``None`` is unit scale, ``"auto"`` is ``diag(info(theta0))**-0.5`` (needs an objective with
  ``terms``), an array is used as given;
- rollback: a trial point whose value, gradient or information is not finite is never accepted; the driver keeps
  the last good point and shortens the step (L-BFGS) or raises the damping (Gauss-Newton);
- a :class:`FitResult` with the history of accepted steps.

An objective is any object with ``value(theta) -> float`` and ``value_and_grad(theta) -> (float, grad[p])`` and, for
Gauss-Newton, ``terms(theta) -> GNTerms(nll, grad[p], info[p, p])``, where ``info`` is the Fisher information (the
Gauss-Newton matrix). :mod:`hepsimlib.fit.objectives` builds them from residuals, expected counts or a scalar
function; a plain callable is accepted too (``terms`` for Gauss-Newton, ``value_and_grad`` for L-BFGS).
"""
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


class GNTerms(NamedTuple):
    """NLL, its gradient and the Fisher information (Gauss-Newton matrix) at one point, float64."""
    nll: float
    grad: np.ndarray
    info: np.ndarray


@dataclass
class FitResult:
    theta: np.ndarray
    nll: float
    grad: np.ndarray
    info: object                       # Fisher information at the optimum (None when the objective has none)
    converged: bool
    iterations: int
    rejected_steps: int
    message: str
    fixed: np.ndarray                  # boolean, True where held
    scale: np.ndarray
    decrement: float = np.nan          # Newton decrement 0.5 g^T H^-1 g at the end (Gauss-Newton)
    history: list = field(default_factory=list)


class _Wrapped:
    """Adapter for plain callables: terms(theta) -> GNTerms, or value_and_grad(theta) -> (value, grad)."""

    def __init__(self, terms=None, value_and_grad=None):
        self._terms, self._vg = terms, value_and_grad

    def terms(self, theta):
        return self._terms(theta)

    def value_and_grad(self, theta):
        if self._vg is not None:
            return self._vg(theta)
        t = self._terms(theta)
        return t.nll, t.grad

    def value(self, theta):
        return self.value_and_grad(theta)[0]


def _objective(obj, need_terms):
    if hasattr(obj, "value_and_grad"):
        if need_terms and not hasattr(obj, "terms"):
            raise TypeError("Gauss-Newton needs an objective with terms(theta) -> GNTerms")
        return obj
    if not callable(obj):
        raise TypeError("the objective must be callable or have value_and_grad / terms methods")
    return _Wrapped(terms=obj) if need_terms else _Wrapped(value_and_grad=obj)


def _fixed_mask(fixed, p):
    m = np.zeros(p, bool)
    if fixed is None:
        return m
    f = np.asarray(fixed)
    if f.dtype == bool:
        if f.shape != (p,):
            raise ValueError(f"fixed mask must have shape ({p},)")
        return f.copy()
    m[f.astype(int)] = True
    return m


def _scale(scale, p, info=None):
    if scale is None:
        return np.ones(p)
    if isinstance(scale, str):
        if scale != "auto":
            raise ValueError("scale must be None, 'auto' or an array")
        if info is None:
            raise ValueError("scale='auto' needs the information matrix (an objective with terms)")
        d = np.diag(np.asarray(info, np.float64))
        good = np.isfinite(d) & (d > 0)
        return np.where(good, 1 / np.sqrt(np.where(good, d, 1.0)), 1.0)
    s = np.broadcast_to(np.asarray(scale, np.float64), (p,)).copy()
    if not np.all(np.isfinite(s) & (s > 0)):
        raise ValueError("scale must be finite and positive")
    return s


def _quiet():
    """Trial points may overflow; the drivers reject them, so NumPy's floating-point warnings are not raised there."""
    return np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore")


def _finite(*xs):
    return all(np.all(np.isfinite(x)) for x in xs)


def _decrement(g, H):
    """0.5 g^T H^-1 g on the free block (least squares if H is singular)."""
    if g.size == 0:
        return 0.0
    try:
        x = np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(H, g, rcond=None)[0]
    v = 0.5 * float(g @ x)
    return v if np.isfinite(v) else np.inf


def gauss_newton(objective, theta0, *, fixed=None, scale=None, max_iter=100, tol=1e-10, damping=1e-3,
                 damping_up=10.0, damping_down=0.3, max_reject=40, callback=None):
    """Levenberg-Marquardt-damped Gauss-Newton (Fisher scoring) on ``objective.terms``.

    Each iteration solves ``(H + lambda diag(H)) d = -g`` on the free parameters, in scaled coordinates, and accepts
    the step when the NLL at the trial point is finite and lower, and the gradient and information there are finite.
    Otherwise the step is rolled back and ``lambda`` multiplied by ``damping_up``. Converged when the Newton
    decrement ``0.5 g^T H^-1 g`` (the predicted NLL decrease) is at most ``tol``.
    """
    obj = _objective(objective, need_terms=True)
    theta = np.array(theta0, dtype=np.float64, copy=True)
    p = theta.size
    held = _fixed_mask(fixed, p)
    free = ~held
    t = obj.terms(theta)
    if not _finite(t.nll, t.grad, t.info):
        raise ValueError("the start point has a non-finite NLL, gradient or information")
    s = _scale(scale, p, t.info)
    sf = s[free]
    lam = float(damping)
    history, rejected, consecutive = [], 0, 0
    message, converged, it = "maximum iterations reached", False, 0
    for it in range(1, max_iter + 1):
        g = t.grad[free] * sf
        H = t.info[np.ix_(free, free)] * np.outer(sf, sf)
        dec = _decrement(g, H)
        if dec <= tol:
            converged, message = True, "Newton decrement below tol"
            it -= 1
            break
        dH = np.diag(H).copy()
        dH[~(np.isfinite(dH) & (dH > 0))] = 1.0
        accepted = False
        while consecutive < max_reject:
            try:
                d = np.linalg.solve(H + lam * np.diag(dH), -g)
            except np.linalg.LinAlgError:
                d = None
            if d is not None and _finite(d):
                trial = theta.copy()
                trial[free] = theta[free] + sf * d
                with _quiet():
                    ft = obj.value(trial)
                if np.isfinite(ft) and ft < t.nll:
                    with _quiet():
                        tt = obj.terms(trial)
                    if _finite(tt.nll, tt.grad, tt.info):
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
        history.append(dict(iteration=it, nll=float(tt.nll), decrease=float(t.nll - tt.nll), damping=lam,
                            decrement=dec))
        theta, t = trial, tt
        lam = max(lam * damping_down, 1e-12)
        if callback is not None:
            callback(theta, t)
    g = t.grad[free] * sf
    H = t.info[np.ix_(free, free)] * np.outer(sf, sf)
    return FitResult(theta=theta, nll=float(t.nll), grad=np.asarray(t.grad), info=np.asarray(t.info),
                     converged=converged, iterations=it, rejected_steps=rejected, message=message, fixed=held,
                     scale=s, decrement=_decrement(g, H), history=history)


def lbfgs(objective, theta0, *, fixed=None, scale=None, max_iter=500, history=10, gtol=1e-6, ftol=1e-14,
          max_linesearch=40, c1=1e-4, callback=None):
    """L-BFGS (two-loop recursion, backtracking Armijo line search) on ``objective.value_and_grad``.

    A trial point with a non-finite value or gradient is rolled back and the step halved. A curvature pair with
    ``s.y <= 0`` is skipped. Converged when the largest scaled gradient component is at most ``gtol`` or an accepted
    step lowers the NLL by at most ``ftol * max(1, |nll|)``. The information at the optimum is filled when the
    objective has ``terms``.
    """
    obj = _objective(objective, need_terms=False)
    theta = np.array(theta0, dtype=np.float64, copy=True)
    p = theta.size
    held = _fixed_mask(fixed, p)
    free = ~held
    info0 = obj.terms(theta).info if (isinstance(scale, str) and hasattr(obj, "terms")) else None
    s = _scale(scale, p, info0)
    sf = s[free]
    f, grad = obj.value_and_grad(theta)
    f, grad = float(f), np.asarray(grad, np.float64)
    if not _finite(f, grad):
        raise ValueError("the start point has a non-finite NLL or gradient")
    S, Y = [], []
    hist, rejected = [], 0
    message, converged, it = "maximum iterations reached", False, 0
    for it in range(1, max_iter + 1):
        g = grad[free] * sf
        if np.max(np.abs(g), initial=0.0) <= gtol:
            converged, message = True, "gradient below gtol"
            it -= 1
            break
        q = g.copy()
        alphas = []
        for sv, yv in zip(reversed(S), reversed(Y)):
            a = (sv @ q) / (yv @ sv)
            alphas.append(a)
            q -= a * yv
        gamma = (S[-1] @ Y[-1]) / (Y[-1] @ Y[-1]) if S else min(1.0, 1.0 / max(np.linalg.norm(g), 1e-300))
        r = gamma * q
        for (sv, yv), a in zip(zip(S, Y), reversed(alphas)):
            b = (yv @ r) / (yv @ sv)
            r += sv * (a - b)
        d = -r
        slope = g @ d
        if not (slope < 0) or not _finite(d):
            S, Y = [], []          # not a descent direction: restart from steepest descent
            d = -g * min(1.0, 1.0 / max(np.linalg.norm(g), 1e-300))
            slope = g @ d
        step, ok = 1.0, False
        for _ in range(max_linesearch):
            trial = theta.copy()
            trial[free] = theta[free] + sf * (step * d)
            with _quiet():
                ft, gt = obj.value_and_grad(trial)
            ft, gt = float(ft), np.asarray(gt, np.float64)
            if _finite(ft, gt) and ft <= f + c1 * step * slope:
                ok = True
                break
            rejected += 1
            step *= 0.5
        if not ok:
            message = "line search failed"
            converged = np.max(np.abs(g), initial=0.0) <= max(gtol, 1e-4)
            break
        sv = step * d
        yv = gt[free] * sf - g
        if sv @ yv > 1e-12 * np.linalg.norm(sv) * np.linalg.norm(yv):
            S.append(sv)
            Y.append(yv)
            if len(S) > history:
                S.pop(0)
                Y.pop(0)
        decrease = f - ft
        hist.append(dict(iteration=it, nll=ft, decrease=decrease, step=step))
        theta, f, grad = trial, ft, gt
        if callback is not None:
            callback(theta, f)
        if decrease <= ftol * max(1.0, abs(f)):
            converged, message = True, "relative decrease below ftol"
            break
    info = obj.terms(theta).info if hasattr(obj, "terms") else None
    return FitResult(theta=theta, nll=f, grad=grad, info=None if info is None else np.asarray(info),
                     converged=converged, iterations=it, rejected_steps=rejected, message=message, fixed=held,
                     scale=s, history=hist)
