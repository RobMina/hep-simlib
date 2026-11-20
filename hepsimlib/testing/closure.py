"""Closure tests of a fit: multi-seed pulls, model-mismatch closures and the zero expected score (docs/calibration.md).

A calibration is closed when, on data simulated with known parameters, its estimates are unbiased and its errors
describe their spread. Three checks, each usable from any package's tests:

- :func:`closure`: simulate with the fitted model at ``truth`` for many seeds, fit each, and collect pulls
  ``(estimate - truth) / error``. Closed: pull mean ``0`` and pull width ``1`` within their statistical errors.
- :func:`mismatch_closure`: the same with a generator that is *not* the fitted model (Poisson photo-electrons on a
  log-normal yield, a wrong width, an extra smearing). It measures the bias that the model's simplifications cost.
  It does not assert by default: the result is a number to report, not a pass/fail.
- :func:`expected_score_zero` / :func:`mean_score_zero`: the score (gradient of the NLL) at the truth is zero on
  expected (Asimov) data, and zero on average over sampled data. A non-zero expected score means the likelihood is
  not the one the data follow, whatever the optimizer does.

Simulations take a ``numpy.random.Generator`` built from the seed, so every toy is reproducible on its own.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClosureResult:
    truth: np.ndarray            # [k]
    estimates: np.ndarray        # [S, k]
    errors: np.ndarray           # [S, k]
    seeds: np.ndarray            # [S] seeds of the fits that succeeded
    names: tuple = ()
    failures: dict = field(default_factory=dict)   # seed -> repr of the exception
    label: str = "closure"
    mismatch: bool = False

    @property
    def n(self):
        return self.estimates.shape[0]

    @property
    def pulls(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.estimates - self.truth) / self.errors

    @property
    def bias(self):
        return self.estimates.mean(axis=0) - self.truth

    @property
    def bias_err(self):
        return self.estimates.std(axis=0, ddof=1) / np.sqrt(self.n)

    @property
    def spread(self):
        return self.estimates.std(axis=0, ddof=1)

    @property
    def mean_error(self):
        return self.errors.mean(axis=0)

    @property
    def pull_mean(self):
        return np.nanmean(self.pulls, axis=0)

    @property
    def pull_std(self):
        return np.nanstd(self.pulls, axis=0, ddof=1)

    @property
    def pull_std_err(self):
        """Standard error of the pull width for normal pulls: 1 / sqrt(2 (S - 1))."""
        return 1 / np.sqrt(2 * (self.n - 1))

    def coverage(self, k=1.0):
        """Fraction of toys with |pull| <= k (0.683 for k = 1 when closed)."""
        return np.mean(np.abs(self.pulls) <= k, axis=0)

    def summary(self):
        names = self.names or tuple(f"p{i}" for i in range(self.truth.size))
        head = (f"{self.label}: {self.n} toys" + (f", {len(self.failures)} failed" if self.failures else "")
                + (" (model mismatch)" if self.mismatch else ""))
        lines = [head, f"{'parameter':>16} {'truth':>11} {'bias':>11} {'+-':>9} {'spread':>9} {'<err>':>9} "
                       f"{'pull mean':>9} {'pull std':>9} {'cov68':>6}"]
        cov = self.coverage()
        for i, nm in enumerate(names):
            lines.append(f"{nm:>16} {self.truth[i]:>11.5g} {self.bias[i]:>11.4g} {self.bias_err[i]:>9.3g} "
                         f"{self.spread[i]:>9.3g} {self.mean_error[i]:>9.3g} {self.pull_mean[i]:>9.3f} "
                         f"{self.pull_std[i]:>9.3f} {cov[i]:>6.3f}")
        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def assert_unbiased(self, k=3.0, width_tol=None, max_failures=0):
        """Pull means within ``k`` standard errors of 0 and pull widths within ``width_tol`` (default ``k`` standard
        errors) of 1; at most ``max_failures`` failed fits. Raises AssertionError with the summary."""
        problems = []
        if len(self.failures) > max_failures:
            problems.append(f"{len(self.failures)} fits failed: {dict(list(self.failures.items())[:3])}")
        se_mean = self.pull_std / np.sqrt(self.n)
        bad_mean = np.abs(self.pull_mean) > k * se_mean
        wt = k * self.pull_std_err if width_tol is None else width_tol
        bad_width = np.abs(self.pull_std - 1) > wt
        if bad_mean.any():
            problems.append(f"pull means off zero for parameters {np.flatnonzero(bad_mean).tolist()}")
        if bad_width.any():
            problems.append(f"pull widths off one (tolerance {wt:.3g}) for parameters "
                            f"{np.flatnonzero(bad_width).tolist()}")
        if problems:
            raise AssertionError("; ".join(problems) + "\n" + self.summary())
        return self


def closure(simulate, fit, truth, seeds, *, names=(), label="closure", catch=(Exception,), _mismatch=False):
    """Fit toys simulated at ``truth``: ``simulate(truth, rng) -> data`` (``rng`` = ``default_rng(seed)``),
    ``fit(data) -> (estimates [k], errors [k])``. A fit that raises one of ``catch`` is recorded in ``failures``.
    """
    truth = np.asarray(truth, np.float64)
    est, err, ok, failures = [], [], [], {}
    for seed in seeds:
        data = simulate(truth, np.random.default_rng(seed))
        try:
            e, s = fit(data)
        except catch as exc:     # noqa: PERF203 - one try per toy is the point
            failures[int(seed)] = repr(exc)
            continue
        est.append(np.asarray(e, np.float64).ravel())
        err.append(np.asarray(s, np.float64).ravel())
        ok.append(int(seed))
    if not est:
        raise RuntimeError(f"every fit failed: {dict(list(failures.items())[:3])}")
    return ClosureResult(truth.ravel(), np.array(est), np.array(err), np.array(ok), tuple(names), failures, label,
                         _mismatch)


def mismatch_closure(simulate_other, fit, truth, seeds, *, names=(), label="model-mismatch closure", catch=(Exception,)):
    """:func:`closure` with a generator that differs from the fitted model; the result carries ``mismatch=True``.
    Report its bias and pulls; assert only a bound you have decided on beforehand."""
    return closure(simulate_other, fit, truth, seeds, names=names, label=label, catch=catch, _mismatch=True)


@dataclass
class ScoreReport:
    grad: np.ndarray
    decrement: float          # 0.5 g^T info^-1 g (NaN without info)
    tol: float
    ok: bool

    def __str__(self):
        return (f"expected score at truth: {'ok' if self.ok else 'FAILED'}; max |grad| {np.max(np.abs(self.grad)):.3g}"
                f", decrement {self.decrement:.3g} (tol {self.tol:g})\ngrad = {np.array2string(self.grad, precision=4)}")

    def assert_ok(self):
        if not self.ok:
            raise AssertionError(str(self))
        return self


def _grad(nll, theta, grad):
    if grad is None:
        import jax
        import jax.numpy as jnp
        return np.asarray(jax.grad(nll)(jnp.asarray(theta)), np.float64)
    if callable(grad):
        return np.asarray(grad(theta), np.float64)
    raise TypeError("grad must be None (jax.grad) or a callable")


def expected_score_zero(nll, truth, *, grad=None, info=None, tol=None):
    """The gradient of ``nll`` (the NLL of expected, Asimov, data) at ``truth`` is zero.

    With ``info`` (the Fisher information at truth), the test is scale free: the Newton decrement
    ``0.5 g^T info^-1 g`` (the NLL a fit would gain by moving) must be at most ``tol`` (default 1e-8). Without it,
    ``max |g| <= tol`` (default 1e-8). ``grad``: None (``jax.grad``) or a callable."""
    theta = np.asarray(truth, np.float64)
    g = _grad(nll, theta, grad).ravel()
    tol = 1e-8 if tol is None else float(tol)
    if info is not None:
        H = np.asarray(info, np.float64)
        dec = 0.5 * float(g @ np.linalg.lstsq(H, g, rcond=None)[0])
        return ScoreReport(g, dec, tol, bool(np.isfinite(dec) and dec <= tol))
    return ScoreReport(g, np.nan, tol, bool(np.all(np.isfinite(g)) and np.max(np.abs(g)) <= tol))


@dataclass
class MeanScoreReport:
    mean: np.ndarray
    err: np.ndarray
    z: np.ndarray
    k: float
    ok: bool

    def __str__(self):
        return (f"mean score at truth: {'ok' if self.ok else 'FAILED'} (|z| <= {self.k:g})\n"
                f"mean = {np.array2string(self.mean, precision=4)}\n z   = {np.array2string(self.z, precision=2)}")

    def assert_ok(self):
        if not self.ok:
            raise AssertionError(str(self))
        return self


def mean_score_zero(score, simulate, truth, seeds, *, k=4.0):
    """The score at the truth averages to zero over sampled data: ``score(data, truth) -> grad [p]`` for data from
    ``simulate(truth, default_rng(seed))``; passes when every component of the mean is within ``k`` standard
    errors of zero."""
    theta = np.asarray(truth, np.float64)
    g = np.array([np.asarray(score(simulate(theta, np.random.default_rng(s)), theta), np.float64).ravel()
                  for s in seeds])
    mean = g.mean(axis=0)
    err = g.std(axis=0, ddof=1) / np.sqrt(len(g))
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.where(err > 0, mean / err, np.where(mean == 0, 0.0, np.inf))
    return MeanScoreReport(mean, err, z, k, bool(np.all(np.abs(z) <= k)))
