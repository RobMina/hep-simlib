"""hepsimlib.fit: drivers, hygiene, channel profiling, Schur elimination, gauge, Cramér-Rao errors, Jacobians.

Synthetic toys only. The channel toy is a light-guide readout: rows at a distance x from the sensor of channel c,
expected charge Y exp(-x / lambda) with a per-channel gain, expected time t0 + s x with a per-channel offset.
NumPy (finite-difference Jacobians) everywhere; the JAX tests (jacfwd, float32) skip without JAX.
"""
import numpy as np
import pytest

from hepsimlib import fit
from hepsimlib.fit import _backend as B
from hepsimlib.fit import channels as CH


def xp_of(x):
    if B.is_jax(x):
        import jax.numpy as jnp
        return jnp
    return np


# ------------------------------------------------------------------------------------------------ generic toys
def decay_data(seed=1, n=200):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 4, n)
    truth = np.array([np.log(30.0), 0.7, 2.0])
    y = np.exp(truth[0] - truth[1] * x) + truth[2] + rng.normal(0, 0.5, n)
    return x, y, truth


def decay_residual(x, y, sigma=0.5):
    def r(theta):
        xp = xp_of(theta)
        xx = xp.asarray(x, theta.dtype)
        return (xp.exp(theta[0] - theta[1] * xx) + theta[2] - xp.asarray(y, theta.dtype)) / sigma
    return r


def counts_data(seed=2, n=60):
    rng = np.random.default_rng(seed)
    x = np.linspace(-3, 3, n)
    truth = np.array([np.log(200.0), 0.3, 0.8, np.log(5.0)])

    def expected(theta):
        xp = xp_of(theta)
        xx = xp.asarray(x, theta.dtype)
        return xp.exp(theta[0] - 0.5 * ((xx - theta[1]) / theta[2]) ** 2) + xp.exp(theta[3])
    return expected, rng.poisson(expected(truth)), truth


def rosenbrock(theta):
    xp = xp_of(theta)
    return xp.stack([10 * (theta[1] - theta[0] ** 2), 1 - theta[0]])


def test_gn_and_lbfgs_agree_on_least_squares():
    x, y, truth = decay_data()
    obj = fit.least_squares(decay_residual(x, y), jac="fd")
    a = fit.gauss_newton(obj, [3.0, 0.5, 1.0])
    b = fit.lbfgs(obj, [3.0, 0.5, 1.0], scale="auto", gtol=1e-9, max_iter=2000)
    assert a.converged and b.converged
    np.testing.assert_allclose(a.theta, b.theta, rtol=0, atol=1e-5)
    assert abs(a.nll - b.nll) < 1e-8
    err = fit.errors(fit.covariance(a.info))
    assert np.all(np.abs(a.theta - truth) < 5 * err)


def test_gn_and_lbfgs_agree_on_poisson():
    expected, k, truth = counts_data()
    obj = fit.poisson(expected, k, jac="fd")
    a = fit.gauss_newton(obj, [5.0, 0.0, 1.0, 1.0])
    b = fit.lbfgs(obj, [5.0, 0.0, 1.0, 1.0], scale="auto", gtol=1e-8, max_iter=3000)
    assert a.converged and b.converged, (a.message, b.message)
    np.testing.assert_allclose(a.theta, b.theta, atol=1e-4)
    # Poisson score at the optimum, in units of the information: the Newton decrement
    assert a.decrement < 1e-10


def test_rosenbrock_valley():
    obj = fit.least_squares(rosenbrock, jac="fd")
    a = fit.gauss_newton(obj, [-1.2, 1.0])
    b = fit.lbfgs(obj, [-1.2, 1.0], gtol=1e-10, max_iter=5000)
    np.testing.assert_allclose(a.theta, [1, 1], atol=1e-5)
    np.testing.assert_allclose(b.theta, [1, 1], atol=1e-5)


@pytest.mark.parametrize("driver", ["gn", "lbfgs"])
def test_fixed_parameters_are_bit_identical(driver):
    x, y, _ = decay_data()
    obj = fit.least_squares(decay_residual(x, y), jac="fd")
    start = np.array([3.0, 0.5, 1.2345678901234567])
    if driver == "gn":
        r = fit.gauss_newton(obj, start, fixed=[2])
    else:
        r = fit.lbfgs(obj, start, fixed=np.array([False, False, True]), max_iter=3000, gtol=1e-8)
    assert r.theta[2] == start[2] and r.fixed.tolist() == [False, False, True]
    assert r.converged


@pytest.mark.parametrize("driver", ["gn", "lbfgs"])
def test_non_finite_region_is_rolled_back(driver):
    """r = exp(theta) - 2 (root log 2) is undefined (NaN) for theta > 1. From theta = -3 the full Gauss-Newton step
    lands near 36: it must be rolled back, and no iterate may enter the NaN region."""
    base = fit.least_squares(lambda t: np.exp(t) - 2.0, jac="fd")
    seen = []

    class Guarded:
        def value(self, th):
            return np.nan if th[0] > 1 else base.value(th)

        def terms(self, th):
            t = base.terms(th)
            return t._replace(nll=np.nan) if th[0] > 1 else t

        def value_and_grad(self, th):
            v, g = base.value_and_grad(th)
            return (np.nan, g) if th[0] > 1 else (v, g)

    def cb(th, *_):
        seen.append(th.copy())

    if driver == "gn":
        r = fit.gauss_newton(Guarded(), [-3.0], damping=1e-12, callback=cb)
        assert r.rejected_steps > 0
    else:
        r = fit.lbfgs(Guarded(), [-3.0], callback=cb, max_iter=3000, gtol=1e-10)
    assert r.converged, r.message
    assert all(th[0] <= 1 for th in seen)
    np.testing.assert_allclose(r.theta, [np.log(2.0)], atol=1e-6)


def test_optimum_does_not_depend_on_scale():
    x, y, _ = decay_data()
    obj = fit.least_squares(decay_residual(x, y), jac="fd")
    ref = fit.gauss_newton(obj, [3.0, 0.5, 1.0])
    for scale in (None, "auto", [1e-3, 1e2, 1.0], [10.0, 0.01, 3.0]):
        a = fit.gauss_newton(obj, [3.0, 0.5, 1.0], scale=scale)
        np.testing.assert_allclose(a.theta, ref.theta, atol=1e-8)
        b = fit.lbfgs(obj, [3.0, 0.5, 1.0], scale=scale if scale != [10.0, 0.01, 3.0] else "auto",
                      gtol=1e-9, max_iter=20000)
        np.testing.assert_allclose(b.theta, ref.theta, atol=1e-4)


def test_covariance_helpers():
    H = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
    inv = np.linalg.inv(H)
    np.testing.assert_allclose(fit.covariance(H), inv, rtol=1e-12)
    np.testing.assert_allclose(fit.marginal_covariance(H, [0, 1]), inv[:2, :2], rtol=1e-12)
    c = fit.covariance(H, fixed=[2])
    np.testing.assert_allclose(c[:2, :2], np.linalg.inv(H[:2, :2]), rtol=1e-12)
    assert np.all(c[2] == 0)
    assert np.all(np.isnan(fit.covariance(-H)))


# ------------------------------------------------------------------------------------------------ channel toy
TRUTH = np.array([np.log(40.0), 0.6, 5.0, 4.5])     # log yield, 1/lambda [1/m], global t0 [ns], slowness [ns/m]


def toy_model(theta, aux):
    xp = xp_of(theta)
    x = xp.asarray(aux, theta.dtype)
    return dict(charge=xp.exp(theta[0] - theta[1] * x), time=theta[2] + theta[3] * x)


def toy_data(seed, C=12, per=150, sigma_t=0.4, walk=0.0):
    rng = np.random.default_rng(seed)
    ch = np.repeat(np.arange(C), per)
    x = rng.uniform(0, 3, ch.size)
    g = rng.normal(0, 0.2, C)
    g -= g.mean()
    o = rng.normal(0, 1.0, C)
    o -= o.mean()
    raw = toy_model(TRUTH, x)
    q = rng.poisson(np.exp(g[ch]) * raw["charge"])
    t = raw["time"] + o[ch] + rng.normal(0, sigma_t, ch.size)
    if walk:
        t = t + walk * np.exp(-0.5 * (g[ch] + np.log(raw["charge"]) - np.log(20.0)))
    return dict(ch=ch, x=x, q=q, t=t, g=g, o=o, C=C)


def toy_problem(d, sigma_t=0.4, **kw):
    kw.setdefault("jac", "fd")
    return fit.ChannelProblem(d["ch"], d["C"], toy_model, charge=fit.Observed(d["q"], "poisson"),
                              time=fit.Observed(d["t"], "gaussian", sigma_t), aux=d["x"], **kw)


GAUGE = dict(log_gain=0, offset=2)


def test_closed_forms():
    ch = np.array([0, 0, 1, 1, 1, 2])
    M = np.array([1.0, 2.0, 1.0, 1.0, 2.0, 1.0])
    Q = np.array([2.0, 4.0, 3.0, 3.0, 6.0, 0.0])
    lg = fit.profile_log_gain(ch, Q, M, 4)
    np.testing.assert_allclose(lg[:2], np.log([2.0, 3.0]))
    assert np.isnan(lg[2]) and np.isnan(lg[3])
    np.testing.assert_allclose(fit.profile_log_gain(ch, Q, M, 4, "gaussian", 1.0)[:2], np.log([2.0, 3.0]))
    o = fit.profile_offset(ch, M + 1.5, M, 3, sigma=[1, 2, 1, 1, 1, 1])
    np.testing.assert_allclose(o, 1.5)
    s, c = fit.gauge([1.0, 2.0, np.nan, 3.0])
    assert s == 2.0 and np.isnan(c[2]) and c[3] == 1.0


def test_closed_form_profile_is_the_joint_optimum():
    d = toy_data(3)
    prob = toy_problem(d)
    start = TRUTH + [0.0, 0.1, 0.0, -0.3]
    a = fit.fit_channels(prob, start, profile="closed_form", gauge=GAUGE)
    b = fit.fit_channels(prob, start, profile="schur", gauge=GAUGE, nuisance0=prob.zeros())
    c = fit.fit_channels(prob, start, method="lbfgs", gauge=GAUGE, scale="auto")
    assert a.converged and b.converged and c.converged, (a.message, b.message, c.message)
    for r in (b, c):
        np.testing.assert_allclose(r.theta, a.theta, atol=2e-5)
        np.testing.assert_allclose(r.log_gain, a.log_gain, atol=2e-5)
        np.testing.assert_allclose(r.offset, a.offset, atol=2e-5)
    # at the profiled optimum the channel gradients vanish and the gauge holds
    bl = fit.blocks(prob, a.theta, a.nuisance)
    assert np.max(np.abs(bl.g_n)) < 1e-6 and a.decrement < 1e-10
    assert abs(a.log_gain.mean()) < 1e-12 and abs(a.offset.mean()) < 1e-12
    assert abs(a.gauge_nll_change) < 1e-8
    # truth inside the errors
    assert np.all(np.abs(a.theta - TRUTH) < 5 * a.theta_err)
    assert np.all(np.abs(a.log_gain - d["g"]) < 5 * a.log_gain_err)


def test_schur_step_equals_dense_step():
    d = toy_data(4, C=6, per=80)
    prob = toy_problem(d)
    theta = TRUTH + [0.05, 0.1, 0.2, -0.3]
    nu = prob.zeros() + 0.1
    bl = fit.blocks(prob, theta, nu)
    free = np.array([False, True, False, True])     # the gauge parameters are degenerate with the channel terms
    for lam in (0.0, 1e-2):
        dt, dn = CH._step(bl, free, lam)
        H, g = bl.dense(free)
        if lam:
            dg = np.diag(H).copy()
            H = H + lam * np.diag(dg)
        full = np.linalg.solve(H, -g)
        np.testing.assert_allclose(np.r_[dt, dn.ravel()], full, rtol=1e-8, atol=1e-12)


def _dense_reported_cov(prob, bl, free, spec, weights):
    """Dense reference: invert the full information, then the linear gauge map."""
    H, _ = bl.dense(free)
    cov = np.linalg.inv(H)
    C, m, p = prob.n_channels, prob.m, free.size
    pf = int(free.sum())
    # reported: theta (all p, held ones = constant + gauge shift), nu centred for gauge kinds
    T = np.zeros((p + C * m, pf + C * m))
    T[np.flatnonzero(free), np.arange(pf)] = 1.0
    for a, kind in enumerate(prob.kinds):
        cols = pf + np.arange(C) * m + a
        w = weights
        if kind in spec:
            T[spec[kind], cols] = w
        for c in range(C):
            r = p + c * m + a
            T[r, pf + c * m + a] += 1.0
            if kind in spec:
                T[r, cols] -= w
    return T @ cov @ T.T


@pytest.mark.parametrize("weights", [None, "information"])
def test_block_errors_equal_the_dense_inverse(weights):
    d = toy_data(5, C=8, per=100)
    prob = toy_problem(d)
    r = fit.fit_channels(prob, TRUTH, gauge=GAUGE, gauge_weights=weights)
    bl = fit.blocks(prob, r.theta, r.nuisance)
    free = np.array([False, True, False, True])
    w = np.full(prob.n_channels, 1 / prob.n_channels) if weights is None else None
    if weights == "information":
        # the weights differ per kind: check each kind separately with its own map
        covs = []
        for a, kind in enumerate(prob.kinds):
            wk = bl.H_nn[:, a, a] / bl.H_nn[:, a, a].sum()
            covs.append(_dense_reported_cov(prob, bl, free, {kind: GAUGE[kind]}, wk))
    else:
        covs = [_dense_reported_cov(prob, bl, free, {"log_gain": 0, "offset": 2}, w)]
    p, C, m = 4, prob.n_channels, prob.m
    for a, kind in enumerate(prob.kinds):
        ref = covs[a] if weights == "information" else covs[0]
        np.testing.assert_allclose(r.cov_theta[np.ix_(free, free)], ref[:p, :p][np.ix_(free, free)], rtol=1e-8)
        j = GAUGE[kind]
        np.testing.assert_allclose(r.cov_theta[j, j], ref[j, j], rtol=1e-8)
        np.testing.assert_allclose(r.cov_theta[j, free], ref[j, :p][free], rtol=1e-7, atol=1e-14)
        err_ref = np.sqrt(np.diag(ref)[p + a::m])
        np.testing.assert_allclose(getattr(r, kind + "_err"), err_ref, rtol=1e-8)
    if weights is None:
        np.testing.assert_allclose(r.cov_theta[0, 2], covs[0][0, 2], rtol=1e-7, atol=1e-14)


def test_absolute_channels_without_gauge():
    d = toy_data(6, C=5, per=100)
    prob = toy_problem(d)
    r = fit.fit_channels(prob, TRUTH, fixed=[0, 2])
    bl = fit.blocks(prob, r.theta, r.nuisance)
    free = np.array([False, True, False, True])
    H, _ = bl.dense(free)
    cov = np.linalg.inv(H)
    np.testing.assert_allclose(r.cov_theta[np.ix_(free, free)], cov[:2, :2], rtol=1e-8)
    np.testing.assert_allclose(r.log_gain_err, np.sqrt(np.diag(cov)[2::2]), rtol=1e-8)
    np.testing.assert_allclose(r.offset_err, np.sqrt(np.diag(cov)[3::2]), rtol=1e-8)
    assert r.theta[0] == TRUTH[0] and r.theta[2] == TRUTH[2]


def test_wrong_gauge_parameter_warns():
    d = toy_data(7, C=4, per=60)
    with pytest.warns(RuntimeWarning, match="gauge move changed the NLL"):
        fit.fit_channels(toy_problem(d), TRUTH, gauge=dict(log_gain=1))


def test_crb_matches_the_spread_over_seeds():
    """300 toys: pulls of the global parameters and of the centred channel terms have unit width (the standard
    error of each width is 0.04)."""
    pulls = []
    for seed in range(300):
        d = toy_data(5000 + seed, C=8, per=60)
        r = fit.fit_channels(toy_problem(d), TRUTH, gauge=GAUGE)
        assert r.converged
        pulls.append(np.r_[(r.theta - TRUTH) / np.where(r.theta_err > 0, r.theta_err, 1),
                           (r.log_gain - d["g"]) / r.log_gain_err, (r.offset - d["o"]) / r.offset_err])
    pulls = np.array(pulls)
    std, mean = pulls.std(axis=0), pulls.mean(axis=0)
    assert np.all(np.abs(std - 1) < 0.15), std
    assert np.all(np.abs(mean) < 0.3), mean


def test_problem_validation():
    d = toy_data(8, C=3, per=10)
    with pytest.raises(ValueError, match="no rows"):
        fit.ChannelProblem(d["ch"], 4, toy_model, charge=fit.Observed(d["q"], "poisson"), time=None, offset=False,
                           aux=d["x"], jac="fd")
    with pytest.raises(ValueError, match="needs a time"):
        fit.ChannelProblem(d["ch"], 3, toy_model, charge=fit.Observed(d["q"], "poisson"), aux=d["x"], jac="fd")
    p = fit.ChannelProblem(d["ch"], 3, toy_model, charge=fit.Observed(d["q"], "poisson"), offset=False,
                           aux=d["x"], jac="fd")
    assert p.kinds == ("log_gain",)
    r = fit.fit_channels(p, TRUTH, fixed=[2, 3], gauge=dict(log_gain=0))
    assert r.offset is None and r.converged


# ------------------------------------------------------------------------------------------------ coupled model
def walk_model(theta, aux, log_gain, offset):
    """Time walk falling with the expected charge (a walk linear in the expected log charge would be degenerate with
    the slowness and the offsets)."""
    xp = xp_of(theta)
    x = xp.asarray(aux, theta.dtype)
    logm = theta[0] - theta[1] * x
    lq = logm + log_gain
    return dict(charge=xp.exp(lq), time=theta[2] + theta[3] * x + offset + theta[4] * xp.exp(-0.5 * (lq - np.log(20.0))))


def walk_problem(d, jac="fd"):
    return fit.ChannelProblem(d["ch"], d["C"], walk_model, charge=fit.Observed(d["q"], "poisson"),
                              time=fit.Observed(d["t"], "gaussian", 0.4), aux=d["x"], coupled=True, jac=jac)


def test_coupled_walk_fit_and_dense_errors():
    truth = np.r_[TRUTH, 0.8]
    d = toy_data(9, C=6, per=150, walk=0.8)
    prob = walk_problem(d)
    r = fit.fit_channels(prob, truth + [0, 0.1, 0, 0.2, -0.3], gauge=GAUGE)
    assert r.converged, r.message
    assert np.all(np.abs(r.theta - truth) < 5 * np.where(r.theta_err > 0, r.theta_err, 1))
    bl = fit.blocks(prob, r.theta, r.nuisance)
    assert np.max(np.abs(bl.H_nn[:, 0, 1])) > 0      # the 2 x 2 blocks really are coupled
    free = np.array([False, True, False, True, True])
    ref = _dense_reported_cov(prob, bl, free, GAUGE, np.full(prob.n_channels, 1 / prob.n_channels))
    np.testing.assert_allclose(r.cov_theta, ref[:5, :5], rtol=1e-7, atol=1e-14)
    np.testing.assert_allclose(r.log_gain_err, np.sqrt(np.diag(ref)[5::2]), rtol=1e-8)
    np.testing.assert_allclose(r.offset_err, np.sqrt(np.diag(ref)[6::2]), rtol=1e-8)


# ------------------------------------------------------------------------------------------------ JAX
def test_jacfwd_matches_fd_and_float32_agrees(x64):
    import jax.numpy as jnp
    rng = np.random.default_rng(0)
    A = rng.normal(size=(500, 6))

    def fn(theta):
        xp = xp_of(theta)
        return xp.exp(0.1 * (xp.asarray(A, theta.dtype) @ theta)) + theta[0] * theta[1]

    th = rng.normal(size=6)
    J64 = np.asarray(fit.jacobian(fn, th, "fwd", dtype=np.float64))
    J32 = np.asarray(fit.jacobian(fn, th, "fwd", dtype=np.float32))
    Jfd = fit.jacobian(fn, th, "fd")
    assert J32.dtype == np.float32 and J64.dtype == np.float64
    np.testing.assert_allclose(Jfd, J64, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(J32, J64, rtol=1e-5, atol=1e-6)
    # the normal equations reduce on the device at full precision
    H32, _ = B.normal_equations(jnp.asarray(J32))
    np.testing.assert_allclose(H32, J64.T @ J64, rtol=2e-5)


def test_float64_jax_needs_x64():
    jax = pytest.importorskip("jax")
    if jax.config.read("jax_enable_x64"):
        pytest.skip("x64 already enabled in this process")
    with pytest.raises(ValueError, match="jax_enable_x64"):
        fit.jacobian(lambda t: t, np.zeros(2), "fwd", dtype=np.float64)


def test_channel_fit_jax_float64_and_float32(x64):
    d = toy_data(10, C=10, per=120)
    ref = fit.fit_channels(toy_problem(d, jac="fd"), TRUTH, gauge=GAUGE)
    r64 = fit.fit_channels(toy_problem(d, jac="fwd", dtype=np.float64), TRUTH, gauge=GAUGE)
    r32 = fit.fit_channels(toy_problem(d, jac="fwd", dtype=np.float32), TRUTH, gauge=GAUGE, tol=1e-4)
    assert r64.converged and r32.converged, r32.message
    np.testing.assert_allclose(r64.theta, ref.theta, atol=1e-6)
    np.testing.assert_allclose(r64.theta_err, ref.theta_err, rtol=1e-5)
    assert np.all(np.abs(r32.theta - ref.theta) < 0.02 * np.where(ref.theta_err > 0, ref.theta_err, 1))
    assert np.all(np.abs(r32.log_gain - ref.log_gain) < 0.02 * ref.log_gain_err)


def test_coupled_jvp_rows_match_fd(x64):
    d = toy_data(11, C=4, per=50, walk=0.8)
    theta = np.r_[TRUTH, 0.8]
    nu = np.random.default_rng(1).normal(0, 0.1, (4, 2))
    a = fit.blocks(walk_problem(d, "fd"), theta, nu)
    b = fit.blocks(walk_problem(d, "fwd"), theta, nu)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-8)


def test_jax_objectives_gn(x64):
    x, y, _ = decay_data()
    a = fit.gauss_newton(fit.least_squares(decay_residual(x, y), jac="fd"), [3.0, 0.5, 1.0])
    b = fit.gauss_newton(fit.least_squares(decay_residual(x, y), jac="fwd"), [3.0, 0.5, 1.0])
    np.testing.assert_allclose(a.theta, b.theta, atol=1e-8)
    expected, k, _ = counts_data()
    c = fit.lbfgs(fit.scalar(lambda t: fit.poisson(expected, k, jac="fd").value(np.asarray(t)), grad="fd"),
                  [5.0, 0.0, 1.0, 1.0], scale=[1, 1, 1, 1], gtol=1e-6, max_iter=5000)
    d_ = fit.gauss_newton(fit.poisson(expected, k, jac="fwd"), [5.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(c.theta, d_.theta, atol=2e-3)
