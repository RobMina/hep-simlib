"""hepsimlib.likelihood: Poisson detection probabilities (exact, stable, derivatives), the censored crossing
likelihood (normalised, analytic gradient = FD, JAX twin, zero mean score at truth, closure with fitted widths), and
order-statistic timing (Monte Carlo of sampled arrivals, normalisation, gradients, t0 closure)."""
import numpy as np
import pytest

scipy = pytest.importorskip("scipy")
from scipy import integrate, optimize, stats  # noqa: E402

from hepsimlib import likelihood as L  # noqa: E402
from hepsimlib import testing as T  # noqa: E402


# ------------------------------------------------------------------------------------------------ detection
MUS = np.geomspace(1e-6, 1e4, 81)


@pytest.mark.parametrize("n", [1, 2, 5, 20, 60])
def test_detection_matches_the_poisson_tail(n):
    lp = L.log_p_detect(MUS, n)
    lq = L.log_p_miss(MUS, n)
    ref_p = stats.poisson.logsf(n - 1, MUS)
    ref_q = stats.poisson.logcdf(n - 1, MUS)
    ok = np.isfinite(ref_p) & (ref_p > -700)
    np.testing.assert_allclose(lp[ok], ref_p[ok], rtol=1e-9, atol=1e-12)
    okq = np.isfinite(ref_q) & (ref_q > -700)
    np.testing.assert_allclose(lq[okq], ref_q[okq], rtol=1e-9, atol=1e-12)
    assert np.all(np.isfinite(lp)) and np.all(np.diff(lp) >= 0)
    assert np.all(lq <= 0) and np.all(np.diff(lq) <= 0)
    np.testing.assert_allclose(np.exp(lp) + np.exp(lq), 1.0, atol=1e-12)
    # deep tail: log P ~ log Pois(n; mu) for tiny mu
    np.testing.assert_allclose(lp[0], stats.poisson.logpmf(n, MUS[0]), rtol=1e-6)


def test_detection_derivatives_and_bernoulli_nll():
    for n in (1, 3, 12):
        for mu in (1e-3, 0.7, 3.0, 11.0, 40.0):
            x = np.array([mu])
            step = 6e-6 * min(1.0, mu)          # the function varies on the scale of mu
            T.assert_gradient(lambda m: float(L.log_p_detect(m, n)[0]), x, L.d_log_p_detect(x, n), rel_step=step)
            T.assert_gradient(lambda m: float(L.log_p_miss(m, n)[0]), x, L.d_log_p_miss(x, n), rel_step=step)
    mu = np.array([0.5, 2.0, 7.0])
    seen = np.array([1.0, 0.0, 0.3])
    nll, g = L.detection_nll(mu, [1, 2, 3], seen, grad=True)
    T.assert_gradient(lambda m: float(L.detection_nll(m, [1, 2, 3], seen).sum()), mu, g)
    with pytest.raises(ValueError):
        L.log_p_detect(1.0, 0)
    with pytest.raises(ValueError):
        L.log_p_detect(1.0, 1.5)


def test_detection_expected_score_zero():
    """Asimov data (seen fraction = P) give a zero score at the truth in mu = A exp(-x / lam)."""
    x = np.linspace(0, 3, 25)
    n = 3
    truth = np.array([np.log(8.0), 1.2])

    def mu(th):
        return np.exp(th[0] - x / th[1])

    asimov = np.exp(L.log_p_detect(mu(truth), n))

    def nll(th):
        return float(L.detection_nll(mu(th), n, asimov).sum())

    def grad(th):
        m = mu(th)
        _, g = L.detection_nll(m, n, asimov, grad=True)
        return np.array([np.sum(g * m), np.sum(g * m * x / th[1] ** 2)])

    T.expected_gradient_zero(nll, truth, grad=grad, tol=1e-12).assert_ok()


# ------------------------------------------------------------------------------------------------ censored
def pred_sel(n=1, **kw):
    p = dict(mu_logq=np.log(8.0), mu_t=3.0, log_sigma_q=np.log(0.4), log_sigma_t=np.log(1.2), walk=0.9,
             logq_ref=np.log(8.0))
    p.update({k: v for k, v in kw.items() if k in p})
    s = dict(log_threshold=np.log(4.0), t_lo=-1.0, t_hi=6.0)
    s.update({k: v for k, v in kw.items() if k in s})
    return L.CrossingPrediction(**p), L.Selection(**s)


CASES = [dict(), dict(mu_logq=np.log(3.0)), dict(mu_logq=np.log(40.0), walk=-0.5), dict(walk=0.0),
         dict(mu_t=8.0, log_sigma_t=np.log(3.0)), dict(log_threshold=np.log(30.0)), dict(mu_logq=np.log(1.5),
                                                                                         log_threshold=np.log(9))]


def test_bivariate_normal_against_scipy():
    rng = np.random.default_rng(0)
    h = rng.normal(0, 2, 300)
    k = rng.normal(0, 2, 300)
    r = np.r_[rng.uniform(-0.999, 0.999, 200), rng.choice([-1, 1], 100) * rng.uniform(0.925, 0.9999, 100)]
    got = L.censored.bvn_upper(h, k, r)
    ref = np.array([stats.multivariate_normal.cdf([-hi, -ki], cov=[[1, ri], [ri, 1]], abseps=1e-14, releps=1e-14)
                    for hi, ki, ri in zip(h, k, r)])
    np.testing.assert_allclose(got, ref, atol=2e-13)
    np.testing.assert_allclose(L.censored.bvn_upper(h, k, 0.0), stats.norm.sf(h) * stats.norm.sf(k), atol=1e-15)
    for i in range(0, 300, 37):
        x = np.array([h[i], k[i], r[i]])
        T.assert_gradient(lambda v: float(L.censored.bvn_upper(v[0], v[1], v[2])), x,
                          np.array(L.censored._bvn_upper_grad(h[i], k[i], r[i])),
                          rel_step=1e-6 * min(1.0, 1 - abs(r[i])) / max(1.0, np.abs(x).max()), rtol=1e-5)


@pytest.mark.parametrize("case", range(len(CASES)))
def test_censored_probabilities_sum_to_one(case):
    pred, sel = pred_sel(**CASES[case])
    p_miss = float(np.exp(-L.censored_nll(pred, sel, False)))
    sq, st = np.exp(pred.log_sigma_q), np.exp(pred.log_sigma_t)

    def seen_mass(lq):
        m = pred.mu_t - pred.walk * (lq - pred.logq_ref)
        return stats.norm.pdf(lq, pred.mu_logq, sq) * (stats.norm.cdf(sel.t_hi, m, st) - stats.norm.cdf(sel.t_lo, m, st))
    seen, _ = integrate.quad(seen_mass, sel.log_threshold, pred.mu_logq + 15 * sq + 20, epsabs=1e-13, epsrel=1e-12)
    assert abs(seen + p_miss - 1) < 1e-9
    # the seen density integrates over the selected region too (2D)
    dens = lambda t, lq: np.exp(-float(L.censored_nll(pred, sel, True, lq, t)))   # noqa: E731
    two_d, _ = integrate.dblquad(dens, sel.log_threshold, pred.mu_logq + 8 * sq + 5, sel.t_lo, sel.t_hi,
                                 epsabs=1e-10, epsrel=1e-9)
    assert abs(two_d - seen) < 1e-7


def _rows(rng, n=400):
    pred, sel = pred_sel()
    mq = np.log(8.0) + rng.normal(0, 1.0, n)
    pred = pred._replace(mu_logq=mq, mu_t=3.0 + rng.normal(0, 2.0, n))
    lq = rng.normal(pred.mu_logq, 0.4)
    t = rng.normal(pred.mu_t - pred.walk * (lq - pred.logq_ref), 1.2)
    seen = (lq >= sel.log_threshold) & (t >= sel.t_lo) & (t <= sel.t_hi)
    return pred, sel, seen, np.where(seen, lq, np.nan), np.where(seen, t, np.nan)


def test_censored_analytic_gradient_matches_fd():
    pred, sel, seen, lq, t = _rows(np.random.default_rng(3), 60)
    assert 5 < seen.sum() < 55
    nll, g = L.censored_nll(pred, sel, seen, lq, t, grad=True)
    assert np.all(np.isfinite(nll)) and np.all(np.isfinite(g))
    for i in range(len(seen)):
        def f(eta, i=i):
            p = pred._replace(mu_logq=eta[0], mu_t=eta[1], log_sigma_q=eta[2], log_sigma_t=eta[3], walk=eta[4])
            return float(L.censored_nll(p, sel, seen[i], lq[i], t[i]))
        x = np.array([pred.mu_logq[i], pred.mu_t[i], pred.log_sigma_q, pred.log_sigma_t, pred.walk])
        T.assert_gradient(f, x, g[i], rtol=1e-6)


def test_censored_tails_are_finite():
    pred, sel = pred_sel(mu_logq=np.array([-30.0, 30.0, np.log(4.0)]), mu_t=np.array([3.0, 300.0, -50.0]))
    nll, g = L.censored_nll(pred, sel, np.zeros(3, bool), grad=True)
    assert np.all(np.isfinite(nll)) and np.all(np.isfinite(g))
    assert nll[0] < 1e-12            # far below threshold: a certain miss


def test_censored_mean_score_zero_and_width_closure():
    """Sampled from the model: the score at the truth averages to zero, and a fit of the shared predictors with
    fitted widths closes (pulls N(0, 1))."""
    truth = np.array([np.log(6.0), 3.0, np.log(0.4), np.log(1.2), 0.9])
    sel = L.Selection(np.log(4.0), -1.0, 6.0)

    def simulate(th, rng, n=2000):
        lq = rng.normal(th[0], np.exp(th[2]), n)
        t = rng.normal(th[1] - th[4] * (lq - np.log(8.0)), np.exp(th[3]))
        seen = (lq >= sel.log_threshold) & (t >= sel.t_lo) & (t <= sel.t_hi)
        return seen, np.where(seen, lq, np.nan), np.where(seen, t, np.nan)

    def pred(th):
        return L.CrossingPrediction(th[0], th[1], th[2], th[3], th[4], np.log(8.0))

    def score(data, th):
        seen, lq, t = data
        return L.censored_nll(pred(th), sel, seen, lq, t, grad=True)[1].sum(axis=0)

    T.mean_score_zero(score, simulate, truth, range(100)).assert_ok()

    def fit(data):
        seen, lq, t = data
        f = lambda th: float(L.censored_nll(pred(th), sel, seen, lq, t).sum())   # noqa: E731
        g = lambda th: L.censored_nll(pred(th), sel, seen, lq, t, grad=True)[1].sum(axis=0)   # noqa: E731
        r = optimize.minimize(f, truth + 0.05, jac=g, method="BFGS", options=dict(gtol=1e-8))
        H = np.array([(g(r.x + h) - g(r.x - h)) / 2e-5 for h in np.eye(5) * 1e-5])
        return r.x, np.sqrt(np.diag(np.linalg.inv(0.5 * (H + H.T))))

    T.closure(simulate, fit, truth, range(1000, 1080)).assert_unbiased(k=4)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_censored_jax_twin(dtype, x64):
    import jax
    import jax.numpy as jnp
    pred, sel, seen, lq, t = _rows(np.random.default_rng(5), 200)
    ref, gref = L.censored_nll(pred, sel, seen, lq, t, grad=True)
    jpred = L.CrossingPrediction(*(jnp.asarray(np.asarray(v), dtype) for v in pred))
    jsel = L.Selection(*(jnp.asarray(np.asarray(v), dtype) for v in sel))
    out = L.censored_nll(jpred, jsel, jnp.asarray(seen), jnp.asarray(lq, dtype), jnp.asarray(t, dtype))
    assert out.dtype == dtype
    tol = 1e-10 if dtype == "float64" else 2e-4
    np.testing.assert_allclose(np.asarray(out), ref, rtol=tol, atol=tol)
    if dtype == "float64":
        def total(eta):
            p = jpred._replace(mu_logq=eta[0], mu_t=eta[1], log_sigma_q=eta[2], log_sigma_t=eta[3], walk=eta[4])
            return jnp.sum(L.censored_nll(p, jsel, jnp.asarray(seen), jnp.asarray(lq), jnp.asarray(t)))
        eta = [jpred.mu_logq, jpred.mu_t, jnp.full(200, jpred.log_sigma_q), jnp.full(200, jpred.log_sigma_t),
               jnp.full(200, jpred.walk)]
        gj = jax.grad(lambda e: total(e))(eta)
        np.testing.assert_allclose(np.stack([np.asarray(v) for v in gj], -1), gref, rtol=1e-8, atol=1e-10)


# ------------------------------------------------------------------------------------------------ arrival
DELAYS = {"exp": L.ExponentialDelay(6.0), "emg": L.ExpGaussDelay(6.0, 1.5)}


def sample_kth(rng, n, mu, tau, sigma, k, t0=0.0):
    N = rng.poisson(mu, n)
    out = np.full(n, np.inf)
    for i in np.flatnonzero(N >= k):
        d = rng.exponential(tau, N[i]) + (rng.normal(0, sigma, N[i]) if sigma else 0.0)
        out[i] = t0 + np.partition(d, k - 1)[k - 1]
    return out


@pytest.mark.parametrize("name", ["exp", "emg"])
@pytest.mark.parametrize("mu, k", [(5.0, 1), (5.0, 3), (40.0, 1), (0.4, 1)])
def test_order_statistic_matches_monte_carlo(name, mu, k):
    delay = DELAYS[name]
    sigma = 0.0 if name == "exp" else 1.5
    tk = sample_kth(np.random.default_rng(11), 40000, mu, 6.0, sigma, k)
    grid = np.linspace(-3, 30, 34)
    pred = np.exp(L.arrival_log_cdf(grid, 0.0, mu, delay, k))
    emp = np.array([np.mean(tk <= g) for g in grid])
    err = np.sqrt(np.clip(pred * (1 - pred), 1e-12, None) / tk.size)
    assert np.all(np.abs(emp - pred) < 5 * err + 1e-4)
    # the density integrates to P(N >= k)
    tt = np.linspace(0.0 if name == "exp" else -20.0, 200, 200001)   # the exponential density jumps at 0
    dens = np.exp(L.first_arrival_logpdf(tt, 0.0, mu, delay, k))
    np.testing.assert_allclose(np.trapezoid(dens, tt), stats.poisson.sf(k - 1, mu), rtol=1e-5)


def test_tabulated_delay():
    emg = DELAYS["emg"]
    edges = np.linspace(-10, 80, 901)
    mid = 0.5 * (edges[1:] + edges[:-1])
    tab = L.TabulatedDelay(edges, emg.pdf(mid))
    d = np.linspace(-15, 90, 400)
    np.testing.assert_allclose(tab.cdf(d), emg.cdf(d), atol=2e-4)
    assert tab.cdf(-20.0) == 0 and tab.cdf(100.0) == 1 and tab.pdf(100.0) == 0
    T.assert_gradient(lambda x: float(tab.cdf(x[0])), np.array([3.33]), np.array([float(tab.pdf(3.33))]))
    lp = L.first_arrival_logpdf(np.array([2.0, 5.0]), 0.0, 4.0, tab, 2)
    lp_ref = L.first_arrival_logpdf(np.array([2.0, 5.0]), 0.0, 4.0, emg, 2)
    np.testing.assert_allclose(lp, lp_ref, atol=1e-2)      # histogram of 0.1 bins


def test_windowed_nll_normalisation():
    delay, mu, k, lo, hi = DELAYS["emg"], 3.0, 2, -2.0, 15.0
    tt = np.linspace(lo, hi, 100001)
    seen_mass = np.trapezoid(np.exp(-L.first_arrival_nll(tt, 0.0, mu, delay, k, window=(lo, hi))), tt)
    silent = np.exp(-L.first_arrival_nll(np.nan, 0.0, mu, delay, k, window=(lo, hi), seen=False))
    np.testing.assert_allclose(seen_mass + silent, 1.0, rtol=1e-6)
    cond = np.trapezoid(np.exp(-L.first_arrival_nll(tt, 0.0, mu, delay, k, window=(lo, hi), conditional=True)), tt)
    np.testing.assert_allclose(cond, 1.0, rtol=1e-6)
    assert L.first_arrival_nll(np.nan, 0.0, mu, delay, k, window=(lo, hi), seen=False, conditional=True) == 0


def test_t0_closure_from_first_arrivals():
    """Per-toy fit of the time offset from first arrivals of 200 sensors: unbiased, errors right, while the mean
    arrival minus the delay mean would need the full delay shape and mu to be right."""
    delay, mu, k, truth = DELAYS["emg"], 6.0, 1, np.array([2.5])

    def simulate(th, rng):
        t = sample_kth(rng, 200, mu, 6.0, 1.5, k, th[0])
        return t[np.isfinite(t)]

    def fit(t):
        f = lambda t0: float(L.first_arrival_nll(t, t0, mu, delay, k, conditional=True).sum())   # noqa: E731
        r = optimize.minimize_scalar(f, bracket=(0, 5), tol=1e-10)
        h = 1e-3
        curv = (f(r.x + h) - 2 * f(r.x) + f(r.x - h)) / h ** 2
        return np.array([r.x]), np.array([1 / np.sqrt(curv)])

    T.closure(simulate, fit, truth, range(150)).assert_unbiased(k=4)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_arrival_jax_gradients(dtype, x64):
    import jax
    import jax.numpy as jnp
    t = np.array([1.0, 4.0, 9.0, 0.5])
    seen = np.array([True, True, False, True])

    def nll_np(p):
        d = L.ExpGaussDelay(p[2], p[3])
        return float(L.first_arrival_nll(t, p[0], p[1], d, 2, window=(-2.0, 12.0), seen=seen).sum())

    def nll_jax(p):
        d = L.ExpGaussDelay(p[2], p[3])
        return jnp.sum(L.first_arrival_nll(jnp.asarray(t, p.dtype), p[0], p[1], d, 2, window=(-2.0, 12.0),
                                           seen=jnp.asarray(seen)))

    p = np.array([0.3, 5.0, 6.0, 1.5])
    v = nll_jax(jnp.asarray(p, dtype))
    np.testing.assert_allclose(float(v), nll_np(p), rtol=1e-10 if dtype == "float64" else 1e-5)
    g = np.asarray(jax.grad(nll_jax)(jnp.asarray(p, dtype)), np.float64)
    T.assert_gradient(nll_np, p, g, rtol=1e-6 if dtype == "float64" else 1e-3)
    # detection custom JVP: jax.grad equals the analytic derivative
    mu = jnp.asarray([0.01, 1.0, 30.0], dtype)
    gd = jax.grad(lambda m: jnp.sum(L.log_p_detect(m, 3)))(mu)
    np.testing.assert_allclose(np.asarray(gd), L.d_log_p_detect(np.asarray(mu, np.float64), 3),
                               rtol=1e-9 if dtype == "float64" else 1e-4)
