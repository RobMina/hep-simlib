"""hepsimlib.testing: the gradient harness detects wrong gradients and passes right ones in float32 and float64;
closures give unit pulls on a closed toy and report (not assert) a mismatch bias; the expected score is zero at
the truth. NumPy everywhere; the JAX reference-gradient tests skip without JAX."""
import numpy as np
import pytest

from hepsimlib import testing as T


def f_np(x):
    return np.sum(np.sin(x) * np.exp(0.3 * x)) + x[0] * x[1] ** 2


def g_np(x):
    g = np.cos(x) * np.exp(0.3 * x) + 0.3 * np.sin(x) * np.exp(0.3 * x)
    g[0] += x[1] ** 2
    g[1] += 2 * x[0] * x[1]
    return g


X = np.array([0.3, -1.2, 2.5, 0.7])


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("directions", ["axes", 5])
def test_correct_gradient_passes(dtype, directions):
    r = T.check_gradient(f_np, X.astype(dtype), g_np, directions=directions)
    assert r.ok, str(r)
    assert r.dtype == np.dtype(dtype).name
    T.assert_gradient(f_np, X.astype(dtype), g_np(X))


@pytest.mark.parametrize("dtype, off", [(np.float64, 1e-3), (np.float64, 1e-4), (np.float32, 5e-2)])
def test_wrong_gradient_fails(dtype, off):
    def bad(x):
        g = g_np(x)
        g[2] *= 1 + off
        return g
    r = T.check_gradient(f_np, X.astype(dtype), bad)
    assert not r.ok
    assert np.argmax(r.error) == 2
    with pytest.raises(AssertionError, match="FAILED"):
        r.assert_ok()


def test_jacobian_check():
    A = np.array([[1.0, 2.0, 0.5], [0.0, -1.0, 3.0]])

    def F(x):
        return np.tanh(A @ x)

    def J(x):
        return (1 - np.tanh(A @ x) ** 2)[:, None] * A

    x = np.array([0.1, 0.2, -0.3])
    assert T.check_jacobian(F, x, J).ok
    bad = J(x).copy()
    bad[1, 2] *= 1.001
    assert not T.check_jacobian(F, x, bad).ok


def test_sweep_localises_a_wrong_region():
    """A hand-written gradient that is wrong only for x0 > 0.5 (a branch with a typo): the sweep shows where."""
    f = lambda x: np.where(x[0] > 0.5, x[0] ** 3, 0.75 * x[0] - 0.25) + x[1] ** 2   # noqa: E731
    g = lambda x: np.array([np.where(x[0] > 0.5, 3.3 * x[0] ** 2, 0.75), 2 * x[1]])   # noqa: E731
    s = T.gradient_sweep(f, np.array([0.0, 1.0]), [1.0, 0.0], 1.0, n=21, grad=g)
    bad = s["error"] > 1
    assert np.array_equal(bad, s["t"] > 0.45)
    assert np.allclose(s["ad"][~bad], s["fd"][~bad])


# ------------------------------------------------------------------------------------------------ closures
def gauss_sim(truth, rng, n=200, sigma=2.0):
    return truth[0] + sigma * rng.standard_normal(n), sigma


def gauss_fit(data):
    y, s = data
    return np.array([y.mean()]), np.array([s / np.sqrt(y.size)])


def test_closure_pulls():
    r = T.closure(gauss_sim, gauss_fit, [3.0], range(400), names=("mean",))
    r.assert_unbiased()
    assert abs(r.coverage()[0] - 0.683) < 0.07
    assert "mean" in r.summary()


def test_closure_detects_wrong_errors_and_failures():
    def overconfident(data):
        e, s = gauss_fit(data)
        return e, 0.7 * s
    r = T.closure(gauss_sim, overconfident, [3.0], range(400))
    with pytest.raises(AssertionError, match="pull widths"):
        r.assert_unbiased()

    def flaky(data):
        if data[0][0] > 6:
            raise RuntimeError("did not converge")
        return gauss_fit(data)
    r = T.closure(gauss_sim, flaky, [3.0], range(200))
    assert r.failures and r.n + len(r.failures) == 200
    with pytest.raises(AssertionError, match="fits failed"):
        r.assert_unbiased()


def test_mismatch_closure_reports_a_bias():
    """The generator adds a skewed tail the Gaussian mean does not know: a bias to report, not to assert."""
    def skewed(truth, rng, n=200):
        return truth[0] + 2.0 * rng.standard_normal(n) + rng.exponential(0.5, n), 2.0
    r = T.mismatch_closure(skewed, gauss_fit, [3.0], range(300))
    assert r.mismatch and "model mismatch" in r.summary()
    assert abs(r.bias[0] - 0.5) < 4 * r.bias_err[0]
    with pytest.raises(AssertionError):
        r.assert_unbiased()


def poisson_nll(k, x):
    """Expected counts a exp(-b x); NLL of counts k."""
    def nll(theta):
        mu = theta[0] * np.exp(-theta[1] * x)
        return np.sum(mu - k * np.log(mu))

    def grad(theta):
        e = np.exp(-theta[1] * x)
        w = 1 - k / (theta[0] * e)
        return np.array([np.sum(w * e), np.sum(-w * theta[0] * x * e)])
    return nll, grad


def test_expected_and_mean_score_zero():
    x = np.linspace(0, 3, 30)
    truth = np.array([50.0, 0.8])
    asimov = truth[0] * np.exp(-truth[1] * x)
    nll, grad = poisson_nll(asimov, x)
    T.expected_gradient_zero(nll, truth, grad=grad, tol=1e-10).assert_ok()
    e = np.exp(-truth[1] * x)
    J = np.stack([e, -truth[0] * x * e], 1)
    info = J.T @ (J / asimov[:, None])
    assert T.expected_score_zero(nll, truth, grad=grad, info=info).ok
    # a mis-specified likelihood (wrong slope convention) has a non-zero expected score
    nll2, grad2 = poisson_nll(truth[0] * np.exp(-1.05 * truth[1] * x), x)
    assert not T.expected_score_zero(nll2, truth, grad=grad2, info=info).ok

    def score(k, theta):
        return poisson_nll(k, x)[1](theta)
    sim = lambda th, rng: rng.poisson(th[0] * np.exp(-th[1] * x))   # noqa: E731
    T.mean_score_zero(score, sim, truth, range(300)).assert_ok()
    sim_bad = lambda th, rng: rng.poisson(1.1 * th[0] * np.exp(-th[1] * x))   # noqa: E731
    assert not T.mean_score_zero(score, sim_bad, truth, range(300)).ok


# ------------------------------------------------------------------------------------------------ JAX
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_jax_reference_gradient(dtype, x64):
    import jax.numpy as jnp

    def f(x):
        return jnp.sum(jnp.sin(x) * jnp.exp(0.3 * x)) + x[0] * x[1] ** 2

    r = T.check_gradient(f, X.astype(dtype), directions=7)
    assert r.ok, str(r)

    def F(x):
        return jnp.stack([jnp.sum(x ** 2), jnp.prod(jnp.cos(x))])
    assert T.check_jacobian(F, X.astype(dtype)).ok
    # an AD gradient of a function with a custom wrong derivative is caught
    import jax

    @jax.custom_jvp
    def h(x):
        return jnp.sum(x ** 3)

    @h.defjvp
    def h_jvp(primals, tangents):
        (x,), (t,) = primals, tangents
        return h(x), jnp.sum(3.3 * x ** 2 * t)
    assert not T.check_gradient(h, X.astype(dtype)).ok
