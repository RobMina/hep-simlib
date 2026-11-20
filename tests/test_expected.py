"""hepsimlib.expected: a toy scintillator bar read out at both ends (no experiment tables).

Interpolation reproduces the nodes and is linear between them; the end half cells and the axis policies; the
expected counts and binned arrival times equal the mean of a sampled chain that draws photons from the same table;
float32 = float64; gradients in positions, energies, times and scales = finite differences; deterministic; library
round trip; the per-channel arrival distribution plugs into the first-arrival likelihood."""
import numpy as np
import pytest

from hepsimlib import testing as T
from hepsimlib.expected import Steps, TableResponse, table_fields
from hepsimlib.library import Axis, OutOfDomain, store

L_BAR, WIDTH, LAM, V, TAU, Y0 = 1000.0, 20.0, 600.0, 150.0, 8.0, 40.0


def toy_arrays(nx=50, ny=4, policy="clip"):
    ax = Axis("x", "mm", np.linspace(0, L_BAR, nx + 1), frame="bar, along, from end 0", policy=policy, scale="linear")
    ay = Axis("y", "mm", np.linspace(0, WIDTH, ny + 1), frame="bar, across", policy=policy, scale="linear")
    cx = 0.5 * (ax.edges[1:] + ax.edges[:-1])
    cy = 0.5 * (ay.edges[1:] + ay.edges[:-1])
    dist = np.stack([cx, L_BAR - cx], -1)                                  # [nx, C]
    y = Y0 * np.exp(-dist / LAM)[:, None, :] * (1 + 0.2 * cy / WIDTH)[None, :, None]     # [nx, ny, C]
    de = np.linspace(0, 80, 321)
    lag = (dist / V)[:, None, :, None]
    F = np.where(de >= lag, -np.expm1(-(de - lag) / TAU), 0.0)
    F = np.broadcast_to(F / F[..., -1:], (nx, ny, 2, de.size)).copy()
    return [ax, ay], y, de, F


def toy_table(**kw):
    axes, y, de, F = toy_arrays(**kw)
    return TableResponse.from_arrays(axes, y, de, F, name="toy.optics.bar")


STEPS = Steps(position=np.array([[123.4, 3.3], [512.7, 14.1], [880.2, 9.9], [5.0, 1.0], [999.0, 19.5]]),
              edep=np.array([1.2, 0.4, 2.0, 0.7, 0.3]), time=np.array([1.37, 2.11, 0.42, 5.05, 3.3]))


def test_nodes_linearity_and_end_cells():
    tab = toy_table()
    axes, y, _, _ = toy_arrays()
    cx = 0.5 * (axes[0].edges[1:] + axes[0].edges[:-1])
    cy = 0.5 * (axes[1].edges[1:] + axes[1].edges[:-1])
    X, Y = np.meshgrid(cx, cy, indexing="ij")
    got = tab.interpolate_yield(np.stack([X.ravel(), Y.ravel()], 1))
    np.testing.assert_allclose(got, y.reshape(-1, 2), rtol=1e-14)
    # midway between centres: the average of the two nodes
    mid = tab.interpolate_yield(np.array([[0.5 * (cx[3] + cx[4]), cy[1]]]))
    np.testing.assert_allclose(mid[0], 0.5 * (y[3, 1] + y[4, 1]), rtol=1e-14)
    # an exactly linear table is reproduced everywhere inside the centres
    lin = TableResponse.from_arrays(axes, np.broadcast_to((2 + 0.01 * cx)[:, None, None], y.shape).copy(),
                                    np.array([0.0, 1.0]), np.broadcast_to([0.0, 1.0], y.shape + (2,)).copy())
    xs = np.random.default_rng(0).uniform(cx[0], cx[-1], 50)
    np.testing.assert_allclose(lin.interpolate_yield(np.stack([xs, np.full(50, 7.0)], 1))[:, 0], 2 + 0.01 * xs,
                               rtol=1e-13)
    # outer half cells hold the end value; clip beyond the edges
    np.testing.assert_allclose(tab.interpolate_yield(np.array([[1.0, cy[0]], [-50.0, cy[0]]])), y[[0, 0], 0],
                               rtol=1e-14)
    # close to the continuous model (interpolation error, 20 mm cells against a 600 mm attenuation length)
    pts = np.random.default_rng(1).uniform([cx[0], cy[0]], [cx[-1], cy[-1]], (200, 2))
    truth = Y0 * np.exp(-np.stack([pts[:, 0], L_BAR - pts[:, 0]], 1) / LAM) * (1 + 0.2 * pts[:, 1:] / WIDTH)
    np.testing.assert_allclose(tab.interpolate_yield(pts), truth, rtol=2e-4)


def test_policies():
    st = STEPS._replace(position=np.r_[STEPS.position[:4], [[1200.0, 5.0]]])
    with pytest.raises(OutOfDomain, match="'x'"):
        toy_table(policy="refuse").expected(st)
    r = toy_table(policy="refuse").expected(st._replace(mask=np.array([1, 1, 1, 1, 0], bool)))
    assert np.all(np.isfinite(r.counts))
    assert np.all(np.isnan(toy_table(policy="nan").expected(st).counts))
    clip = toy_table().expected(st)
    assert np.all(np.isfinite(clip.counts)) and clip.codes[4, 0] == 2


def test_cdf_pdf_and_binned_are_consistent():
    r = toy_table().expected(STEPS)
    tt = np.linspace(-5, 120, 50001)
    cdf, pdf = r.cdf(tt), r.pdf(tt)
    assert np.all(np.diff(cdf, axis=0) >= -1e-15)
    np.testing.assert_allclose(cdf[0], 0)
    np.testing.assert_allclose(cdf[-1], 1, rtol=1e-12)
    np.testing.assert_allclose(np.trapezoid(pdf, tt, axis=0), 1, rtol=1e-3)
    edges = np.arange(0, 101, 1.0)
    b = r.binned(edges)
    np.testing.assert_allclose(b.sum(1), r.counts, rtol=1e-12)
    np.testing.assert_allclose(b, (r.counts[:, None] * np.diff(r.cdf(edges), axis=0).T), rtol=1e-12, atol=1e-12)


def sample_mean(tab, steps, R, edges, rng):
    """A sampled chain on the same table: per photon, the corner cell is drawn with probability w * yield, the
    delay from that cell's CDF. Returns the mean counts and binned arrivals per event."""
    cells, w = tab._corners(np.asarray(steps.position))
    counts = np.zeros(tab.n_channels)
    hist = np.zeros((tab.n_channels, len(edges) - 1))
    for s in range(len(steps.edep)):
        for c in range(tab.n_channels):
            wy = w[:, s] * tab._yield[cells[:, s], c]
            n = rng.poisson(R * steps.edep[s] * wy.sum())
            k = rng.choice(len(wy), size=n, p=wy / wy.sum())
            u = rng.uniform(size=n)
            d = np.empty(n)
            for kk in np.unique(k):
                sel = k == kk
                d[sel] = np.interp(u[sel], tab._cdf[cells[kk, s], c], tab.delay_edges)
            t = steps.time[s] + d
            counts[c] += n
            hist[c] += np.histogram(t, edges)[0]
    return counts / R, hist / R


def test_expected_equals_the_mean_of_the_sampled_chain():
    tab = toy_table()
    R = 4000
    edges = np.arange(0, 61, 2.0)
    r = tab.expected(STEPS)
    counts, hist = sample_mean(tab, STEPS, R, edges, np.random.default_rng(7))
    err_c = np.sqrt(r.counts / R)
    assert np.all(np.abs(counts - r.counts) < 5 * err_c)
    b = r.binned(edges)
    ok = b > 0
    pulls = (hist[ok] - b[ok]) / np.sqrt(b[ok] / R)
    assert np.max(np.abs(pulls)) < 5
    assert np.mean(pulls ** 2) < 1.5


def test_deterministic_and_dtypes(x64):
    import jax
    host = toy_table()
    a, b = host.expected(STEPS), host.expected(STEPS)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    edges = np.arange(0, 61, 2.0)
    for dtype, tol in (("float64", 1e-12), ("float32", 2e-5)):
        dev = host.to_device(dtype)
        r = dev.expected(STEPS)
        assert r.counts.dtype == dtype
        np.testing.assert_allclose(np.asarray(r.counts), a.counts, rtol=tol)
        np.testing.assert_allclose(np.asarray(r.binned(edges)), a.binned(edges), rtol=10 * tol, atol=tol * 1e2)
        jitted = jax.jit(lambda p, e, t: dev.expected(Steps(p, e, t)).counts)
        np.testing.assert_allclose(np.asarray(jitted(*STEPS[:3])), np.asarray(r.counts), rtol=tol)


def _flatten(pos, edep, times, ys, cs):
    return np.r_[pos.ravel(), edep, times, ys, cs]


def _loss(tab, v, n=5):
    xp = tab.xp
    pos = v[:2 * n].reshape(n, 2)
    edep, times = v[2 * n:3 * n], v[3 * n:4 * n]
    ys, cs = v[4 * n], v[4 * n + 1:4 * n + 3]
    r = tab.expected(Steps(pos, edep, times), yield_scale=ys, channel_scale=cs)
    edges = xp.asarray(np.arange(0, 41, 1.0), tab.dtype)
    wts = xp.asarray(np.linspace(0.5, 1.5, 40), tab.dtype)
    return xp.sum(r.counts * xp.asarray([1.0, 2.0], tab.dtype)) + xp.sum(r.binned(edges) * wts) \
        + xp.sum(r.cdf(xp.asarray([7.31, 12.93], tab.dtype)))   # off the delay-node kinks


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_gradients_match_finite_differences(dtype, x64):
    import jax
    import jax.numpy as jnp
    host = toy_table()
    dev = host.to_device(dtype)
    v = _flatten(STEPS.position, STEPS.edep, STEPS.time, 1.1, np.array([0.9, 1.05]))
    g = np.asarray(jax.grad(lambda x: _loss(dev, x))(jnp.asarray(v, dtype)), np.float64)
    assert np.all(np.isfinite(g)) and np.any(g[:10] != 0)
    # reference: central differences of the float64 host table
    rep = T.check_gradient(lambda x: float(_loss(host, x)), v, g, rtol=1e-6 if dtype == "float64" else 2e-3)
    assert rep.ok, str(rep)


def test_library_round_trip(tmp_path):
    axes, y, de, F = toy_arrays()
    arrays, fields = table_fields(axes, y, de, F)
    fields.update(name="toy.optics.bar", subsystem="toy", version=1, description="toy bar read out at both ends",
                  source=dict(samples=[dict(dataset="analytic", selection="all", events=0)]),
                  generator=dict(name="analytic", release="test"), release=dict(repository="test", commit="0" * 7),
                  seeds=dict(build=0), geometry=dict(tag="toy", hash=None), config_hash=None,
                  split=dict(training="analytic", held_out="none", rule="analytic"),
                  domain=dict(coverage="the bar"), validation=dict(status="not validated", record=None),
                  unrecorded=dict(config_hash="no configuration"))
    store.save(tmp_path / "bar.npz", arrays, fields)
    lib = store.load(tmp_path / "bar.npz")
    tab = TableResponse.from_library(lib)
    ref = toy_table().expected(STEPS)
    got = tab.expected(STEPS)
    assert all(np.array_equal(a, b) for a, b in zip(got, ref))
    assert [a.name for a in tab.axes] == ["x", "y"]


def test_first_arrival_from_the_expected_response():
    """The per-channel arrival distribution plugs into the order-statistic likelihood: a dim response (a few
    photoelectrons, so the first-arrival density is not a spike) integrates to P(N >= 1), and the density is
    mu f exp(-mu F)."""
    L = pytest.importorskip("hepsimlib.likelihood")
    pytest.importorskip("scipy")
    r = toy_table().expected(STEPS, yield_scale=0.02)
    tt = np.linspace(-1, 120, 400001)
    for c in range(2):
        mu = r.counts[c]
        assert 0.3 < mu < 10
        dens = np.exp(L.first_arrival_logpdf(tt, 0.0, mu, r.channel(c)))
        np.testing.assert_allclose(np.trapezoid(dens, tt), -np.expm1(-mu), rtol=2e-4)
        t = np.array([9.7, 14.2])
        ref = mu * r.pdf(t)[:, c] * np.exp(-mu * r.cdf(t)[:, c])
        np.testing.assert_allclose(np.exp(L.first_arrival_logpdf(t, 0.0, mu, r.channel(c))), ref, rtol=1e-12)
