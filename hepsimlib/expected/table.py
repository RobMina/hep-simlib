"""Expected response of a table library: steps in, expected counts and arrival-time distributions out.

docs/calibration.md, 2.4. A table library (optical photon tables, a light-yield map, a crystal response map)
tabulates, per cell of a position grid in its own frame and per readout channel:

    yield      [cells..., C]        expected photoelectrons per unit energy deposit, at the cell centre
    delay_cdf  [cells..., C, T + 1] CDF of the arrival delay on the delay-axis edges (0 at the first edge, 1 at
                                    the last), at the cell centre

For one energy-deposit step at position ``x``, energy ``e`` and time ``t_s``, the expected number of
photoelectrons in channel ``c`` that have arrived by time ``t`` is

    M_c(t) = e * sum_corners w_k(x) yield_k,c * F_k,c(t - t_s)

This is a multilinear interpolation, between cell centres, of the time-resolved expected count ``yield * F``. A
sampled chain that picks the corner cell of each photon with probability proportional to ``w * yield`` has
exactly this mean, which the tests check. Summed over steps, with an optional global ``yield_scale`` and per-channel
``channel_scale`` (efficiency or gain): the expected count of channel ``c`` is ``M_c(inf)``, its arrival-time CDF
``M_c(t) / M_c(inf)``, and its expected counts per time bin the differences of ``M_c`` at the bin edges.

- **Deterministic.** No random numbers, so the same inputs give the same bits.
- **Interpolated, not nearest bin.** The grid is multilinear between cell centres, so position gradients exist
  for alignment fits. Within the outermost half cells the end value holds. Beyond the axis edges the axis policy
  applies: ``clip`` (end value), ``nan`` (the step's contribution is NaN) or ``refuse``. ``refuse`` raises
  :class:`hepsimlib.library.OutOfDomain` before any work on the host. A device response carries per-row domain
  codes for the caller to check after the sync.
- **Delays.** Linear in time between the delay edges (a piecewise-constant density), clamped to 0 and 1 outside.
- **Backends.** :meth:`TableResponse.from_library` / :meth:`TableResponse.from_arrays` give the host table
  (NumPy, float64). :meth:`TableResponse.to_device` gives a JAX copy in float32 or float64, with the same
  ``expected`` method; the result is differentiable in step positions, energies and times, ``yield_scale`` and
  ``channel_scale``.
"""
from itertools import product
from typing import NamedTuple

import numpy as np

from ..library import axes as A
from ..library.domain import OutOfDomain

KIND = "interpolation"


class Steps(NamedTuple):
    """Energy-deposit steps in the table's frame: ``position [n, d]`` (axis order of the table), ``edep [n]``,
    ``time [n]``, ``mask [n]`` (False: padding, contributes nothing)."""
    position: object
    edep: object
    time: object
    mask: object = None


def _xp_of(*xs):
    for x in xs:
        if type(x).__module__.split(".")[0] in ("jax", "jaxlib"):
            import jax.numpy as jnp
            return jnp
    return np


class ChannelResponse(NamedTuple):
    """Expected response per channel (a pytree of arrays: it can leave a ``jax.jit``).

    ``cum [n, C, T + 1]``: expected cumulative photoelectrons of each step and channel at the delay edges (scales
    applied); ``times [n]`` step times; ``delay_edges [T + 1]``; ``counts [C]`` expected photoelectrons per channel;
    ``codes [n, d]`` domain codes of the step positions (``hepsimlib.library.axes``: 0 in, 1 below, 2 above,
    3 nonfinite)."""
    cum: object
    times: object
    delay_edges: object
    counts: object
    codes: object

    def cumulative(self, t):
        """Expected photoelectrons per channel arrived by ``t``: ``[..., C]``."""
        xp = _xp_of(self.cum, t)
        t = xp.asarray(t, self.cum.dtype)
        e = self.delay_edges
        T = e.shape[0] - 1
        d = t[..., None] - self.times                                       # [..., n]
        dc = xp.clip(d, e[0], e[-1])
        i = xp.clip(xp.searchsorted(e, dc, side="right") - 1, 0, T - 1)
        f = (dc - e[i]) / (e[i + 1] - e[i])
        cum_t = xp.swapaxes(self.cum, 1, 2)                                 # [n, T + 1, C]
        s = xp.arange(self.times.shape[0])
        lo, hi = cum_t[s, i], cum_t[s, i + 1]                               # [..., n, C]
        return xp.sum(lo + f[..., None] * (hi - lo), axis=-2)

    def density(self, t):
        """Expected photoelectrons per unit time per channel at ``t``: ``[..., C]`` (piecewise constant)."""
        xp = _xp_of(self.cum, t)
        t = xp.asarray(t, self.cum.dtype)
        e = self.delay_edges
        T = e.shape[0] - 1
        d = t[..., None] - self.times
        inside = (d >= e[0]) & (d < e[-1])
        i = xp.clip(xp.searchsorted(e, d, side="right") - 1, 0, T - 1)
        cum_t = xp.swapaxes(self.cum, 1, 2)
        s = xp.arange(self.times.shape[0])
        slope = (cum_t[s, i + 1] - cum_t[s, i]) / (e[i + 1] - e[i])[..., None]
        return xp.sum(xp.where(inside[..., None], slope, 0.0), axis=-2)

    def _norm(self):
        xp = _xp_of(self.counts)
        return xp.where(self.counts > 0, self.counts, 1.0)

    def cdf(self, t):
        """Arrival-time CDF per channel, ``[..., C]`` (0 for a channel with no expected light)."""
        return self.cumulative(t) / self._norm()

    def pdf(self, t):
        """Arrival-time density per channel, ``[..., C]``."""
        return self.density(t) / self._norm()

    def binned(self, edges):
        """Expected photoelectrons per channel and time bin, ``[C, nbins]`` (exact integral of the density)."""
        xp = _xp_of(self.cum, edges)
        m = self.cumulative(xp.asarray(edges, self.cum.dtype))
        return xp.swapaxes(m[1:] - m[:-1], 0, 1)

    def channel(self, c):
        """Channel ``c``'s arrival-time distribution as a delay object (``cdf``, ``pdf`` of absolute time), e.g. for
        :func:`hepsimlib.likelihood.first_arrival_logpdf` with ``t0 = 0`` and ``mu = counts[c]``."""
        return _Channel(self, c)


class _Channel:
    def __init__(self, resp, c):
        self.resp, self.c = resp, c

    def cdf(self, t):
        return self.resp.cdf(t)[..., self.c]

    def pdf(self, t):
        return self.resp.pdf(t)[..., self.c]


class TableResponse:
    """A table library ready for expected-response queries, on the host (NumPy float64) or a device (JAX)."""

    def __init__(self, axes, yield_, delay_edges, delay_cdf, *, name="table", xp=np, dtype=np.float64, _checked=False):
        self.axes = tuple(axes)
        self.name = name
        self.xp, self.dtype = xp, np.dtype(dtype)
        self.shape = tuple(a.nbins for a in self.axes)
        self.policies = tuple(a.policy for a in self.axes)
        if not _checked:
            y = np.asarray(yield_, np.float64)
            F = np.asarray(delay_cdf, np.float64)
            e = np.asarray(delay_edges, np.float64)
            if y.ndim != len(self.shape) + 1 or y.shape[:-1] != self.shape:
                raise ValueError(f"yield must have shape {self.shape} + (C,)")
            C = y.shape[-1]
            if e.ndim != 1 or e.size < 2 or np.any(np.diff(e) <= 0):
                raise ValueError("delay edges must be strictly increasing, at least two")
            if F.shape != self.shape + (C, e.size):
                raise ValueError(f"delay_cdf must have shape {self.shape + (C, e.size)}")
            if not (np.all(np.isfinite(y)) and np.all(y >= 0) and np.all(np.isfinite(F))):
                raise ValueError("yield must be finite and non-negative, delay_cdf finite")
            if np.any(np.diff(F, axis=-1) < 0) or np.any(np.abs(F[..., 0]) > 1e-9) or np.any(np.abs(F[..., -1] - 1)
                                                                                             > 1e-9):
                raise ValueError("delay_cdf must rise from 0 at the first delay edge to 1 at the last")
            yield_, delay_cdf, delay_edges = y.reshape(-1, C), F.reshape(-1, C, e.size), e
        self.n_channels = int(yield_.shape[-1])
        self._yield = yield_                  # [cells, C]
        self._ycdf = None
        self._cdf = delay_cdf                 # [cells, C, T + 1]
        self.delay_edges = delay_edges
        self.centres = [xp.asarray(0.5 * (a.edges[1:] + a.edges[:-1]), self.dtype) for a in self.axes]
        self.edges = [xp.asarray(a.edges, self.dtype) for a in self.axes]

    # ---------------------------------------------------------------------------------------------- construction
    @classmethod
    def from_arrays(cls, axes, yield_, delay_edges, delay_cdf, *, name="table"):
        """Host table from position axes (:class:`hepsimlib.library.Axis`), ``yield [cells..., C]``, delay edges
        ``[T + 1]`` and ``delay_cdf [cells..., C, T + 1]``."""
        return cls(axes, yield_, delay_edges, delay_cdf, name=name)

    @classmethod
    def from_library(cls, lib, *, yield_array="yield", delay_array="delay_cdf", delay_axis="delay",
                     position_axes=None):
        """Host table from a loaded library (:func:`hepsimlib.library.store.load`); ``position_axes``: names in grid
        order (default: every axis of the manifest except ``delay_axis``)."""
        names = position_axes or tuple(d["name"] for d in lib.manifest["axes"] if d["name"] != delay_axis)
        interp = lib.manifest.get("interpolation") or {}
        if interp.get("method") != "linear":
            raise ValueError(f"{lib.name}: an expected-response table declares interpolation method 'linear'")
        return cls([lib.axis(n) for n in names], lib.arrays[yield_array], lib.axis(delay_axis).edges,
                   lib.arrays[delay_array], name=lib.name)

    def to_device(self, dtype="float32"):
        """JAX copy in ``dtype`` (float32 or float64; float64 needs jax_enable_x64)."""
        import jax
        import jax.numpy as jnp
        d = np.dtype(dtype)
        if d.name not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if d == np.float64 and not jax.config.read("jax_enable_x64"):
            raise ValueError("a float64 device table needs jax_enable_x64")
        return TableResponse(self.axes, jnp.asarray(self._yield, d), jnp.asarray(self.delay_edges, d),
                             jnp.asarray(self._cdf, d), name=self.name, xp=jnp, dtype=d, _checked=True)

    # ---------------------------------------------------------------------------------------------- queries
    def domain_codes(self, position):
        """Per-row, per-axis domain codes ``[n, d]`` (host NumPy or device, like the table)."""
        pos = self.xp.asarray(position, self.dtype)
        f = A.domain_codes if self.xp is np else A.domain_codes_jax
        return self.xp.stack([f(self.edges[j], pos[:, j]) for j in range(len(self.axes))], axis=1)

    def _corners(self, pos):
        """Flat corner cells [2^d, n] and weights [2^d, n] of the multilinear interpolation between centres."""
        xp = self.xp
        idx, frac, step = [], [], []
        for j, c in enumerate(self.centres):
            m = c.shape[0]
            v = pos[:, j]
            if m == 1:
                idx.append(xp.zeros(v.shape, np.int32 if xp is not np else np.int64))
                frac.append(xp.zeros_like(v))
                step.append(0)
                continue
            u = xp.clip(v, c[0], c[-1])
            i = xp.clip(xp.searchsorted(c, u, side="right") - 1, 0, m - 2)
            idx.append(i)
            frac.append((u - c[i]) / (c[i + 1] - c[i]))
            step.append(1)
        cells, weights = [], []
        for bits in product((0, 1), repeat=len(self.axes)):
            cell = idx[0] + bits[0] * step[0]
            w = frac[0] if bits[0] else 1 - frac[0]
            for j in range(1, len(self.axes)):
                cell = cell * self.shape[j] + idx[j] + bits[j] * step[j]
                w = w * (frac[j] if bits[j] else 1 - frac[j])
            cells.append(cell)
            weights.append(w)
        return xp.stack(cells), xp.stack(weights)

    def expected(self, steps, *, yield_scale=1.0, channel_scale=None):
        """:class:`ChannelResponse` of the steps (module docstring)."""
        xp = self.xp
        pos = xp.asarray(steps.position, self.dtype)
        if pos.ndim != 2 or pos.shape[1] != len(self.axes):
            raise ValueError(f"positions must be [n, {len(self.axes)}] in the axis order {[a.name for a in self.axes]}")
        edep = xp.asarray(steps.edep, self.dtype)
        times = xp.asarray(steps.time, self.dtype)
        n = pos.shape[0]
        mask = xp.ones(n, bool) if steps.mask is None else xp.asarray(steps.mask, bool)
        codes = self.domain_codes(pos)
        bad_nan = xp.zeros(n, bool)
        for j, pol in enumerate(self.policies):
            out = codes[:, j] != A.IN
            if pol == "refuse" and xp is np:
                rows = np.flatnonzero(out & np.asarray(mask))
                if rows.size:
                    raise OutOfDomain(self.name, self.axes[j].name, rows, codes[rows, j])
            if pol == "nan":
                bad_nan = bad_nan | out
        safe = xp.stack([c[0] * xp.ones(n, self.dtype) for c in self.centres], axis=1)
        pos_s = xp.where(mask[:, None] & xp.isfinite(pos), pos, safe)
        cells, w = self._corners(pos_s)                                     # [K, n]
        wy = w[..., None] * self._yield[cells]                              # [K, n, C]
        cum = xp.einsum("knc,knct->nct", wy, self._cdf[cells])              # [n, C, T + 1]
        scale = edep * xp.asarray(yield_scale, self.dtype)
        rowscale = xp.where(mask, scale, 0.0)
        rowscale = xp.where(bad_nan & mask, xp.nan, rowscale)
        rowscale = xp.where(mask & ~xp.all(xp.isfinite(pos), axis=1), xp.nan, rowscale)
        cum = cum * rowscale[:, None, None]
        if channel_scale is not None:
            cum = cum * xp.asarray(channel_scale, self.dtype)[None, :, None]
        counts = xp.sum(cum[..., -1], axis=0)
        return ChannelResponse(cum, xp.where(mask, times, 0.0), self.delay_edges, counts, codes)

    def interpolate_yield(self, position):
        """Interpolated expected photoelectrons per unit energy ``[n, C]`` at the positions (clip semantics)."""
        pos = self.xp.asarray(position, self.dtype)
        cells, w = self._corners(pos)
        return self.xp.einsum("kn,knc->nc", w, self._yield[cells])


def table_fields(axes, yield_, delay_edges, delay_cdf, *, delay_axis="delay", delay_unit="ns",
                 count_unit="photoelectrons / MeV"):
    """The arrays and the manifest fields that describe an expected-response table (kind ``interpolation``, the
    axes, the outputs, linear interpolation, both compute dtypes, no random numbers). The caller adds the
    provenance fields (name, subsystem, version, description, source, generator, release, seeds, geometry, split,
    domain, validation) and writes it with :func:`hepsimlib.library.store.save`."""
    delay = A.Axis(delay_axis, delay_unit, delay_edges, frame="arrival delay after the deposit", array="delay_edges",
                   policy="clip", description="delay-CDF nodes")
    arrays = {a.array: a.edges for a in axes}
    arrays.update(delay_edges=delay.edges, **{"yield": np.asarray(yield_, np.float64),
                                              "delay_cdf": np.asarray(delay_cdf, np.float64)})
    fields = dict(kind=KIND, axes=[a.describe() for a in axes] + [delay.describe()],
                  outputs=[dict(name="yield", unit=count_unit, dtype="float",
                                description="expected photoelectrons per unit energy deposit per channel at the cell "
                                            "centre; null answer: 0 for padded steps"),
                           dict(name="delay_cdf", unit="1", dtype="float",
                                description="arrival-delay CDF per channel at the delay edges")],
                  interpolation=dict(method="linear", axes=[a.name for a in axes]),
                  dtype=dict(storage={"yield": "float64", "delay_cdf": "float64"}, compute=["float32", "float64"]),
                  rng=dict(host="none", device="none", draws_per_row=0),
                  assumptions=[dict(name="end half cells", description="the interpolation holds the end value within "
                                    "the outermost half cell of each axis", tested=True)])
    return arrays, fields
