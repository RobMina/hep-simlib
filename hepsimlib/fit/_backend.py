"""Array-backend helpers of :mod:`hepsimlib.fit`: JAX is optional and imported only when a JAX path is used.

- :func:`jacobian`: ``jax.jacfwd`` / ``jax.jacrev`` under the highest matmul precision, or central finite
  differences in NumPy (no JAX needed).
- :func:`highest_precision`: a context in which JAX matrix products run at full float32 (or float64) precision,
  never TF32 or bfloat16 passes. Gauss-Newton gradients and Fisher matrices change by factors of a few under TF32.
- :func:`normal_equations`: ``J^T W J`` and ``J^T W r`` in the array's own backend at the highest precision,
  returned as float64 NumPy.
"""
import contextlib
import sys

import numpy as np


def have_jax():
    try:
        import jax  # noqa: F401
    except ImportError:
        return False
    return True


def is_jax(x):
    return type(x).__module__.split(".")[0] in ("jax", "jaxlib")


def host64(x):
    """Float64 NumPy copy of an array of either backend."""
    return np.asarray(x, dtype=np.float64)


def x64_enabled():
    import jax
    return bool(jax.config.read("jax_enable_x64"))


@contextlib.contextmanager
def highest_precision():
    """Run JAX matmuls (and the matmuls inside ``jacfwd``) at the highest precision; a no-op without JAX."""
    if "jax" not in sys.modules:
        yield
        return
    import jax
    ctx = getattr(jax, "default_matmul_precision", None)
    if ctx is None:   # pragma: no cover - older or newer JAX without the context manager
        yield
        return
    with ctx("highest"):
        yield


def check_dtype(dtype, method):
    """The model dtype as np.dtype (float32 or float64); a float64 JAX path needs jax_enable_x64."""
    d = np.dtype(np.float64 if dtype is None else dtype)
    if d.name not in ("float32", "float64"):
        raise ValueError(f"model dtype must be float32 or float64, not {d.name}")
    if method in ("fwd", "rev") and d == np.float64 and not x64_enabled():
        raise ValueError("a float64 JAX Jacobian needs jax_enable_x64 (jax.config.update('jax_enable_x64', True))")
    return d


def fd_step(theta, dtype):
    """Central-difference step per parameter: eps**(1/3) * max(1, |theta|) in the evaluation dtype."""
    eps = np.finfo(np.dtype(dtype)).eps
    return eps ** (1 / 3) * np.maximum(1.0, np.abs(np.asarray(theta, np.float64)))


def jacobian(fn, theta, method="fwd", *, dtype=None, step=None):
    """Jacobian ``d fn / d theta`` of a vector function, shape ``out.shape + (p,)``.

    ``method``: ``"fwd"`` (``jax.jacfwd``: few parameters, many outputs), ``"rev"`` (``jax.jacrev``), ``"fd"``
    (central differences, NumPy, float64 result; ``step`` per parameter, default :func:`fd_step`) or a callable
    ``method(theta)`` returning the Jacobian. ``theta`` is cast to ``dtype`` (default float64) before ``fn`` is
    called. JAX results stay JAX arrays; FD results are float64 NumPy.
    """
    if callable(method):
        return method(theta)
    d = check_dtype(dtype, method)
    if method in ("fwd", "rev"):
        import jax
        import jax.numpy as jnp
        jac = jax.jacfwd if method == "fwd" else jax.jacrev
        with highest_precision():
            return jac(fn)(jnp.asarray(np.asarray(theta, np.float64), d))
    if method != "fd":
        raise ValueError(f"unknown Jacobian method {method!r}: 'fwd', 'rev', 'fd' or a callable")
    theta = np.asarray(theta, np.float64)
    h = fd_step(theta, d) if step is None else np.broadcast_to(np.asarray(step, np.float64), theta.shape)
    cols = []
    for j in range(theta.size):
        e = np.zeros_like(theta)
        e[j] = h[j]
        up = host64(fn((theta + e).astype(d)))
        dn = host64(fn((theta - e).astype(d)))
        cols.append((up - dn) / (2 * h[j]))
    return np.stack(cols, axis=-1)


def normal_equations(J, r=None, w=None):
    """``(J^T W J, J^T W r)`` as float64 NumPy; ``W = diag(w)`` (None: identity). JAX arrays are reduced on their
    device at the highest precision, NumPy arrays in float64."""
    if is_jax(J):
        import jax
        import jax.numpy as jnp
        hp = jax.lax.Precision.HIGHEST
        Jw = J if w is None else J * w[:, None]
        H = jnp.matmul(Jw.T, J, precision=hp)
        g = None if r is None else jnp.matmul(Jw.T, r, precision=hp)
        return host64(H), (None if g is None else host64(g))
    J = host64(J)
    Jw = J if w is None else J * host64(w)[:, None]
    return Jw.T @ J, (None if r is None else Jw.T @ host64(r))
