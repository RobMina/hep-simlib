"""Backend resolution for the likelihood templates: NumPy + SciPy, or jax.numpy + jax.scipy, from the inputs."""
import numpy as np


def is_jax(*xs):
    return any(type(x).__module__.split(".")[0] in ("jax", "jaxlib") for x in xs)


class Backend:
    """``xp`` (numpy or jax.numpy) and the special functions the templates need, with one signature each."""

    def __init__(self, use_jax):
        self.jax = bool(use_jax)
        if self.jax:
            import jax.numpy as jnp
            import jax.scipy.special as jsp
            self.xp = jnp
            self.log_ndtr, self.ndtr, self.ndtri = jsp.log_ndtr, jsp.ndtr, jsp.ndtri
            self.gammaln, self.logsumexp = jsp.gammaln, jsp.logsumexp
        else:
            try:
                import scipy.special as sp
            except ImportError as exc:     # pragma: no cover - SciPy is in the 'likelihood' extra
                raise ImportError("the NumPy likelihood templates need SciPy (pip install 'hep-simlib[likelihood]')") \
                    from exc
            self.xp = np
            self.log_ndtr, self.ndtr, self.ndtri = sp.log_ndtr, sp.ndtr, sp.ndtri
            self.gammaln, self.logsumexp = sp.gammaln, sp.logsumexp
            self._ndtri_exp = sp.ndtri_exp

    def ndtri_exp(self, y):
        """Phi^-1(exp(y)), accurate in the lower tail (SciPy); the JAX twin clamps exp(y) to the dtype's range,
        which only matters where the probability it serves is below the dtype's tiny."""
        if not self.jax:
            return self._ndtri_exp(y)
        xp = self.xp
        dt = y.dtype
        tiny = xp.finfo(dt).tiny
        return self.ndtri(xp.clip(xp.exp(y), tiny, 1 - xp.finfo(dt).epsneg))

    def where_safe(self, cond, x, safe):
        """x where cond, else a safe value (the double-where trick: no NaN gradient from the unused branch)."""
        return self.xp.where(cond, x, safe)


def backend(*xs):
    return Backend(is_jax(*xs))


def floating(b, *xs):
    """Cast inputs to one floating dtype: float64 for NumPy (host code is float64); for JAX the widest floating dtype
    among the JAX inputs (float32 or float64), else JAX's default float."""
    if not b.jax:
        return [np.asarray(x, np.float64) for x in xs]
    xp = b.xp
    dts = [x.dtype for x in xs if is_jax(x) and xp.issubdtype(x.dtype, xp.floating)]
    dtype = max(dts, key=lambda d: np.dtype(d).itemsize) if dts else xp.zeros(()).dtype
    if not xp.issubdtype(dtype, xp.floating):
        dtype = xp.float32
    return [xp.asarray(x, dtype) for x in xs]
