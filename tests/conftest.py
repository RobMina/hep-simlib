import pytest

from hepsimlib import profile


@pytest.fixture(autouse=True)
def generic_profile():
    """Every test starts and ends with the generic profile."""
    previous = profile.configure(profile.GENERIC)
    yield
    profile.configure(previous)


@pytest.fixture
def x64():
    """JAX with float64 enabled for the test (restored afterwards); skips without JAX."""
    jax = pytest.importorskip("jax")
    previous = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield jax
    jax.config.update("jax_enable_x64", previous)
