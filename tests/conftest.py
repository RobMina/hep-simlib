import pytest

from hepsimlib import profile


@pytest.fixture(autouse=True)
def generic_profile():
    """Every test starts and ends with the generic profile."""
    previous = profile.configure(profile.GENERIC)
    yield
    profile.configure(previous)
