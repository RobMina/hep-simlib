"""Experiment profiles: what an experiment can set, the checks on it, and its effect on every module that reads it."""
from pathlib import Path

import numpy as np
import pytest

from hepsimlib import component as CR, profile
from hepsimlib.library import dtypes, manifest, registry, store

from test_component_response import bank
from test_library import _arrays, _fields

EXP = profile.Profile(name="exp", subsystems=("core", "calo", "secondaries"), dtype_env="EXP_FLOAT",
                      default_dtype="float32", cache_env="EXP_LIBRARY_CACHE", cache_dir="~/.cache/exp/libraries",
                      release_url="https://example.org/exp/releases/download/{tag}/{file}",
                      bank_subsystem="secondaries", bank_prefix="secondaries.component.",
                      bank_reasons=dict(profile.BANK_REASONS, generator="hashes not recorded yet (exp phase 2)"),
                      bank_validation_reason="exp: not admissible until the accounting closes")


def test_generic_profile_allows_any_identifier():
    assert profile.current() is profile.GENERIC and manifest.SUBSYSTEMS is None
    for good in ("core", "optics", "muon_beam2"):
        assert profile.GENERIC.allows(good)
    for bad in ("Optics", "2d", "a-b", "", None):
        assert not profile.GENERIC.allows(bad)
    store.finalize(_arrays(), _fields(name="optics.test.toy", subsystem="optics"))


def test_profile_subsystems_are_enforced():
    with profile.using(EXP):
        assert manifest.SUBSYSTEMS == ("core", "calo", "secondaries")
        store.finalize(_arrays(), _fields())
        with pytest.raises(manifest.ManifestError, match="is not one of"):
            store.finalize(_arrays(), _fields(name="optics.test.toy", subsystem="optics"))
    assert profile.current() is profile.GENERIC


def test_profile_dtype_and_cache(monkeypatch):
    monkeypatch.delenv("EXP_FLOAT", raising=False)
    monkeypatch.delenv("EXP_LIBRARY_CACHE", raising=False)
    monkeypatch.delenv("HEPSIMLIB_LIBRARY_CACHE", raising=False)
    with profile.using(EXP):
        assert dtypes.compute_dtype() == np.float32
        monkeypatch.setenv("EXP_FLOAT", "float64")
        assert dtypes.compute_dtype() == np.float64
        assert registry.CACHE_ENV == "EXP_LIBRARY_CACHE"
        assert registry.default_cache() == Path.home() / ".cache" / "exp" / "libraries"
        assert registry.default_cache(subdir="calo") == Path.home() / ".cache" / "exp" / "libraries" / "calo"
        monkeypatch.setenv("EXP_LIBRARY_CACHE", "/tmp/exp-cache")
        assert registry.default_cache(subdir="calo") == Path("/tmp/exp-cache")
        monkeypatch.setenv("CALO_CACHE", "/tmp/calo")
        assert registry.default_cache("CALO_CACHE") == Path("/tmp/calo")
    assert registry.default_cache() == Path.home() / ".cache" / "hepsimlib" / "libraries"


def test_profile_release_url_reaches_registry_entries(tmp_path):
    path = tmp_path / "toy.npz"
    store.save(path, _arrays(), _fields())
    assert "url" not in registry.Registry().entry(path)            # generic: no default URL
    with profile.using(EXP):
        e = registry.Registry().entry(path)
        assert e["url"] == "https://example.org/exp/releases/download/lib-core.test.toy-v1/toy.npz"
        assert registry.RELEASE_URL == EXP.release_url


def test_profile_bank_names_and_texts(tmp_path):
    lib = bank()
    generic = CR.save_library(tmp_path / "generic.npz", lib)
    assert generic["subsystem"] == "component" and generic["name"] == "component.finite-segmented-A"
    assert generic["unrecorded"]["generator"] == profile.BANK_REASONS["generator"]
    with profile.using(EXP):
        m = CR.save_library(tmp_path / "exp.npz", lib)
        assert CR.BANK_PREFIX == "secondaries.component."
        assert m["subsystem"] == "secondaries" and m["name"] == "secondaries.component.finite-segmented-A"
        assert m["unrecorded"]["generator"] == "hashes not recorded yet (exp phase 2)"
        assert m["validation"]["reason"] == "exp: not admissible until the accounting closes"
        assert CR.load_library(tmp_path / "exp.npz").component == lib.component
        with pytest.raises(store.LibraryError):
            CR.load_library(tmp_path / "generic.npz")              # subsystem and prefix of another profile
    assert m["content_hash"] == generic["content_hash"]            # the profile changes names and texts, not arrays


@pytest.mark.parametrize("kw, msg", [
    (dict(subsystems=()), "subsystems"), (dict(subsystems=("Calo",)), "subsystems"),
    (dict(default_dtype="float16"), "default_dtype"), (dict(release_url="https://x/{file}"), "release_url"),
    (dict(bank_subsystem="calo", bank_prefix="calo.component."), "bank_subsystem"),
    (dict(bank_prefix="component"), "bank_prefix"), (dict(bank_reasons={"source": "x"}), "bank_reasons")])
def test_profile_is_checked(kw, msg):
    with pytest.raises(ValueError, match=msg):
        profile.Profile(**{"subsystems": ("core", "component")} | kw if "subsystems" not in kw else kw)


def test_configure_takes_a_profile():
    with pytest.raises(TypeError):
        profile.configure({"name": "exp"})
