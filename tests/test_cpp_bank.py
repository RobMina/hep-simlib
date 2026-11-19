"""The C++ bank host against the Python one (docs/geant4-host.md): hepsimlib-bank accepts exactly the banks Python
accepts and recomputes the same content hash; hepsimlib-replay reproduces replay() row for row. Skipped unless the
binaries are built (HEPSIMLIB_CPP_BIN, a build directory, or on PATH)."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hepsimlib import component as CR
from hepsimlib.library import store
from test_component_replay_dtype import PLACEMENTS, _placement
from test_component_response import FixedRandom, bank, incoming


def _binary(name):
    d = os.environ.get("HEPSIMLIB_CPP_BIN")
    if d and (Path(d) / name).exists():
        return str(Path(d) / name)
    return shutil.which(name)


BANK, REPLAY = _binary("hepsimlib-bank"), _binary("hepsimlib-replay")
pytestmark = pytest.mark.skipif(not (BANK and REPLAY), reason="C++ tools not built (set HEPSIMLIB_CPP_BIN)")


class Draws:
    """rng.random(n) returning given uniforms."""

    def __init__(self, u):
        self.u = np.asarray(u, float)

    def random(self, n):
        assert n == len(self.u)
        return self.u.copy()


def _save(tmp_path, name="b.npz", **kw):
    path = tmp_path / name
    CR.save_library(path, bank(), name=CR.bank_name(bank().component) + ".cpp-test", **kw)
    return path


def _check(path):
    return subprocess.run([BANK, str(path)], capture_output=True, text=True)


def test_reader_accepts_a_bank_and_recomputes_its_content_hash(tmp_path):
    for compressed in (True, False):
        path = tmp_path / f"b{int(compressed)}.npz"
        m = store.save(path, CR.bank_arrays(bank()), CR.bank_manifest_fields(bank(), name=CR.bank_name(bank().component) + ".x"),
                       compressed=compressed)
        r = _check(path)
        assert r.returncode == 0, r.stderr
        info = json.loads(r.stdout)
        assert info["content_hash"] == m["content_hash"] == store.load(path).content_hash
        assert info["trials"] == 2 and info["species"] == [22] and info["volumes"] == 2


def _rewrite(src, dst, change=None, drop=None, manifest=None):
    with np.load(src, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    if change:
        k, fn = change
        arrays[k] = fn(arrays[k].copy())
    if drop:
        del arrays[drop]
    if manifest:
        m = json.loads(str(arrays[store.MANIFEST_KEY]))
        manifest(m)
        arrays[store.MANIFEST_KEY] = np.str_(json.dumps(m, sort_keys=True, separators=(",", ":")))
    np.savez_compressed(dst, **arrays)


def test_reader_refuses_changed_missing_or_mislabelled_banks(tmp_path):
    good = _save(tmp_path)

    def bump(a):
        a[0] += 1e-6
        return a
    cases = {
        "changed.npz": (dict(change=("steps_edep", bump)), "content hash"),
        "missing.npz": (dict(drop="exits_time"), "missing array"),
        "kind.npz": (dict(manifest=lambda m: m.update(kind="table")), "not a component bank"),
        "nomanifest.npz": (dict(drop=store.MANIFEST_KEY), "no embedded __manifest__"),
    }
    for name, (how, why) in cases.items():
        _rewrite(good, tmp_path / name, **how)
        r = _check(tmp_path / name)
        assert r.returncode == 1 and why in r.stderr, (name, r.stderr)


def test_reader_runs_the_structural_checks(tmp_path):
    """A manifest consistent with its arrays but a structurally invalid bank: refused by both hosts."""
    arrays = dict(CR.bank_arrays(bank()))
    arrays["steps_edep"] = arrays["steps_edep"].copy()
    arrays["steps_edep"][1] = -1.0
    path = tmp_path / "bad.npz"
    store.save(path, arrays, CR.bank_manifest_fields(bank(), name=CR.bank_name(bank().component) + ".bad"))
    with pytest.raises(ValueError):
        CR.load_library(path)
    r = _check(path)
    assert r.returncode == 1 and "negative deposits" in r.stderr


def _replay_cpp(tmp_path, library_path, p, inc, interaction, uniforms, next_track):
    inp, out = tmp_path / "in.npz", tmp_path / "out"
    out.mkdir(exist_ok=True)
    np.savez(inp, rotation=p.rotation, translation=p.translation, volume_ids=p.volume_ids.astype(np.int64),
             instance=np.int64(p.instance), interaction=np.asarray(interaction, np.int64),
             uniforms=np.asarray(uniforms, float), next_track=np.int64(next_track),
             **{k: (np.asarray(v, np.int64) if k in ("event", "track", "parent", "pdg") else np.asarray(v, float))
                for k, v in inc.items()})
    r = subprocess.run([REPLAY, str(library_path), str(inp), str(out)], capture_output=True, text=True)
    return r, {f.stem: np.load(f) for f in out.glob("*.npy")}


@pytest.mark.parametrize("angles,translation", PLACEMENTS)
def test_replay_matches_python_row_for_row(tmp_path, angles, translation):
    path = _save(tmp_path)
    lib = CR.load_library(path)
    p = _placement(angles, translation)
    n = 300
    inc = incoming(p, n=n)
    inc["time"] = inc["time"] * 37.25 + 1500.
    u = np.random.default_rng(11).random(n)
    py = CR.replay(lib, p, inc, np.arange(n), Draws(u), next_track=10_000)
    r, cpp = _replay_cpp(tmp_path, path, p, inc, np.arange(n), u, 10_000)
    assert r.returncode == 0, r.stderr
    assert np.array_equal(cpp["trials"], py.trials) and int(cpp["next_track"][0]) == py.next_track
    for table, ref in (("particles", py.particles), ("steps", py.steps), ("exits", py.exits)):
        for k, v in ref.items():
            got = cpp[f"{table}_{k}"]
            v = np.asarray(v)
            if v.dtype.kind in "iub":
                assert np.array_equal(got.astype(v.dtype), v), (table, k)
            else:
                assert got.shape == v.shape and np.allclose(got, v, rtol=0, atol=1e-9), (table, k)


def test_replay_refuses_the_same_rows(tmp_path):
    path = _save(tmp_path)
    lib = CR.load_library(path)
    p = _placement()
    inc = incoming(p, n=5)
    inc["energy"][[1, 3]] = 11.0                 # no recorded state at this energy
    inc["momentum"][[1, 3]] *= 1.1
    with pytest.raises(CR.UnsupportedEntrance) as e:
        CR.replay(lib, p, inc, np.arange(5), FixedRandom(0.5), next_track=100)
    r, cpp = _replay_cpp(tmp_path, path, p, inc, np.arange(5), np.full(5, 0.5), 100)
    assert r.returncode == 3 and cpp["unsupported"].tolist() == e.value.rows.tolist()
