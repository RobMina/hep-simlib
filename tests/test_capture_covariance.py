"""Placement covariance (hepsimlib.capture.covariance): replaying a capture's own bank in the capture's placement must
give the capture's own global outcomes (one trial per state, so the replay is deterministic), and identical outcome sets
compare with p = 1."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from hepsimlib import component as cr
from hepsimlib.capture import bank, covariance
from test_capture_audit import tables  # noqa: F401 -- the shared synthetic capture fixture
from test_capture_bank import bankable, write
from test_capture_recorder_format import shrink

SPEC = {"edep_log10": [-3, 4, 35], "exits": [-0.5, 10.5, 11], "hit_volume": [-0.5, 9.5, 10],
        "exit_species": [-0.5, 2.5, 3], "exit_kinetic_log10": [-4, 4, 32], "exit_time_log10": [-3, 3, 24],
        "exit_x": [-240, 240, 48], "exit_ux": [-1, 1, 20]}


def _bank(tmp_path, tables):
    tables, links = bankable(tables)
    shrink(tables, links)
    path = write(tmp_path / "capture.root", tables, links)
    bank.convert([path], tmp_path / "bank.npz", 2, relative_to=tmp_path)
    return tables, cr.load_library(tmp_path / "bank.npz")


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_replay_in_the_capture_placement_reproduces_its_outcomes(tmp_path, tables, dtype):
    tables, library = _bank(tmp_path, tables)
    g4 = covariance.from_capture(tables, threshold=0.1)
    rp = covariance.from_replay(library, tables, np.random.default_rng(1), dtype=dtype, threshold=0.1)
    tol = 1e-9 if dtype == np.float64 else 1e-4
    assert list(g4["trials"]["cls"]) == list(rp["trials"]["cls"])
    np.testing.assert_allclose(rp["trials"]["edep"], g4["trials"]["edep"], rtol=1e-6, atol=tol)
    np.testing.assert_array_equal(rp["trials"]["exits"], g4["trials"]["exits"])
    np.testing.assert_array_equal(rp["hits"], g4["hits"])
    order_g, order_r = (np.lexsort((o["exits"]["kinetic"], o["exits"]["trial"])) for o in (g4, rp))
    for key in ("trial", "pdg"):
        np.testing.assert_array_equal(rp["exits"][key][order_r], g4["exits"][key][order_g])
    for key in ("kinetic", "position", "momentum", "time"):
        np.testing.assert_allclose(rp["exits"][key][order_r], g4["exits"][key][order_g], rtol=1e-6, atol=tol)


def test_identical_outcomes_compare_with_p_one(tmp_path, tables):
    tables, library = _bank(tmp_path, tables)
    g4 = covariance.from_capture(tables)
    result = covariance.compare(g4, g4, SPEC, min_entries=1)
    assert result["family"] > 0 and result["bonferroni_min_p"] == 1.0
    assert all(abs(m["ratio"] - 1) < 1e-12 for m in result["mean_edep"].values())


def test_undeclared_observable_is_refused(tmp_path, tables):
    tables, _ = _bank(tmp_path, tables)
    g4 = covariance.from_capture(tables)
    with pytest.raises(ValueError, match="undeclared"):
        covariance.compare(g4, g4, {"edep_mass": [0, 1, 2]})


def test_a_shifted_placement_is_detected(tmp_path, tables):
    """Exits displaced by a wrong translation fail the exit-position shape test."""
    tables, library = _bank(tmp_path, tables)
    g4 = covariance.from_capture(tables)
    n = 400
    big = {"trials": {k: np.tile(v, n) for k, v in g4["trials"].items()}, "hits": np.tile(g4["hits"], (n, 1)),
           "nvol": g4["nvol"]}
    rep = len(g4["trials"]["edep"])
    big["exits"] = {k: (np.concatenate([g4["exits"][k] + i * rep for i in range(n)]) if k == "trial" else
                        np.concatenate([g4["exits"][k]] * n)) for k in g4["exits"]}
    rng = np.random.default_rng(3)
    big["exits"]["position"] = big["exits"]["position"] + rng.normal(0, 20, big["exits"]["position"].shape)
    moved = {**big, "exits": {**big["exits"], "position": big["exits"]["position"] + [60., 0, 0]}}
    result = covariance.compare(big, moved, {"exit_x": [-240, 240, 48]}, min_entries=50)
    assert result["bonferroni_min_p"] < 1e-6
