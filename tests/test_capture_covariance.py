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


def _only(group, run):
    """The rows of one event (run number after shrink) in every table of a group."""
    return {name: {f: np.asarray(v)[np.asarray(table["run"]) == run] for f, v in table.items()}
            for name, table in group.items()}


def _bank(tmp_path, tables, *, single=False):
    """The fixture's bank. Its two primaries share one local entrance state, so the replay draws between their two
    trials; ``single`` keeps only event 2 (rotated and translated placement, a stopping child and an exit at birth),
    whose one trial the replay must reproduce exactly."""
    tables, links = bankable(tables)
    shrink(tables, links)
    events = 2
    if single:
        tables, links, events = _only(tables, 2), _only(links, 2), 1
    path = write(tmp_path / "capture.root", tables, links)
    bank.convert([path], tmp_path / "bank.npz", events, relative_to=tmp_path)
    return tables, cr.load_library(tmp_path / "bank.npz")


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_replay_in_the_capture_placement_reproduces_its_outcomes(tmp_path, tables, dtype):
    tables, library = _bank(tmp_path, tables, single=True)
    assert library.exit_offsets.tolist() == [0, 1] and len(library.steps["edep"]) == 2   # the approach step is outside
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


def _showers(rng, n, shift=0.0):
    """Synthetic outcomes: ~5 exits per trial whose positions share a per-trial offset (strong within-trial
    correlation, as a shower's exits), one class."""
    k = rng.poisson(5, n) + 1
    trial = np.repeat(np.arange(n), k)
    common = rng.normal(0, 50, n)[trial]
    x = common + rng.normal(0, 2, len(trial)) + shift
    pos = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    mom = np.tile([1.0, 0, 0], (len(trial), 1))
    return dict(trials=dict(cls=np.full(n, "c", object), edep=rng.exponential(5, n), exits=k,
                            cluster=np.full(n, -1, np.int64)),
                hits=np.zeros((n, 1), bool), nvol=1,
                exits=dict(trial=trial, pdg=np.full(len(trial), 22), kinetic=np.ones(len(trial)), position=pos,
                           momentum=mom, time=np.ones(len(trial))))


def _resample(bank, rng, n):
    """A replay-like sample: n draws with replacement from the bank's trials, clustered on the drawn trial."""
    pick = rng.integers(0, len(bank["trials"]["edep"]), n)
    rows = [np.flatnonzero(bank["exits"]["trial"] == p) for p in pick]
    idx = np.concatenate(rows)
    out = dict(trials={k: v[pick] for k, v in bank["trials"].items()}, hits=bank["hits"][pick], nvol=1)
    out["trials"]["cluster"] = pick.astype(np.int64)
    out["exits"] = {k: v[idx] for k, v in bank["exits"].items()}
    out["exits"]["trial"] = np.repeat(np.arange(n), [len(r) for r in rows])
    return out


SHOWER_SPEC = {"exits": [-0.5, 14.5, 15], "exit_x": [-245, 235, 48]}


def test_robust_comparison_is_calibrated_for_correlated_exits():
    rng = np.random.default_rng(11)
    a, b = _showers(rng, 2000), _showers(rng, 2000)
    old = covariance.compare(a, b, SHOWER_SPEC, min_entries=50)
    new = covariance.compare_robust(a, b, SHOWER_SPEC)
    assert old["bonferroni_min_p"] < 1e-3        # entries counted as independent: a false failure
    assert new["bonferroni_min_p"] > 1e-3 and new["family"] == 2


def test_robust_comparison_is_calibrated_for_a_resampled_bank():
    rng = np.random.default_rng(12)
    bank, reference = _showers(rng, 1000), _showers(rng, 2000)
    replay = _resample(bank, rng, 2000)
    result = covariance.compare_robust(reference, replay, SHOWER_SPEC)
    assert result["bonferroni_min_p"] > 1e-3
    m = result["mean_edep"]["c"]
    assert abs(m["ratio"] - 1) < 4 * m["err"]


def test_robust_comparison_detects_a_shift():
    rng = np.random.default_rng(13)
    a, b = _showers(rng, 2000), _showers(rng, 2000, shift=15.0)
    assert covariance.compare_robust(a, b, SHOWER_SPEC)["bonferroni_min_p"] < 1e-6
