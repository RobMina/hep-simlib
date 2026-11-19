"""A4 analysis: the exact checks D1-D5 and the statistical comparisons of the declared criteria (README.md).

    python validation/a4/analyze.py DIR --output record.json [--criteria validation/a4/criteria.json]
                                        [--bank-tool PATH/hepsimlib-bank]

DIR holds the hybrid outputs, bank.npz and shards.json (campaign.py). The record carries the criteria file's sha256.
Needs uproot and scipy.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import uproot

from hepsimlib import component as CR
from hepsimlib.capture import covariance
from hepsimlib.library import store

HERE = Path(__file__).resolve().parent
import campaign as CAMPAIGN  # noqa: E402  (validation/a4/campaign.py, on sys.path as the script's directory)

TABLES = ("events", "deposits", "exits", "trials", "runinfo")


def read(directory, rows):
    """{arm: {table: {column: array}}} with the shards of each arm concatenated in shard order."""
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
    out = {}
    for arm, shards in by_arm.items():
        parts = defaultdict(lambda: defaultdict(list))
        for r in sorted(shards, key=lambda r: r["shard"]):
            with uproot.open(Path(directory) / r["output"]) as f:
                for t in TABLES:
                    tree = f[f"hybrid/{t}"]
                    for k, v in tree.arrays(library="np").items():
                        if v.dtype.kind == "O" and t != "runinfo":
                            v = np.stack(v)
                        if k in ("event",) and t != "runinfo":
                            v = v.astype(np.int64)
                        parts[t][k].append((r["shard"], v))
        out[arm] = {t: {k: (np.concatenate([v for _, v in vs]) if t != "runinfo" else [v for _, v in vs])
                        for k, vs in cols.items()} for t, cols in parts.items()}
        out[arm]["shard_of_event"] = np.concatenate(
            [np.full(int(r["events"]), r["shard"]) for r in sorted(shards, key=lambda r: r["shard"])])
    return out


def _table(run, t, mask):
    return {k: v[mask] for k, v in run[t].items()}


# ------------------------------------------------------------------ exact checks
def d1_reader(directory, bank_tool, runs):
    path = Path(directory) / "bank.npz"
    py = store.load(path).content_hash
    res = dict(python_content_hash=py)
    if bank_tool:
        r = subprocess.run([bank_tool, str(path)], capture_output=True, text=True)
        res.update(returncode=r.returncode, cpp=json.loads(r.stdout) if r.returncode == 0 else r.stderr)
        res["cpp_content_hash"] = res["cpp"]["content_hash"] if r.returncode == 0 else None
    hashes = {str(h) for run in runs.values() for shard in run["runinfo"]["bankContentHash"]
              for h in np.atleast_1d(shard) if str(h)}
    res["hybrid_content_hashes"] = sorted(hashes)
    res["passed"] = bool((not bank_tool or res.get("cpp_content_hash") == py) and hashes == {py})
    return res


def d2_parity(directory, campaign, runs, arm, atol):
    lib = CR.load_library(Path(directory) / "bank.npz")
    rot, tr = CAMPAIGN.placement(campaign, campaign["arms"][arm]["placement"])
    p = CR.Placement(lib.component, 0, rot, np.asarray(tr), np.arange(lib.component.volumes))
    run = runs[arm]
    t, d, x = run["trials"], run["deposits"], run["exits"]
    shard = run["shard_of_event"][0]  # detail is written by shard 0, whose rows come first
    problems, checked, worst = [], 0, 0.0
    for i in np.flatnonzero(t["track"] == 1):
        ev, trial, u = int(t["event"][i]), int(t["trial"][i]), float(t["uniform"][i])
        gpos = np.array([[t["x"][i], t["y"][i], t["z"][i]]])
        gmom = np.array([[t["px"][i], t["py"][i], t["pz"][i]]])
        inc = dict(event=np.array([[1, int(shard), ev]]), track=np.array([1]), parent=np.array([-1]),
                   pdg=np.array([int(t["pdg"][i])]), position=gpos, momentum=gmom,
                   energy=np.array([t["energy"][i]]), time=np.array([t["time"][i]]))
        local = {**inc, "position": p.local_points(gpos), "momentum": p.local_vectors(gmom)}
        left, right = CR._matches(lib, local)
        if len(right) != int(t["matches"][i]):
            problems.append(dict(event=ev, problem="match count", python=len(right), cpp=int(t["matches"][i])))
            continue

        class Draw:
            def random(self, n):
                return np.full(n, u)
        r = CR.replay(lib, p, inc, np.array([0]), Draw(), next_track=2)
        if int(r.trials[0]) != trial:
            problems.append(dict(event=ev, problem="trial", python=int(r.trials[0]), cpp=trial))
            continue
        dm = (d["event"] == ev) & (d["trial"] == trial)
        xm = (x["event"] == ev) & (x["trial"] == trial)
        if dm.sum() != len(r.steps["edep"]) or xm.sum() != len(r.exits["energy"]):
            problems.append(dict(event=ev, problem="row count", python=[len(r.steps["edep"]), len(r.exits["energy"])],
                                 cpp=[int(dm.sum()), int(xm.sum())]))
            continue
        if not (np.array_equal(d["pdg"][dm], r.steps["pdg"]) and np.array_equal(d["volume"][dm], r.steps["volume"]) and
                np.array_equal(x["pdg"][xm], r.exits["pdg"])):
            problems.append(dict(event=ev, problem="integers"))
            continue
        reals = [(np.column_stack([d["x0"][dm], d["y0"][dm], d["z0"][dm]]), r.steps["x0"]),
                 (np.column_stack([d["x1"][dm], d["y1"][dm], d["z1"][dm]]), r.steps["x1"]),
                 (np.column_stack([d["px0"][dm], d["py0"][dm], d["pz0"][dm]]), r.steps["p0"]),
                 (np.column_stack([d["px1"][dm], d["py1"][dm], d["pz1"][dm]]), r.steps["p1"]),
                 (d["t0"][dm], r.steps["t0"]), (d["t1"][dm], r.steps["t1"]), (d["length"][dm], r.steps["length"]),
                 (d["edep"][dm], r.steps["edep"]),
                 (np.column_stack([x["x"][xm], x["y"][xm], x["z"][xm]]), r.exits["position"]),
                 (np.column_stack([x["px"][xm], x["py"][xm], x["pz"][xm]]), r.exits["momentum"]),
                 (x["energy"][xm], r.exits["energy"]), (x["time"][xm], r.exits["time"])]
        diff = max((float(np.max(np.abs(a - b))) if np.size(a) else 0.0) for a, b in reals)
        worst = max(worst, diff)
        if diff > atol:
            problems.append(dict(event=ev, problem="reals", max_abs_diff=diff))
        checked += 1
    return dict(arm=arm, events_checked=checked, max_abs_diff=worst, problems=problems[:20],
                n_problems=len(problems), passed=bool(checked > 0 and not problems))


def _rows_equal(a, b, mask_a, mask_b, skip=("cpu_s",)):
    bad = [k for k in a if k not in skip and not np.array_equal(a[k][mask_a], b[k][mask_b])]
    return bad


def d3_refusal(runs, n_bank_states):
    hy, g4 = runs["HYD"]["events"], runs["G4D"]["events"]
    bank_rows = hy["entrance"] < n_bank_states
    ok_trigger = bool(np.all(hy["primary_triggers"][bank_rows] == 1) and np.all(hy["triggers"][~bank_rows] == 0))
    order_ok = np.array_equal(hy["event"], g4["event"])
    # the model's own lookup counters are bookkeeping, not output of the event (README, amendment before any output)
    bad = _rows_equal(hy, g4, ~bank_rows, ~bank_rows, skip=("cpu_s", "lookups", "refused")) if order_ok else ["event order"]
    return dict(bank_events=int(bank_rows.sum()), refused_events=int((~bank_rows).sum()),
                triggers_as_declared=ok_trigger, differing_columns=bad, passed=bool(ok_trigger and not bad))


def d4_plumbing(runs):
    a, b = runs["G4D"]["events"], runs["PUD"]["events"]
    if len(a["event"]) != len(b["event"]):
        return dict(passed=False, differing_columns=["row count"])
    all_rows = np.ones(len(a["event"]), bool)
    bad = _rows_equal(a, b, all_rows, all_rows)
    return dict(events=int(len(a["event"])), differing_columns=bad, passed=not bad)


def d5_coverage(runs, arm):
    e = runs[arm]["events"]
    once = e["primary_triggers"] == 1
    return dict(arm=arm, events=int(len(once)), not_once=int((~once).sum()),
                reentrant_exits=int(e["reentrant"].sum()), refused_lookups=int(e["refused"].sum()),
                passed=bool(once.all()))


# ------------------------------------------------------------------ statistical comparison
def outcomes(run, n_bank_states, catchers, bank):
    e = run["events"]
    keep = e["entrance"] < n_bank_states
    ev = {k: v[keep] for k, v in e.items()}
    trials = dict(cls=covariance._classes(ev["pdg"], ev["kinetic"]),
                  cluster=(ev["trial"] if bank else np.full(len(ev["pdg"]), -1)).astype(np.int64),
                  module_edep=ev["module_edep_g4"] + ev["model_edep"])
    trials["edep"] = trials["module_edep"]
    for c in catchers:
        trials[f"{c}_edep"] = ev[f"{c}_edep"]
    return dict(trials=trials, events=ev)


def entries(o, spec, zero):
    ev, n = o["events"], len(o["events"]["pdg"])
    idx = np.arange(n)
    values = {"module_edep_log10": (np.log10(np.maximum(o["trials"]["module_edep"], zero)), idx)}
    for c in ("floor", "wall"):
        hit = ev[f"{c}_edep"] > 0
        values[f"{c}_edep_log10"] = (np.log10(np.maximum(ev[f"{c}_edep"], zero)), idx)
        values[f"{c}_enter"] = (ev[f"{c}_enter"].astype(float), idx)
        for a in "xyz":
            values[f"{c}_{a}"] = (ev[f"{c}_{a}"][hit], idx[hit])
        values[f"{c}_time_log10"] = (np.log10(np.maximum(ev[f"{c}_time"][hit], zero)), idx[hit])
    unknown = set(spec) - set(values)
    if unknown:
        raise ValueError(f"undeclared observables {sorted(unknown)}")
    return {k: values[k] for k in spec}


def run(directory, output, criteria_path=HERE / "criteria.json", bank_tool=None):
    directory = Path(directory)
    criteria = json.loads(Path(criteria_path).read_text())
    campaign = CAMPAIGN.load()
    rows = json.loads((directory / "shards.json").read_text())
    runs = read(directory, rows)
    n_bank = sum(1 for s in CAMPAIGN.states(campaign) if not s[4])
    catchers = [c["name"] for c in campaign["catchers"]]
    exact = {}
    exact["D1_reader"] = d1_reader(directory, bank_tool, {a: r for a, r in runs.items() if a.startswith("HY")})
    atol = next(c["atol"] for c in criteria["exact"] if c["name"] == "D2_parity")
    for arm in ("HYN", "HYT"):
        if arm in runs:
            exact[f"D2_parity:{arm}"] = d2_parity(directory, campaign, runs, arm, atol)
            exact[f"D5_coverage:{arm}"] = d5_coverage(runs, arm)
    if {"HYD", "G4D"} <= set(runs):
        exact["D3_refusal"] = d3_refusal(runs, n_bank)
    if {"G4D", "PUD"} <= set(runs):
        exact["D4_inert_plumbing"] = d4_plumbing(runs)
    stats = {}
    zero = float(criteria["zero"])
    for comp in criteria["comparisons"]:
        a, b = comp["reference"], comp["candidate"]
        if not {a, b} <= set(runs):
            continue
        res = covariance.compare_clustered(
            outcomes(runs[a], n_bank, catchers, a.startswith("HY")), outcomes(runs[b], n_bank, catchers, b.startswith("HY")),
            criteria["observables"], lambda o, s: entries(o, s, zero), means=tuple(criteria["means"]),
            min_bin_entries=int(criteria["min_bin_entries"]), min_trials=int(criteria["min_trials"]))
        shapes_ok = res["bonferroni_min_p"] is not None and res["bonferroni_min_p"] >= criteria["criteria"][0]["tol"]
        k = criteria["criteria"][1]["k"]
        means_ok = all(w is not None and abs(w["ratio"] - 1) <= k * w["err"] for w in res["worst_means"].values())
        tested = res["family"] > 0
        stats[comp["name"]] = dict(family=res["family"], bonferroni_min_p=res["bonferroni_min_p"], worst=res["worst"],
                                   worst_means=res["worst_means"], untested=len(res["untested"]),
                                   shapes_passed=bool(shapes_ok), means_passed=bool(means_ok),
                                   passed=bool(shapes_ok and means_ok) if tested else None)
    timing = {arm: dict(events=int(len(r["events"]["cpu_s"])), mean_cpu_ms=float(1e3 * r["events"]["cpu_s"].mean()))
              for arm, r in runs.items()}
    complete = {"D1_reader", "D2_parity:HYN", "D2_parity:HYT", "D3_refusal", "D4_inert_plumbing",
                "D5_coverage:HYN", "D5_coverage:HYT"} <= set(exact) and \
        {c["name"] for c in criteria["comparisons"]} <= set(stats)
    complete = complete and all(v["passed"] is not None for v in stats.values())
    passed = all(v["passed"] for v in exact.values()) and all(v["passed"] is not False for v in stats.values())
    record = dict(campaign=campaign["name"],
                  criteria_sha256=hashlib.sha256(Path(criteria_path).read_bytes()).hexdigest(),
                  bank=dict(content_hash=store.load(directory / "bank.npz").content_hash,
                            file_sha256=store.load(directory / "bank.npz").file_sha256),
                  exact=exact, statistical=stats, cpu_per_event=timing,
                  overall=("passed" if passed and complete else "failed" if not passed else "incomplete (pilot)"))
    Path(output).write_text(json.dumps(record, indent=1, default=float) + "\n")
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--criteria", type=Path, default=HERE / "criteria.json")
    p.add_argument("--bank-tool", default=shutil.which("hepsimlib-bank"))
    a = p.parse_args(argv)
    r = run(a.directory, a.output, a.criteria, a.bank_tool)
    print(json.dumps(dict(overall=r["overall"], exact={k: v["passed"] for k, v in r["exact"].items()},
                          statistical={k: dict(p=v["bonferroni_min_p"], passed=v["passed"])
                                       for k, v in r["statistical"].items()}, cpu_per_event=r["cpu_per_event"]),
                     indent=1))


if __name__ == "__main__":
    main()
