"""Placement covariance of a component bank against full Geant4 (validation/a3/README.md).

A bank recorded with the component at one placement is replayed at the entrance states of held-out Geant4 captures
taken at other placements (moved, rotated). Both sides give the same per-trial outcomes in the *global* frame, read
from the capture's recorded global columns and from the replay's transformed outputs:

    trials   class (incoming species and kinetic energy), total deposit, exit count
    hits     per trial and local volume: deposit above a threshold (occupancy)
    exits    species, kinetic energy, position relative to the placement's translation, direction, time after entry

``compare`` histograms every declared observable per class on the declared binning and gives the two-sample chi2
(ROOT "UU", hepsimlib.library.fidelity.shape_chi2) with its p-value, Bonferroni-corrected over the family, and the
ratio of mean total deposits per class with its independent-sample error. Needs scipy (chi2 survival function).
"""
from __future__ import annotations

import numpy as np

from .. import component as cr
from ..library import fidelity
from .bank import _volumes
from .tables import EVENT, frame, match, require, unique, vectors

ZERO = 1e-12          # a deposit or time at or below this is "none" (it enters the lowest, underflow bin)


def _classes(pdg, kinetic, decimals=3):
    """Class label per trial: incoming PDG code and kinetic energy (MeV, rounded)."""
    return np.array([f"{int(p)}@{round(float(k), decimals):g}" for p, k in zip(pdg, kinetic)], dtype=object)


def from_capture(tables, threshold=0.1):
    """Per-trial outcomes of an audited capture (one trial per event), in the global frame."""
    e, t, s = tables["events"], tables["tracks"], tables["steps"]
    events = unique(e, EVENT, "event").assign(_event=np.arange(len(e["event"])))
    rows, _ = _volumes(tables)
    local = {int(i): k for k, i in enumerate(tables["volumes"]["id"][rows])}
    nvol = len(rows)
    primary = np.flatnonzero(t["parent"] == 0)
    pe = match(frame(t, EVENT).iloc[primary], events, EVENT, "primary")["_event"].to_numpy()
    order = np.argsort(pe)
    primary = primary[order]
    require(np.array_equal(pe[order], np.arange(len(events))), "one primary per event")
    t0 = t["entrance_time"][primary]
    kin = t["entrance_kinetic"][primary]
    se = match(frame(s, EVENT), events, EVENT, "step")["_event"].to_numpy()
    inside = s["preInside"].astype(bool)
    edep = np.bincount(se[inside], weights=s["deposit"][inside], minlength=len(events))
    vol = np.array([local.get(int(i), -1) for i in s["pre_volume"][inside]], dtype=np.int64)
    require(np.all(vol >= 0), "in-component step outside the envelope inventory")
    per_volume = np.zeros((len(events), nvol))
    np.add.at(per_volume, (se[inside], vol), s["deposit"][inside])
    translation = np.asarray(e["translation"], dtype=float)
    first = np.flatnonzero(s["firstExit"].astype(bool))
    tr = match(frame(t, EVENT), events, EVENT, "track")["_event"].to_numpy()
    tracks = unique(t, [*EVENT, "track"], "track").assign(_track=np.arange(len(t["track"])))
    first_track = match(frame(s, [*EVENT, "track"]).iloc[first], tracks, [*EVENT, "track"], "exit")["_track"].to_numpy()
    born = np.flatnonzero(t["reason"] == 4)
    ex_event = np.r_[se[first], tr[born]].astype(np.int64)
    exits = dict(trial=ex_event, pdg=np.r_[t["pdg"][first_track], t["pdg"][born]].astype(np.int64),
                 kinetic=_kinetic(np.r_[s["post_energy"][first], t["begin_energy"][born]],
                                  np.vstack([vectors(s, "post_p", "xyz")[first], vectors(t, "begin_p", "xyz")[born]])),
                 position=np.vstack([vectors(s, "post_", "xyz")[first], vectors(t, "begin_", "xyz")[born]])
                 - translation[ex_event],
                 momentum=np.vstack([vectors(s, "post_p", "xyz")[first], vectors(t, "begin_p", "xyz")[born]]),
                 time=np.r_[s["post_time"][first], t["begin_time"][born]] - t0[ex_event])
    return dict(trials=dict(cls=_classes(t["pdg"][primary], kin), edep=edep,
                            exits=np.bincount(ex_event, minlength=len(events)),
                            cluster=np.full(len(events), -1, np.int64)),
                hits=per_volume > threshold, exits=exits, nvol=nvol)


def _kinetic(energy, momentum):
    """Kinetic energy from total energy and momentum (the same definition on both sides)."""
    energy = np.asarray(energy, float)
    return energy - np.sqrt(np.maximum(energy ** 2 - np.sum(np.asarray(momentum, float) ** 2, axis=1), 0))


def from_replay(library, tables, rng, *, dtype=np.float64, threshold=0.1):
    """Per-trial outcomes of replaying ``library`` at the entrance states of a capture's primaries, each in its own
    event's placement (same global frame as from_capture)."""
    e, t = tables["events"], tables["tracks"]
    events = unique(e, EVENT, "event").assign(_event=np.arange(len(e["event"])))
    primary = np.flatnonzero(t["parent"] == 0)
    pe = match(frame(t, EVENT).iloc[primary], events, EVENT, "primary")["_event"].to_numpy()
    primary = primary[np.argsort(pe)]
    n = len(primary)
    rot, tra = np.asarray(e["rotation"], float), np.asarray(e["translation"], float)
    groups = np.unique(np.hstack([rot, tra]), axis=0, return_inverse=True)[1].ravel()
    incoming = dict(event=np.column_stack([np.arange(n), np.zeros(n, np.int64), np.zeros(n, np.int64)]),
                    track=np.ones(n, np.int64), parent=np.full(n, -1, np.int64), pdg=t["pdg"][primary].astype(np.int64),
                    position=vectors(t, "entrance_", "xyz")[primary], momentum=vectors(t, "entrance_p", "xyz")[primary],
                    energy=t["entrance_energy"][primary].astype(float), time=t["entrance_time"][primary].astype(float))
    steps, exits = [], []
    drawn = np.full(n, -1, np.int64)
    for g in range(groups.max() + 1 if n else 0):
        rows = np.flatnonzero(groups == g)
        placement = cr.Placement(library.component, 0, rot[rows[0]].reshape(3, 3), tra[rows[0]],
                                 np.arange(library.component.volumes))
        r = cr.replay(library, placement, {k: v[rows] for k, v in incoming.items()}, np.zeros(len(rows), np.int64),
                      rng, next_track=2, dtype=dtype)
        drawn[rows] = r.trials
        steps.append((r.steps["event"][:, 0], r.steps["volume"], np.asarray(r.steps["edep"], float)))
        trial = r.exits["event"][:, 0]
        exits.append(dict(trial=trial, pdg=r.exits["pdg"],
                          kinetic=_kinetic(r.exits["energy"], r.exits["momentum"]),
                          position=np.asarray(r.exits["position"], float) - placement.translation,
                          momentum=np.asarray(r.exits["momentum"], float),
                          time=np.asarray(r.exits["time"], float) - incoming["time"][trial]))
    trial_s = np.concatenate([x[0] for x in steps]) if steps else np.empty(0, np.int64)
    volume = np.concatenate([x[1] for x in steps]) if steps else np.empty(0, np.int64)
    edep_s = np.concatenate([x[2] for x in steps]) if steps else np.empty(0)
    ex = {k: np.concatenate([x[k] for x in exits]) for k in exits[0]} if exits else {}
    per_volume = np.zeros((n, library.component.volumes))
    np.add.at(per_volume, (trial_s, volume), edep_s)
    return dict(trials=dict(cls=_classes(incoming["pdg"], t["entrance_kinetic"][primary]),
                            edep=np.bincount(trial_s, weights=edep_s, minlength=n),
                            exits=np.bincount(ex["trial"], minlength=n), cluster=drawn),
                hits=per_volume > threshold, exits=ex, nvol=library.component.volumes)


def _observables(o, spec):
    """{observable: (values per entry, class per entry)} for the declared observables."""
    return {k: (v, o["trials"]["cls"][t]) for k, (v, t) in _entries(o, spec).items()}


def _entries(o, spec):
    """{observable: (values per entry, trial index per entry)} for the declared observables."""
    tr, ex = o["trials"], o["exits"]
    p = np.linalg.norm(ex["momentum"], axis=1)
    direction = ex["momentum"] / np.where(p > 0, p, 1)[:, None]
    hit_trial, hit_volume = np.nonzero(o["hits"])
    species = np.where(ex["pdg"] == 22, 0, np.where(np.abs(ex["pdg"]) == 11, 1, 2))
    trials = np.arange(len(tr["edep"]))
    te = ex["trial"]
    values = {
        "edep_log10": (np.log10(np.maximum(tr["edep"], ZERO)), trials),
        "exits": (tr["exits"].astype(float), trials),
        "hit_volume": (hit_volume.astype(float), hit_trial),
        "exit_species": (species.astype(float), te),
        "exit_kinetic_log10": (np.log10(np.maximum(ex["kinetic"], ZERO)), te),
        "exit_time_log10": (np.log10(np.maximum(ex["time"], ZERO)), te),
        **{f"exit_{a}": (ex["position"][:, i], te) for i, a in enumerate("xyz")},
        **{f"exit_u{a}": (direction[:, i], te) for i, a in enumerate("xyz")},
    }
    unknown = set(spec) - set(values)
    require(not unknown, f"undeclared observables {sorted(unknown)}")
    return {k: values[k] for k in spec}


def _edges(binning):
    lo, hi, n = binning
    return np.r_[-np.inf, np.linspace(lo, hi, int(n) + 1), np.inf]


def compare(reference, candidate, spec, *, min_entries=50):
    """Per class and declared observable: two-sample chi2 of the histograms (fidelity.shape_chi2), its p-value and
    the Bonferroni-corrected minimum over the family; per class: ratio of mean total deposits (candidate / reference)
    with its independent-sample error. Classes with fewer than ``min_entries`` entries on either side are not tested
    (listed)."""
    from scipy.stats import chi2 as chi2_dist
    obs_r, obs_c = _observables(reference, spec), _observables(candidate, spec)
    classes = sorted(set(reference["trials"]["cls"]) | set(candidate["trials"]["cls"]))
    tests, untested = [], []
    for name, binning in spec.items():
        edges = _edges(binning)
        (vr, cr_), (vc, cc) = obs_r[name], obs_c[name]
        for k in classes:
            a, b = vr[cr_ == k], vc[cc == k]
            if min(len(a), len(b)) < min_entries:
                untested.append(dict(observable=name, cls=k, reference=len(a), candidate=len(b)))
                continue
            ha, hb = np.histogram(a, edges)[0], np.histogram(b, edges)[0]
            c2, ndf = fidelity.shape_chi2(hb, ha)
            p = float(chi2_dist.sf(c2, ndf)) if ndf > 0 else 1.0
            tests.append(dict(observable=name, cls=k, chi2=c2, ndf=ndf, p=p, reference=len(a), candidate=len(b)))
    family = len(tests)
    worst = min(tests, key=lambda x: x["p"]) if tests else None
    means = {}
    for k in classes:
        a = reference["trials"]["edep"][reference["trials"]["cls"] == k]
        b = candidate["trials"]["edep"][candidate["trials"]["cls"] == k]
        if len(a) < 2 or len(b) < 2 or a.mean() <= 0:
            continue
        ratio = b.mean() / a.mean()
        err = ratio * np.hypot(a.std(ddof=1) / np.sqrt(len(a)) / a.mean(),
                               b.std(ddof=1) / np.sqrt(len(b)) / max(b.mean(), 1e-300))
        means[k] = dict(ratio=float(ratio), err=float(err), reference=len(a), candidate=len(b))
    worst_mean = max(means.values(), key=lambda m: abs(m["ratio"] - 1) / m["err"] if m["err"] > 0 else 0) if means else None
    return dict(tests=tests, untested=untested, family=family,
                bonferroni_min_p=min(1.0, worst["p"] * family) if worst else None, worst=worst,
                mean_edep=means, worst_mean_edep=worst_mean)


# ------------------------------------------------------------------ cluster-robust comparison (A3 v2)
def _rates(o, entries, trials, edges):
    """Per-trial entry counts per bin for the given trials: (rate vector, cluster-robust covariance of the rate,
    entries per bin). A G4 trial is its own cluster; replayed trials that drew the same bank trial form one cluster
    (they are the same recorded outcome), so neither correlated entries within a trial nor repeated draws are
    counted as independent."""
    values, trial = entries
    local = -np.ones(len(o["trials"]["edep"]), np.int64)
    local[trials] = np.arange(len(trials))
    keep = local[trial] >= 0
    n, nb = len(trials), len(edges) - 1
    h = np.zeros((n, nb))
    np.add.at(h, (local[trial[keep]], np.searchsorted(edges, values[keep], side="right") - 1), 1.0)
    cl = o["trials"]["cluster"][trials]
    cl = np.where(cl < 0, -1 - np.arange(n), cl)                 # own cluster for every G4 trial
    codes, inverse = np.unique(cl, return_inverse=True)
    g = np.zeros((len(codes), nb))
    np.add.at(g, inverse.ravel(), h)
    m = np.bincount(inverse.ravel(), minlength=len(codes)).astype(float)
    r = h.mean(axis=0)
    u = g - m[:, None] * r
    c = len(codes)
    cov = (u.T @ u) / n ** 2 * (c / (c - 1) if c > 1 else 1.0)
    return r, cov, h.sum(axis=0)


def _scalar(o, values, trials):
    """Mean of a per-trial value over the given trials and its cluster-robust error."""
    v = values[trials]
    cl = o["trials"]["cluster"][trials]
    cl = np.where(cl < 0, -1 - np.arange(len(trials)), cl)
    codes, inverse = np.unique(cl, return_inverse=True)
    mean = v.mean()
    u = np.bincount(inverse.ravel(), weights=v - mean, minlength=len(codes))
    c = len(codes)
    return mean, np.sqrt((u @ u) / len(v) ** 2 * (c / (c - 1) if c > 1 else 1.0))


def compare_robust(reference, candidate, spec, *, min_bin_entries=10, min_trials=50):
    """Per class and declared observable: Hotelling T^2 of the per-trial rate vectors over the bins holding at least
    ``min_bin_entries`` entries on both sides, with cluster-robust covariances (see _rates), its chi2 p-value
    (k = rank of the summed covariance), and the Bonferroni-corrected minimum over the family; per class, the ratio
    of mean total deposits with its cluster-robust error. Classes with fewer than ``min_trials`` trials on either side
    are not tested (listed)."""
    from scipy.stats import chi2 as chi2_dist
    ent_r, ent_c = _entries(reference, spec), _entries(candidate, spec)
    cls_r, cls_c = reference["trials"]["cls"], candidate["trials"]["cls"]
    classes = sorted(set(cls_r) | set(cls_c))
    tests, untested, means = [], [], {}
    for k in classes:
        tr_r, tr_c = np.flatnonzero(cls_r == k), np.flatnonzero(cls_c == k)
        if min(len(tr_r), len(tr_c)) < min_trials:
            untested.append(dict(cls=k, reference=len(tr_r), candidate=len(tr_c)))
            continue
        for name, binning in spec.items():
            edges = _edges(binning)
            ra, sa, na = _rates(reference, ent_r[name], tr_r, edges)
            rb, sb, nb = _rates(candidate, ent_c[name], tr_c, edges)
            use = (na >= min_bin_entries) & (nb >= min_bin_entries)
            if not use.any():
                untested.append(dict(cls=k, observable=name, reason="no bin with enough entries"))
                continue
            d = (rb - ra)[use]
            s = (sa + sb)[np.ix_(use, use)]
            w, v = np.linalg.eigh(s)
            good = w > w.max() * 1e-10
            t2 = float(np.sum((v[:, good].T @ d) ** 2 / w[good]))
            dof = int(good.sum())
            p = float(chi2_dist.sf(t2, dof))
            tests.append(dict(observable=name, cls=k, t2=t2, dof=dof, p=p, bins=int(use.sum()),
                              reference=len(tr_r), candidate=len(tr_c)))
        ma, ea = _scalar(reference, reference["trials"]["edep"], tr_r)
        mb, eb = _scalar(candidate, candidate["trials"]["edep"], tr_c)
        if ma > 0:
            ratio = mb / ma
            means[k] = dict(ratio=float(ratio), err=float(ratio * np.hypot(ea / ma, eb / mb if mb > 0 else 0.0)),
                            reference=len(tr_r), candidate=len(tr_c))
    family = len(tests)
    worst = min(tests, key=lambda x: x["p"]) if tests else None
    worst_mean = max(means.values(), key=lambda m: abs(m["ratio"] - 1) / m["err"] if m["err"] > 0 else 0) if means else None
    return dict(tests=tests, untested=untested, family=family,
                bonferroni_min_p=min(1.0, worst["p"] * family) if worst else None, worst=worst,
                mean_edep=means, worst_mean_edep=worst_mean)


def compare_clustered(reference, candidate, spec, entries, *, means=("edep",), min_bin_entries=10, min_trials=50):
    """compare_robust for any per-trial outcome: ``entries(o, spec)`` gives {observable: (values, trial index)} and
    ``o["trials"]`` holds ``cls``, ``cluster`` (-1: its own cluster), ``edep`` (the trial count) and every name in
    ``means``. Returns the tests, the Bonferroni minimum, and per mean quantity the per-class ratios (candidate /
    reference) with cluster-robust errors and the class furthest from 1 in errors (validation/a4)."""
    from scipy.stats import chi2 as chi2_dist
    ent_r, ent_c = entries(reference, spec), entries(candidate, spec)
    cls_r, cls_c = reference["trials"]["cls"], candidate["trials"]["cls"]
    tests, untested, ratios = [], [], {m: {} for m in means}
    for k in sorted(set(cls_r) | set(cls_c)):
        tr_r, tr_c = np.flatnonzero(cls_r == k), np.flatnonzero(cls_c == k)
        if min(len(tr_r), len(tr_c)) < min_trials:
            untested.append(dict(cls=k, reference=len(tr_r), candidate=len(tr_c)))
            continue
        for name, binning in spec.items():
            edges = _edges(binning)
            ra, sa, na = _rates(reference, ent_r[name], tr_r, edges)
            rb, sb, nb = _rates(candidate, ent_c[name], tr_c, edges)
            use = (na >= min_bin_entries) & (nb >= min_bin_entries)
            if not use.any():
                untested.append(dict(cls=k, observable=name, reason="no bin with enough entries"))
                continue
            d = (rb - ra)[use]
            w, v = np.linalg.eigh((sa + sb)[np.ix_(use, use)])
            good = w > w.max() * 1e-10
            t2 = float(np.sum((v[:, good].T @ d) ** 2 / w[good]))
            dof = int(good.sum())
            tests.append(dict(observable=name, cls=k, t2=t2, dof=dof, p=float(chi2_dist.sf(t2, dof)),
                              bins=int(use.sum()), reference=len(tr_r), candidate=len(tr_c)))
        for m in means:
            ma, ea = _scalar(reference, reference["trials"][m], tr_r)
            mb, eb = _scalar(candidate, candidate["trials"][m], tr_c)
            if ma > 0:
                ratio = mb / ma
                ratios[m][k] = dict(ratio=float(ratio), err=float(ratio * np.hypot(ea / ma, eb / mb if mb > 0 else 0.0)),
                                    reference=len(tr_r), candidate=len(tr_c))
    worst = min(tests, key=lambda x: x["p"]) if tests else None
    worst_means = {m: (max(r.values(), key=lambda x: abs(x["ratio"] - 1) / x["err"] if x["err"] > 0 else 0)
                       if r else None) for m, r in ratios.items()}
    return dict(tests=tests, untested=untested, family=len(tests),
                bonferroni_min_p=min(1.0, worst["p"] * len(tests)) if worst else None, worst=worst,
                means=ratios, worst_means=worst_means)
