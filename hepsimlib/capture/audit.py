"""Strict, fail-closed audit of component captures (schema 1, docs/recorder.md); passing never certifies a bank.

Ported from the Mu2e fast simulation's capture audit (mu2e-fastsim-core a058b04). The checks a framework adds of its
own (its status codes, its resource-kill process) are passed in as ``checks``; ``resource_processes`` names step
processes that are resource kills.
"""
from __future__ import annotations

import numpy as np

from .tables import (EVENT, FIELDS, FLOATS, INT64, LINK_FIELDS, LINKS, MANIFEST_FIELDS, STATE, STATES, TABLES, TRACK,
                     fnv1a, frame, match, require, unique, vectors)


def same(a, b, label, atol):
    require(np.allclose(a, b, rtol=0, atol=atol), label + " mismatch")


def state_equal(a, ap, ai, b, bp, bi, atol):
    same(vectors(a, ap, FLOATS)[ai], vectors(b, bp, FLOATS)[bi], "state continuity", atol)


def check_links(tables, links):
    """Validate row-aligned links and manifests; return (exitAtBirth, bornOnSurface, creator step rows)."""
    t, s, e = tables["tracks"], tables["steps"], tables["events"]
    require(links is not None and set(LINKS) <= set(links), "missing links/manifest tables")
    lk, mf = links["links"], links["manifest"]
    require(set([*EVENT, *LINK_FIELDS]) <= set(lk), "missing links fields")
    require(set([*EVENT, *MANIFEST_FIELDS]) <= set(mf), "missing manifest fields")
    for field in [*EVENT, "index", "track", "parent"]:
        require(len(lk[field]) == len(t[field]) and np.array_equal(lk[field], t[field]), f"links {field} misaligned")
    for field in EVENT:
        require(len(mf[field]) == len(e[field]) and np.array_equal(mf[field], e[field]), f"manifest {field} misaligned")
    for field in ("linksPresent", "linksIDsMatch"):
        require(np.all(mf[field] == True), f"manifest {field} is not true")  # noqa: E712
    require(np.all(mf["schema"] == 1), "unsupported links schema")
    require(np.array_equal(mf["nLinks"], e["nTracks"]), "links census mismatch")
    hashes = np.array([fnv1a(str(text)) for text in mf["manifest"]], dtype=object)
    require(np.array_equal(hashes, mf["manifestHash"].astype(str).astype(object)), "manifest hash mismatch")
    require(all(str(text).startswith("manifest|1\n") for text in mf["manifest"]), "unsupported manifest format")
    require(len(set(map(str, mf["manifestHash"]))) <= 1, "manifest differs between events")
    birth, surface = lk["exitAtBirth"].astype(bool), lk["bornOnSurface"].astype(bool)
    events = unique(e, EVENT, "event").assign(_event=np.arange(len(e["event"])))
    er = match(frame(t, EVENT), events, EVENT, "links")["_event"].to_numpy()
    require(np.array_equal(np.bincount(er, weights=birth, minlength=len(events)).astype(np.int64),
                           mf["nBirthExits"]), "exit-at-birth census mismatch")
    primary = t["parent"] == 0
    creator = lk["creatorStep"]
    require(np.all(creator[primary] == -1) and np.all(creator[~primary] >= 0), "creator step identity mismatch")
    steps = frame(s, [*EVENT, "index"]).assign(_step=np.arange(len(s["event"])))
    keys = frame(lk, EVENT).assign(index=creator)
    rows = np.full(len(creator), -1, dtype=np.int64)
    rows[~primary] = match(keys.loc[~primary], steps, [*EVENT, "index"], "creator step")["_step"].to_numpy()
    require(np.array_equal(s["track"][rows[~primary]], t["parent"][~primary]), "creator step is not the parent's")
    require(np.all(s["preInside"][rows[~primary]]), "secondary created outside the component")
    counts = np.bincount(rows[~primary], minlength=len(s["event"]))
    require(np.array_equal(counts, s["secondaries"]), "creator step secondary census mismatch")
    require(np.all(~birth | ((t["reason"] == 4) & ~primary)), "exit at birth must be a reason-4 secondary")
    require(np.all(birth | (t["reason"] != 4)), "reason 4 requires an exit-at-birth link")
    require(np.all(~surface | ~primary), "primary flagged as surface-born")
    return birth, surface, rows


def audit(tables, expected_events, *, atol=1e-7, links=None, resource_processes=(), checks=()):
    """Fail closed on missing, partial or inconsistent events; ``atol`` is an absolute geometry/state tolerance
    (mm, MeV, ns). With links, flagged exits at birth (reason 4) and surface-born children are accepted.
    ``checks`` are callables ``(tables, links)`` run after the generic checks (a framework's own rules)."""
    require(isinstance(expected_events, (int, np.integer)) and expected_events > 0, "positive expected events required")
    require(np.isfinite(atol) and 0 < atol <= 1e-5, "invalid audit tolerance")
    for name in TABLES:
        require(name in tables, f"missing table {name}")
        table = tables[name]
        fields = [*EVENT, *FIELDS[name].split(),
                  *(prefix + field for prefix in STATES.get(name, ()) for field in STATE)]
        require(set(fields) <= set(table), f"missing {name} fields: {set(fields) - set(table)}")
        for key in fields:
            try:
                value = np.asarray(table[key])
            except ValueError as exc:
                raise ValueError(f"{name}: malformed {key}") from exc
            require(value.ndim > 0 and len(value) == len(table["event"]), f"{name}: malformed {key}")
            if key in INT64 or key.endswith("_volume"):
                require(value.dtype.kind == "i" and value.dtype.itemsize == 8, f"{name}: {key} must be int64")
            if value.dtype.kind == "f":
                require(np.isfinite(value).all(), f"{name}: nonfinite {key}")
        require((frame(table, EVENT) >= 0).all().all(), f"{name}: negative event identity")
    e, t, s, c, v = (tables[name] for name in TABLES)
    events = unique(e, EVENT, "event").assign(_event=np.arange(len(e["event"])))
    require(len(events) == expected_events, "expected event census mismatch")
    require(np.all(e["schema"] == 1), "unsupported capture schema")
    require(np.all(e["complete"] == True), "event complete is not true")  # noqa: E712
    for field in "bankAdmissible nErrors nResourceKills".split():
        require(np.all(e[field] == 0), f"invalid event {field}")
    require(np.all(e["errors"] == ""), "event errors present")
    for field in ("geometryTag", "physicsTag", "envelope"):
        require(np.all(e[field] != ""), f"empty {field}")
    event_rows = {}
    for name in TABLES[1:]:
        table = tables[name]
        rows = unique(table, [*EVENT, "index"], name)
        event_rows[name] = match(rows, events, EVENT, name)["_event"].to_numpy()
        counts = np.bincount(event_rows[name], minlength=len(events))
        require(np.array_equal(e["n" + name.capitalize()], counts), f"{name} census mismatch")
        require(np.all((table["index"] >= 0) & (table["index"] < counts[event_rows[name]])), f"{name} row index gap")
    require(np.all(e["nTracks"] > 0) and np.all(e["nVolumes"] > 0), "empty capture event")
    birth = surface = np.zeros(len(t["track"]), dtype=bool)
    if links is not None:
        birth, surface, _ = check_links(tables, links)
    births = np.bincount(event_rows["tracks"], weights=birth, minlength=len(events)).astype(np.int64)
    require(np.array_equal(e["nG4Tracks"], e["nTracks"] - births), "G4 track census mismatch")
    tracks = unique(t, TRACK, "track").assign(_track=np.arange(len(t["track"])))
    require(np.all(t["track"] > 0) and np.all(t["parent"] >= 0), "invalid track/parent identity")
    require(np.all((t["entered"] == True) | birth) and np.all(t["entered"][birth] == False) and  # noqa: E712
            np.all(t["ended"] == True), "unclosed/unentered track")  # noqa: E712
    require(np.all(np.isin(t["reason"], [1, 2]) | (birth & (t["reason"] == 4))), "unsupported terminal reason")
    require(np.all(np.where(t["reason"] == 1, t["status"] == 2,
                            np.where(t["reason"] == 4, t["status"] == 2, np.isin(t["status"], [0, 1])))),
            "terminal status mismatch")
    primary = t["parent"] == 0
    require(np.all(np.bincount(event_rows["tracks"], weights=primary, minlength=len(events)) == 1),
            "expected one primary per event")
    parents = frame(t, [*EVENT, "parent"]).rename(columns={"parent": "track"})
    parent_rows = match(parents.loc[~primary], tracks, TRACK, "parent")["_track"].to_numpy()
    require(np.all(t["index"][parent_rows] < t["index"][~primary]), "ancestry cycle/order mismatch")
    require(np.array_equal(np.bincount(parent_rows, minlength=len(tracks)), t["secondaries"]),
            "secondary ancestry census mismatch")
    unique(c, TRACK, "classification")
    classified = match(frame(c, TRACK), tracks, TRACK, "classification")["_track"].to_numpy()
    require(len(classified) == len(tracks) and np.array_equal(c["killed"], birth[classified]),
            "classification acceptance census mismatch")
    require(np.all(np.isin(c["status"], [0, 1, 2])), "unsupported classification status")
    require(np.all((c["status"] != 2) | ((t["steps"][classified] == 0) & ~primary[classified])),
            "terminal classification requires a zero-step child")
    for field in ("parent", "pdg"):
        require(np.array_equal(c[field], t[field][classified]), f"classification {field} mismatch")
    state_equal(c, "state_", slice(None), t, "begin_", classified, atol)
    rows = match(frame(s, [*TRACK, "index", "number"]), tracks, TRACK, "step")
    rows = rows.sort_values([*EVENT, "index"])
    require(np.array_equal(rows["number"], rows.groupby(TRACK, sort=False).cumcount() + 1),
            "step sequence gap/order mismatch")
    sr = match(frame(s, TRACK), tracks, TRACK, "step")["_track"].to_numpy()
    require(np.all(t["steps"] >= 0) and np.array_equal(np.bincount(sr, minlength=len(tracks)), t["steps"]),
            "track step census mismatch")
    require(np.all(s["secondaries"] >= 0), "negative secondary count")
    child_counts = np.zeros(len(tracks), dtype=np.int64)
    np.add.at(child_counts, sr, s["secondaries"])
    require(np.array_equal(child_counts, t["secondaries"]), "step secondary census mismatch")
    zero_step = t["steps"] == 0
    require(np.all(np.isin(t["reason"][zero_step], [1, 4]) & ~primary[zero_step]), "unsupported zero-step track")
    state_equal(t, "begin_", zero_step, t, "end_", zero_step, atol)
    for field in ("deposit", "nonIonizingDeposit", "length"):
        require(np.all(s[field] >= 0), f"negative raw {field}")
    exit_mask = s["preInside"] & ~s["postInside"]
    require(np.array_equal(exit_mask, s["firstExit"]), "first exit flag mismatch")
    require(np.array_equal(np.bincount(sr[exit_mask], minlength=len(tracks)), (t["reason"] == 2).astype(int)),
            "first exit census mismatch")
    require(np.all(s["number"][exit_mask] == t["steps"][sr[exit_mask]]), "steps after first exit")
    require(np.all(np.isin(s["trackStatus"][exit_mask], [0, 1])), "nontransport exit status")
    ordered = rows.sort_values([*TRACK, "number"])
    consecutive = ordered["_track"].to_numpy()[1:] == ordered["_track"].to_numpy()[:-1]
    previous, following = ordered.index.to_numpy()[:-1][consecutive], ordered.index.to_numpy()[1:][consecutive]
    require(np.all(np.isin(s["trackStatus"][previous], [0, 1])), "unsupported intermediate track status")
    state_equal(s, "post_", previous, s, "pre_", following, atol)
    require(np.array_equal(s["postInside"][previous], s["preInside"][following]), "step inside continuity mismatch")
    first, last = rows.groupby(TRACK, sort=False).head(1), rows.groupby(TRACK, sort=False).tail(1)
    state_equal(t, "begin_", first["_track"], s, "pre_", first.index, atol)
    state_equal(t, "end_", last["_track"], s, "post_", last.index, atol)
    require(np.array_equal(s["trackStatus"][last.index], t["status"][last["_track"]]), "last step status mismatch")
    entry = ~s["preInside"] & s["postInside"]
    require(np.array_equal(np.bincount(sr[entry], minlength=len(tracks)), primary.astype(int)), "entry census mismatch")
    state_equal(t, "entrance_", sr[entry], s, "post_", entry, atol)
    state_equal(t, "entrance_", ~primary & ~birth, t, "begin_", ~primary & ~birth, atol)
    outside = ~s["preInside"]
    require(np.all((t["parent"][sr[outside]] == 0) & (s["secondaries"][outside] == 0) &
                   (s["pre_material"][outside] == "G4_Galactic")), "nontrivial outside transport")
    for field, width in (("translation", 3), ("rotation", 9), ("halfSize", 3)):
        require(np.shape(e[field]) == (len(events), width), f"invalid event {field} shape")
    rotation, translation, half = e["rotation"].reshape(-1, 3, 3), e["translation"], e["halfSize"]
    same(rotation @ rotation.transpose(0, 2, 1), np.eye(3), "orthonormal rotation", 1e-10)
    same(np.linalg.det(rotation), 1, "proper rotation", 1e-10)
    require(np.all(half > 0), "invalid box halfSize")
    volumes = unique(v, [*EVENT, "id"], "volume")
    require(np.all(v["id"] >= 0), "negative volume ID")
    envelope = v["name"] == e["envelope"][event_rows["volumes"]]
    require(np.all(np.bincount(event_rows["volumes"][envelope], minlength=len(events)) == 1), "envelope census mismatch")
    require(np.all(v["solid"][envelope] == "G4Box"), "envelope is not a box")
    for field in ("translation", "rotation"):
        same(v[field][envelope], e[field][event_rows["volumes"][envelope]], "envelope " + field, atol)
    for name, prefixes in STATES.items():
        table, er = tables[name], event_rows[name]
        for prefix in prefixes:
            keep = ~birth if (name, prefix) == ("tracks", "entrance_") else slice(None)
            for local, world, shift in (("l", "", translation[er]), ("lp", "p", 0)):
                calculated = np.einsum("nij,nj->ni", rotation[er], vectors(table, prefix + local, "xyz")) + shift
                same(calculated[keep], vectors(table, prefix + world, "xyz")[keep],
                     f"{name} {prefix}{world} covariance", atol)
            require(np.all(table[prefix + "weight"] == 1), "biased state weight")
            refs = frame(table, [*EVENT, prefix + "volume"]).rename(columns={prefix + "volume": "id"})
            require(np.all(refs["id"] >= -1), "invalid state volume")
            match(refs.loc[refs["id"] >= 0], volumes, [*EVENT, "id"], "state volume")
    for prefix, inside in (("pre_", s["preInside"]), ("post_", s["postInside"])):
        distance = np.abs(vectors(s, prefix + "l", "xyz")) - half[event_rows["steps"]]
        require(np.all(np.where(inside, np.all(distance <= atol, axis=1), np.any(distance >= -atol, axis=1))),
                "inside/box mismatch")
    boundary = entry | exit_mask
    distance = np.abs(vectors(s, "post_l", "xyz")) - half[event_rows["steps"]]
    require(np.all(np.all(distance[boundary] <= atol, axis=1) & np.any(np.abs(distance[boundary]) <= atol, axis=1)),
            "crossing not on exact box boundary")
    require(np.all(s["post_stepStatus"][boundary] == 1), "crossing lacks geometry boundary status")
    begin_distance = np.abs(vectors(t, "begin_l", "xyz")) - half[event_rows["tracks"]]
    require(np.all(np.any(begin_distance[primary] > atol, axis=1)), "primary not born strictly outside")
    on_face = np.all(begin_distance <= atol, axis=1) & np.any(np.abs(begin_distance) <= atol, axis=1)
    owned = ~primary & ~birth
    require(np.all(np.where(surface[owned], on_face[owned], np.all(begin_distance[owned] < -atol, axis=1))),
            "secondary not born strictly inside")
    require(np.all(np.any(begin_distance[birth] >= -atol, axis=1)), "exit at birth inside the component")
    if resource_processes:
        require(not np.any(np.isin(s["post_process"], list(resource_processes))), "resource kill step")
    for check in checks:
        check(tables, links)
    return {"events": len(events), "rows": {name: len(tables[name]["event"]) for name in TABLES},
            "bank_admissible": False, "experimental": True, "first_exits": int(exit_mask.sum()),
            "natural_stops": int(np.count_nonzero(t["reason"] == 1)), "secondaries": int((~primary).sum()),
            "zero_deposit_steps": int(np.count_nonzero(s["deposit"] == 0)),
            "links_checked": links is not None, "exits_at_birth": int(birth.sum()),
            "surface_born_secondaries": int((surface & ~primary).sum()),
            "raw_deposit": float(np.sum(s["deposit"])), "raw_nonionizing_deposit": float(np.sum(s["nonIonizingDeposit"])),
            "component_deposit": float(np.sum(s["deposit"][s["preInside"]])),
            "approach_deposit": float(np.sum(s["deposit"][outside]))}
