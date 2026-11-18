"""The capture tables (schema 1, docs/recorder.md): field lists, reading, and the small table helpers.

The tables are dictionaries name -> {field: numpy array}. ``read`` gives them from a capture file (ROOT ntuples under
one directory, read with uproot) in their canonical dtypes: integer columns int64, flag columns bool, strings as
numpy string arrays, the fixed-width vectors (translation, rotation, halfSize) as 2-D float arrays.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVENT = ["run", "subrun", "event"]
TRACK = [*EVENT, "track"]
TABLES = ("events", "tracks", "steps", "classifications", "volumes")
LINKS = ("links", "manifest")
STATES = {"tracks": ("begin_", "end_", "entrance_"), "steps": ("pre_", "post_"), "classifications": ("state_",)}
FLOATS = "x y z px py pz lx ly lz lpx lpy lpz energy kinetic mass charge time properTime weight".split()
STATE = [*FLOATS, "volume", "placement", "material", "process", "stepStatus"]
FIELDS = {
    "events": "schema geometryTag physicsTag envelope translation rotation halfSize complete bankAdmissible errors "
              "nTracks nSteps nClassifications nVolumes nErrors nG4Tracks nResourceKills cpuTime realTime",
    "tracks": "index track parent steps secondaries pdg status reason entered ended creator stopProcess",
    "steps": "index track number secondaries deposit nonIonizingDeposit length trackStatus preInside postInside firstExit",
    "classifications": "index track parent pdg status killed",
    "volumes": "index id copy name logical motherLogical material solid density translation rotation",
}
LINK_FIELDS = "index track parent creatorStep bornOnSurface exitAtBirth".split()
MANIFEST_FIELDS = "linksPresent linksIDsMatch schema g4Version manifest manifestHash nLinks nBirthExits".split()
INT64 = set(EVENT + "index track parent steps secondaries number id copy nTracks nSteps nClassifications nVolumes "
                    "nErrors nG4Tracks nResourceKills creatorStep nLinks nBirthExits".split())
BOOL = set("complete bankAdmissible entered ended preInside postInside firstExit killed linksPresent linksIDsMatch "
           "bornOnSurface exitAtBirth".split())
VECTORS = {"translation": 3, "rotation": 9, "halfSize": 3}
DIRECTORY = "capture"


def frame(table, columns):
    return pd.DataFrame({k: table[k] for k in columns})


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique(table, columns, label):
    data = frame(table, columns)
    require(not data.duplicated().any(), f"duplicate {label} identity")
    return data


def match(left, right, on, label):
    merged = left.merge(right, on=on, how="left", validate="many_to_one", indicator=True)
    require(merged["_merge"].eq("both").all(), f"unmatched {label} identity")
    return merged.drop(columns="_merge")


def vectors(table, prefix, fields):
    return np.column_stack([table[prefix + field] for field in fields])


def fnv1a(text):
    """FNV-1a 64-bit hash of the UTF-8 text as 16 lower-case hex digits (the recorder's manifest hash)."""
    h = 14695981039346656037
    for byte in text.encode():
        h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def canonical(name, field, value):
    """A column in its canonical dtype: int64 identities and counts, bool flags (the recorder writes int32 for both)."""
    a = np.asarray(value)
    if field in BOOL and a.dtype.kind in "iub":
        return a.astype(bool)
    if (field in INT64 or field.endswith("_volume")) and a.dtype.kind in "iu":
        return a.astype(np.int64)
    return a


def _column(key, value):
    import awkward as ak
    if key in VECTORS:
        counts = ak.to_numpy(ak.num(value, axis=1))
        if len(counts) and np.any(counts != counts[0]):
            return value                       # ragged: the audit rejects it explicitly
        return np.asarray(ak.to_numpy(value), dtype=np.float64).reshape(len(counts), -1)
    if str(ak.type(value)).endswith("string"):
        return np.asarray(ak.to_list(value), dtype=str)
    return ak.to_numpy(value)


def read(path, directory=DIRECTORY, names=TABLES):
    """The capture tables of one file (highest ROOT cycles), in canonical dtypes."""
    import uproot
    with uproot.open(path) as root:
        out = {}
        for name in names:
            arrays = root[f"{directory}/{name}"].arrays(library="ak", how=dict)
            out[name] = {key: canonical(name, key, _column(key, value)) for key, value in arrays.items()}
        return out


def _has(path, directory, names):
    import uproot
    with uproot.open(path) as root:
        for name in names:
            try:
                root[f"{directory}/{name}"]
            except KeyError:
                return False
    return True


def read_links(path, directory=DIRECTORY):
    """The links and manifest tables, or None for a capture written without them."""
    return read(path, directory, LINKS) if _has(path, directory, LINKS) else None


def read_runinfo(path, directory=DIRECTORY):
    """The recorder's runinfo table (run, subrun, seed, events, physics list, labels, GDML file name), or None."""
    return read(path, directory, ("runinfo",))["runinfo"] if _has(path, directory, ("runinfo",)) else None
