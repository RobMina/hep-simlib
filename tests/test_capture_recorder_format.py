"""What the generic recorder adds over the ported captures: Geant4 ntuple dtypes (int32 identities, int flags) read back
canonical, seeds and the generator recorded from runinfo and the process manifest, and the audit hooks."""
import json

import numpy as np
import pytest
import uproot

from hepsimlib.capture import bank, tables as capture
from hepsimlib.capture.audit import audit
from test_capture_audit import tables  # noqa: F401 -- the shared synthetic capture fixture
from test_capture_bank import bankable
from test_capture_links import make_links

MANIFEST = ("manifest|1\ngeant4|$Name: geant4-11-03-patch-02 $\nphysics|FTFP_BERT|toy-physics\n"
            "geometry|synthetic-box|box\nprocesses|e-|Transportation|msc|eIoni|eBrem\n"
            "cuts|G4_AIR|gamma=0.7mm|e-=0.7mm|e+=0.7mm|proton=0.7mm\nmaterial|G4_AIR|density=0.00120479g/cm3|N=0.755\n")


def as_recorded(table):
    """A table as Geant4's ntuples store it: integers and flags as int32 (no int64 or bool columns)."""
    out = {}
    for field, value in table.items():
        a = np.asarray(value)
        out[field] = a.astype(np.int32) if a.dtype.kind in "iub" and a.ndim == 1 else a
    return out


def write(path, tables, links, runinfo=None):
    with uproot.recreate(path) as root:
        for name, table in {**tables, **links}.items():
            root.mktree("capture/" + name, as_recorded(table))
        if runinfo is not None:
            root.mktree("capture/runinfo", runinfo)
    return path


def runinfo(seed):
    return {"run": np.array([1], np.int32), "subrun": np.array([0], np.int32), "seed": np.array([seed], np.int32),
            "events": np.array([2], np.int32), "physicsList": np.array(["FTFP_BERT"], object),
            "geometryTag": np.array(["synthetic-box"], object), "physicsTag": np.array(["toy-physics"], object),
            "gdml": np.array(["box.gdml"], object)}


def with_manifest(links, text):
    mf = links["manifest"]
    mf["manifest"][:] = text
    mf["manifestHash"][:] = capture.fnv1a(text)
    return links


def shrink(*groups):
    """Identities of 2**53 + k become k: the relations are kept and every identity fits the recorder's int32."""
    for group in groups:
        for table in group.values():
            for field, value in table.items():
                a = np.asarray(value)
                if a.dtype.kind in "iu" and a.ndim == 1 and np.any(a >= 2**53):
                    table[field] = np.where(a >= 2**53, a - 2**53, a)


def test_int32_and_int_flags_read_back_canonical(tmp_path, tables):
    links = make_links(tables)
    shrink(tables, links)
    path = write(tmp_path / "c.root", tables, links)
    loaded, back = capture.read(path), capture.read_links(path)
    assert loaded["tracks"]["track"].dtype == np.int64 and loaded["steps"]["pre_volume"].dtype == np.int64
    assert loaded["steps"]["preInside"].dtype == bool and back["links"]["exitAtBirth"].dtype == bool
    assert loaded["events"]["rotation"].shape == (2, 9) and loaded["events"]["errors"].dtype.kind == "U"
    assert audit(loaded, 2, links=back)["links_checked"]


def test_seeds_and_generator_are_recorded(tmp_path, tables):
    tables, links = bankable(tables)
    shrink(tables, links)
    with_manifest(links, MANIFEST)
    path = write(tmp_path / "capture.root", tables, links, runinfo(12345))
    manifest = bank.convert([path], tmp_path / "bank.npz", 2, relative_to=tmp_path)
    assert manifest["seeds"] == {"capture.root": 12345}
    assert manifest["generator"]["name"] == "Geant4" and manifest["generator"]["physics_list"] == "FTFP_BERT"
    assert manifest["generator"]["release"] == "$Name: geant4-11-03-patch-02 $"
    assert {"seeds", "generator"}.isdisjoint(manifest["unrecorded"])
    sample = manifest["source"]["samples"][0]
    assert sample["seed"] == 12345 and sample["gdml"] == "box.gdml" and "manifest" not in sample


def test_generator_stays_unrecorded_without_cuts_and_materials(tmp_path, tables):
    tables, links = bankable(tables)
    shrink(tables, links)
    with_manifest(links, "manifest|1\ngeant4|x\nphysics|FTFP_BERT|toy\n")
    path = write(tmp_path / "capture.root", tables, links)
    manifest = bank.convert([path], tmp_path / "bank.npz", 2, relative_to=tmp_path)
    assert {"seeds", "generator"} <= set(manifest["unrecorded"])


def test_audit_hooks(tables):
    calls = []
    audit(tables, 2, checks=[lambda t, l: calls.append((len(t["events"]["event"]), l))])
    assert calls == [(2, None)]

    def framework_rule(t, links):
        raise ValueError("framework status code")
    with pytest.raises(ValueError, match="framework status"):
        audit(tables, 2, checks=[framework_rule])
    tables["steps"]["post_process"][3] = "userResourceKill"
    with pytest.raises(ValueError, match="resource kill"):
        audit(tables, 2, resource_processes=("userResourceKill",))
    assert audit(tables, 2)["events"] == 2


def test_cli(tmp_path, tables, capsys):
    from hepsimlib.capture.__main__ import main
    tables, links = bankable(tables)
    shrink(tables, links)
    path = write(tmp_path / "capture.root", tables, links)
    assert main(["audit", str(path), "--expected-events", "2", "--output", str(tmp_path / "a.json")]) == 0
    assert json.loads((tmp_path / "a.json").read_text())["exits_at_birth"] == 1
    assert main(["nonsense"]) == 2
