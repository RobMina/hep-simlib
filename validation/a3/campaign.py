"""A3 campaign shards: one recorder macro per (set, shard), and the shard list a batch system runs (README.md).

    python validation/a3/campaign.py macros OUTDIR [--trials N] [--shards S] [--sets A,B,R2]

writes OUTDIR/<set>_<shard>.mac, copies the GDML next to them, and OUTDIR/shards.json:
[{set, shard, run, subrun, seed, events, macro, output}]. Run each shard from OUTDIR with
``hepsimlib-recorder <physics list> <macro>`` (one Geant4 thread per process). --trials and --shards override the
specification for pilots; the full campaign is campaign.json as committed. Standard library and NumPy only.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(path=HERE / "campaign.json"):
    return json.loads(Path(path).read_text())


def rotation(spec):
    """3x3 rotation (local -> global) of a placement: "identity" or {axis, angle_deg} (Rodrigues)."""
    if spec == "identity":
        return np.eye(3)
    axis = np.asarray(spec["axis"], float)
    axis = axis / np.linalg.norm(axis)
    a = np.deg2rad(float(spec["angle_deg"]))
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    r = np.eye(3) + np.sin(a) * k + (1 - np.cos(a)) * (k @ k)
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-14, rtol=0) and abs(np.linalg.det(r) - 1) < 1e-14
    return r


def states(campaign):
    """[(pdg, kinetic MeV, entrance name, position, unit direction)] in campaign order."""
    out = []
    for pdg, kinetic in campaign["particles"]:
        for e in campaign["entrances"]:
            u = np.asarray(e["direction"], float)
            out.append((int(pdg), float(kinetic), e["name"], np.asarray(e["position"], float), u / np.linalg.norm(u)))
    return out


def macro(campaign, name, shard, trials):
    s = campaign["sets"][name]
    p = campaign["placements"][s["placement"]]
    r = rotation(p["rotation"])
    seed = int(s["seed"]) + shard
    lines = ["/control/verbose 0", "/run/verbose 0",
             f"/recorder/gdml {campaign['component']['gdml']}",
             f"/recorder/envelope {campaign['component']['envelope']}",
             f"/recorder/geometryTag {campaign['component']['geometryTag']}",
             f"/recorder/physicsTag {campaign['physics']['tag']}",
             "/recorder/translation {:.17g} {:.17g} {:.17g} mm".format(*p["translation_mm"]),
             "/recorder/rotation " + " ".join(f"{x:.17g}" for x in r.ravel()),
             f"/recorder/approach {campaign['approach_mm']:.17g} mm",
             f"/recorder/output {name}_{shard:03d}.root",
             f"/recorder/run {int(s['run'])}", f"/recorder/subrun {shard}", f"/recorder/seed {seed}",
             "/run/initialize"]
    events = 0
    for pdg, kinetic, _, x, u in states(campaign):
        lines.append(f"/recorder/entrance {pdg} " + " ".join(f"{v:.17g}" for v in (*x, *u)) + f" {kinetic:.17g} {trials}")
        events += trials
    lines.append(f"/run/beamOn {events}")
    return "\n".join(lines) + "\n", dict(set=name, shard=shard, run=int(s["run"]), subrun=shard, seed=seed,
                                         events=events, macro=f"{name}_{shard:03d}.mac", output=f"{name}_{shard:03d}.root")


def write_macros(out, campaign, *, trials=None, shards=None, sets=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shards = int(shards or campaign["shards_per_set"])
    per_shard = trials if trials is not None else campaign["trials_per_state"] // shards
    if trials is None and per_shard * shards != campaign["trials_per_state"]:
        raise ValueError("trials_per_state must divide into shards_per_set")
    rows = []
    for name in (sets or list(campaign["sets"])):
        for shard in range(shards):
            text, row = macro(campaign, name, shard, per_shard)
            (out / row["macro"]).write_text(text)
            rows.append(row)
    shutil.copy(HERE / campaign["component"]["gdml"], out / campaign["component"]["gdml"])
    (out / "shards.json").write_text(json.dumps(rows, indent=1) + "\n")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("macros")
    m.add_argument("out", type=Path)
    m.add_argument("--campaign", type=Path, default=HERE / "campaign.json")
    m.add_argument("--trials", type=int, help="trials per state per shard (pilot)")
    m.add_argument("--shards", type=int, help="shards per set (pilot)")
    m.add_argument("--sets", help="comma-separated set names (pilot)")
    a = p.parse_args(argv)
    rows = write_macros(a.out, load(a.campaign), trials=a.trials, shards=a.shards,
                        sets=a.sets.split(",") if a.sets else None)
    print(json.dumps({"shards": len(rows), "events": sum(r["events"] for r in rows)}))


if __name__ == "__main__":
    main()
