"""A4 campaign shards: one hepsimlib-hybrid macro per (arm, shard), and the shard list a batch system runs (README.md).

    python validation/a4/campaign.py macros OUTDIR --bank BANK.npz [--per-state N] [--shards S] [--arms G4N1,HYN]

writes OUTDIR/<arm>_<shard>.mac, copies the GDML and the bank (as bank.npz) next to them, and OUTDIR/shards.json:
[{arm, shard, run, seed, events, macro, output}]. Run each shard from OUTDIR with
``hepsimlib-hybrid <physics list> <macro>`` (one Geant4 thread per process). --per-state and --shards override the
specification for pilots (events per state per shard; shards per arm); the full campaign is the file as committed.
The states and placements are A3 v2's (../a3/campaign_v2.json). Standard library and NumPy only.
"""
import argparse
import importlib.util
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _a3():
    spec = importlib.util.spec_from_file_location("a3_campaign", HERE.parent / "a3" / "campaign.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


A3 = _a3()


def load(path=HERE / "campaign.json"):
    return json.loads(Path(path).read_text())


def states(campaign):
    """[(pdg, kinetic, position, unit direction, refused)]: the bank states (A3 v2 order), then the refused states."""
    a3 = A3.load(HERE / campaign["states_from"])
    out = [(pdg, kin, list(x), list(u), False) for pdg, kin, _, x, u in A3.states(a3)]
    for s in campaign["refused_states"]:
        out.append((int(s["pdg"]), float(s["kinetic"]), list(s["position"]), list(s["direction"]), True))
    return out


def placement(campaign, name):
    """(rotation local -> global as a 3x3 array, translation in mm) of a named A3 v2 placement."""
    a3 = A3.load(HERE / campaign["placements_from"])
    p = a3["placements"][name]
    return A3.rotation(p["rotation"]), [float(v) for v in p["translation_mm"]]


def macro(campaign, arm, shard, per_state, per_refused):
    s = campaign["arms"][arm]
    r, t = placement(campaign, s["placement"])
    seed = int(s["seed"]) + shard
    gdml = Path(campaign["component"]["gdml"]).name
    lines = ["/control/verbose 0", "/run/verbose 0",
             f"/hybrid/gdml {gdml}", f"/hybrid/envelope {campaign['component']['envelope']}",
             "/hybrid/translation {:.17g} {:.17g} {:.17g} mm".format(*t),
             "/hybrid/rotation " + " ".join(f"{x:.17g}" for x in r.ravel()),
             f"/hybrid/approach {campaign['approach_mm']:.17g} mm",
             f"/hybrid/output {arm}_{shard:03d}.root", f"/hybrid/run {int(s['run'])}", f"/hybrid/seed {seed}",
             f"/hybrid/plumbing {1 if s.get('plumbing', True) else 0}"]
    for c in campaign["catchers"]:
        lines.append(f"/hybrid/catcher {c['name']} {c['material']} " +
                     " ".join(f"{v:.17g}" for v in (*c["half_mm"], *c["centre_mm"])))
    if s["module"] == "bank":
        lines.append("/hybrid/bank bank.npz")
    if shard == 0 and s.get("detail"):
        lines.append(f"/hybrid/detail {int(s['detail'])}")
    lines.append("/run/initialize")
    events = 0
    for pdg, kinetic, x, u, refused in states(campaign):
        count = per_refused if refused else per_state
        if not count:
            continue
        lines.append(f"/hybrid/entrance {pdg} " + " ".join(f"{v:.17g}" for v in (*x, *u)) + f" {kinetic:.17g} {count}")
        events += count
    lines.append(f"/run/beamOn {events}")
    return "\n".join(lines) + "\n", dict(arm=arm, shard=shard, run=int(s["run"]), seed=seed, events=events,
                                         macro=f"{arm}_{shard:03d}.mac", output=f"{arm}_{shard:03d}.root")


def write_macros(out, campaign, bank, *, per_state=None, shards=None, arms=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in (arms or list(campaign["arms"])):
        s = campaign["arms"][arm]
        n = int(shards or s["shards"])
        each = per_state if per_state is not None else int(s["per_state"]) // n
        if per_state is None and each * n != int(s["per_state"]):
            raise ValueError(f"arm {arm}: per_state must divide into its shards")
        refused = int(s.get("per_refused", 0))
        refused_each = refused if per_state is None else (min(refused, per_state) if refused else 0)
        for shard in range(n):
            text, row = macro(campaign, arm, shard, each, refused_each // n if per_state is None else refused_each)
            (out / row["macro"]).write_text(text)
            rows.append(row)
    shutil.copy(HERE / campaign["component"]["gdml"], out / Path(campaign["component"]["gdml"]).name)
    shutil.copy(bank, out / "bank.npz")
    (out / "shards.json").write_text(json.dumps(rows, indent=1) + "\n")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("macros")
    m.add_argument("out", type=Path)
    m.add_argument("--bank", type=Path, required=True)
    m.add_argument("--campaign", type=Path, default=HERE / "campaign.json")
    m.add_argument("--per-state", type=int, help="events per state per shard (pilot)")
    m.add_argument("--shards", type=int, help="shards per arm (pilot)")
    m.add_argument("--arms", help="comma-separated arm names (pilot)")
    a = p.parse_args(argv)
    rows = write_macros(a.out, load(a.campaign), a.bank, per_state=a.per_state, shards=a.shards,
                        arms=a.arms.split(",") if a.arms else None)
    print(json.dumps({"shards": len(rows), "events": sum(r["events"] for r in rows)}))


if __name__ == "__main__":
    main()
