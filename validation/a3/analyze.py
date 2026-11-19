"""A3 analysis: bank of set A, placement covariance of every other set against it, validation record (README.md).

    python validation/a3/analyze.py DIR --output record.json [--criteria validation/a3/criteria.json]

DIR holds the shard captures and shards.json (campaign.py). Every capture passes the strict linked audit first. The
criteria are read from the declared file and entered in the ValidationRecord before the first held-out read; the
record carries the file's sha256. Needs the ``capture`` extra and scipy.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from hepsimlib import component as cr
from hepsimlib.capture import bank, covariance
from hepsimlib.capture.audit import audit
from hepsimlib.capture.tables import read, read_links
from hepsimlib.library import fidelity, store

HERE = Path(__file__).resolve().parent


def merge(parts):
    """One outcome set from per-shard outcomes (trial indices offset)."""
    out = dict(trials={k: np.concatenate([p["trials"][k] for p in parts]) for k in parts[0]["trials"]},
               hits=np.vstack([p["hits"] for p in parts]), nvol=parts[0]["nvol"])
    offsets = np.cumsum([0] + [len(p["trials"]["edep"]) for p in parts])[:-1]
    out["exits"] = {k: np.concatenate([p["exits"][k] + (o if k == "trial" else 0) for p, o in zip(parts, offsets)])
                    for k in parts[0]["exits"]}
    return out


def summary(result):
    return dict(family=result["family"], bonferroni_min_p=result["bonferroni_min_p"], worst=result["worst"],
                worst_mean_edep=result["worst_mean_edep"], untested=len(result["untested"]),
                mean_edep=result["mean_edep"])


def run(directory, output, criteria_path=HERE / "criteria.json"):
    """Bank, comparisons and validation record for the criteria file (v1: chi2 "UU" against the bank of A for
    every set; v2: the cluster-robust statistic, the bank set and the comparisons the file names)."""
    directory = Path(directory)
    criteria_bytes = Path(criteria_path).read_bytes()
    spec = json.loads(criteria_bytes)
    method = spec.get("method", "chi2_uu")
    bank_set = spec.get("bank", "A")
    comparisons = spec.get("comparisons") or [dict(name=s, reference=s, candidate="replay") for s in spec["sets"]]
    shards = json.loads((directory / "shards.json").read_text())
    by_set = defaultdict(list)
    for row in shards:
        by_set[row["set"]].append(row)
    bank_path = directory / f"bank_{bank_set}.npz"
    rows = by_set[bank_set]
    events = {r["events"] for r in rows}
    if len(events) != 1:
        raise ValueError("the bank's shards must have the same number of events")
    if not bank_path.exists():
        bank.convert([directory / r["output"] for r in rows], bank_path, events.pop(), relative_to=directory,
                     configuration=spec.get("campaign", "a3"))
    library = cr.load_library(bank_path)
    record = fidelity.ValidationRecord(library=store.load(bank_path).provenance(),
                                       notes=f"criteria file sha256 {hashlib.sha256(criteria_bytes).hexdigest()}; "
                                             f"method {method}")
    comparisons = [c for c in comparisons if c["reference"] in by_set
                   and (c["candidate"] == "replay" or c["candidate"] in by_set)]
    for c in comparisons:
        for k in spec["criteria"]:
            record.declare(fidelity.Criterion(name=f"{k['name']}:{c['name']}", observable=k["observable"],
                                              metric=k["metric"], quantity=k["quantity"], rule=k["rule"],
                                              target=k.get("target", 0.0), tol=k.get("tol", 0.0), k=k.get("k", 2.0),
                                              held_out=c["name"]))
    record.read_held_out("comparisons " + ", ".join(c["name"] for c in comparisons))
    threshold = spec["hit_threshold_MeV"]
    g4_cache, replay_cache = {}, {}

    def g4(name):
        if name not in g4_cache:
            parts = []
            for row in sorted(by_set[name], key=lambda r: r["shard"]):
                path = directory / row["output"]
                tables, links = read(path), read_links(path)
                audit(tables, row["events"], links=links)
                parts.append(covariance.from_capture(tables, threshold))
            g4_cache[name] = merge(parts)
        return g4_cache[name]

    def replayed(name, dtype):
        if (name, dtype) not in replay_cache:
            parts = []
            for row in sorted(by_set[name], key=lambda r: r["shard"]):
                path = directory / row["output"]
                tables = read(path)
                rng = np.random.default_rng([spec["replay_seed"], row["run"], row["shard"]])
                parts.append(covariance.from_replay(library, tables, rng, dtype=np.dtype(dtype), threshold=threshold))
            replay_cache[(name, dtype)] = merge(parts)
        return replay_cache[(name, dtype)]

    def compare(reference, candidate):
        if method == "cluster_robust":
            return covariance.compare_robust(reference, candidate, spec["observables"],
                                             min_bin_entries=spec["min_bin_entries"], min_trials=spec["min_trials"])
        return covariance.compare(reference, candidate, spec["observables"], min_entries=spec["min_entries"])

    details = {}
    for c in comparisons:
        details[c["name"]] = {}
        for dtype in ("float64", "float32"):
            if c["candidate"] == "replay":
                result = compare(g4(c["reference"]), replayed(c["reference"], dtype))
            elif dtype == "float64" or "float64" not in details[c["name"]]:
                result = compare(g4(c["reference"]), g4(c["candidate"]))
            details[c["name"]][dtype] = summary(result)
            record.add(dtype, f"shapes:{c['name']}", dict(metric="placement_covariance",
                                                           bonferroni_min_p=result["bonferroni_min_p"],
                                                           family=result["family"]))
            worst = result["worst_mean_edep"] or {}
            record.add(dtype, f"mean_edep:{c['name']}", dict(metric="placement_covariance", ratio=worst.get("ratio"),
                                                              err=worst.get("err")))
    out = dict(record=record.as_dict(), details=details, bank=str(bank_path.name), method=method,
               criteria_sha256=hashlib.sha256(criteria_bytes).hexdigest())
    Path(output).write_text(json.dumps(out, indent=1, default=str) + "\n")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--criteria", type=Path, default=HERE / "criteria.json")
    a = p.parse_args(argv)
    out = run(a.directory, a.output, a.criteria)
    print(json.dumps({"status": out["record"]["status"], "overall": out["record"]["overall"],
                      **{s: {d: {k: v[k] for k in ("family", "bonferroni_min_p", "worst_mean_edep")}
                             for d, v in per.items()} for s, per in out["details"].items()}}, indent=1, default=str))


if __name__ == "__main__":
    main()
