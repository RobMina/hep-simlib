"""Shared core of response and interaction libraries (spec: docs/libraries.md).

A library is any tabulated, pooled or fitted artifact that a fast simulation loads and queries in place of a
simulation step: response tables, shower libraries, donor pools, parametrisations, component interaction banks.
Experiment packages keep their own libraries and build them on these pieces:

    hashing     canonical JSON, file sha256, array content hash, short digests
    axes        Axis (key, unit, frame, edges, out-of-domain policy), Grid (row-major cells), NumPy/JAX bin lookup
    domain      OutOfDomain (loud refusal), DomainCounter (declared fallbacks and clips, counted)
    manifest    the manifest schema (provenance, split, seeds, dtype and RNG conventions, validation status)
    store       .npz library files with an embedded or sidecar manifest; load() checks everything it can
    registry    shipped names, registry JSON, release assets and the download cache (sha256 checked each time)
    dtypes      the float32/float64 compute policy and device copies
    provenance  process-wide record of the libraries a run loaded, pinned or unpinned
    fidelity    the validation vocabulary: status levels, metrics, criteria declared before the held-out read,
                validation records, and the bench performance entry

NumPy and the standard library only; the ``*_jax`` helpers import JAX when they are called. The few settings that
differ between experiments (allowed subsystems, the device-float and cache variables, the release URL) come from
the current :mod:`hepsimlib.profile`.
"""
from . import axes, domain, dtypes, fidelity, hashing, manifest, provenance, registry, store  # noqa: F401
from .axes import Axis, Grid  # noqa: F401
from .domain import DomainCounter, OutOfDomain  # noqa: F401
from .registry import Registry  # noqa: F401
from .store import Library, LibraryError, load, save  # noqa: F401
