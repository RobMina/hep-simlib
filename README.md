# hep-simlib

Checked response libraries and component interaction banks for HEP fast simulation, in NumPy.

A fast simulation replaces simulation steps with things it loads and looks up: response tables, shower libraries,
donor pools, fitted parametrisations, recorded interactions. `hep-simlib` gives all of them one contract, so that a
library can be checked, cited, distributed and compared the same way, whatever experiment or subsystem owns it:

| Module | What it holds |
|---|---|
| `hepsimlib.library.manifest` | the manifest: what a library answers, its axes and outputs, source samples, generator, release, seeds, geometry, train/held-out split, dtype and RNG conventions, domain, validation status; unknown provenance is `null` with a reason, never invented |
| `hepsimlib.library.store` | `.npz` files with an embedded or sidecar manifest; `load()` checks the schema, the declared arrays, the content hash and the axes, and lists every problem at once |
| `hepsimlib.library.hashing` | one definition each of the file sha256, the array content hash and the short configuration digest |
| `hepsimlib.library.axes`, `domain` | keys with units, frames and out-of-domain policies (`refuse`, `clip`, `nan`, never extrapolation); refusal before any random draw; counted clips and fallbacks; coverage |
| `hepsimlib.library.registry` | shipped names, registry JSON, release assets and a download cache, sha256-checked on every resolution |
| `hepsimlib.library.dtypes` | float32 and float64 for every library, device copies for JAX |
| `hepsimlib.library.provenance` | the libraries a run actually loaded, pinned or not, for its outputs' provenance |
| `hepsimlib.library.fidelity` | one validation vocabulary: status levels from *bit-identical* to *not validated*, paired ratios and shapes, shape tests against a stock-vs-stock null, criteria declared and hashed before the held-out read, the bench performance entry |
| `hepsimlib.component` | component interaction banks: whole Geant4 trials of one finite component (particle tree, deposit steps, first-boundary exits, null outcomes included) replayed exactly in any rigid placement, with a field-free vacuum handoff between placements |
| `hepsimlib.capture` | recorder output to banks: the strict capture audit, the per-step and per-event energy accounting, and the converter (extra `capture`: pandas, uproot, awkward) |
| `recorder/` | `hepsimlib-recorder`, a standalone Geant4 application that records complete trials of any GDML component (a box envelope) in any placement, and an Apptainer definition with Geant4 11.3.2 |
| `hepsimlib.fit` | calibration fits: damped Gauss-Newton (Fisher scoring) and L-BFGS with fixed parameters, non-finite rollback and scale preconditioning; per-channel gains and time offsets profiled in closed form, Schur elimination of the channel blocks, sum-to-zero gauge, Cramér-Rao errors with the nuisances marginalised; `jax.jacfwd` Jacobians at the highest matmul precision, or finite differences without JAX |
| `hepsimlib.testing` | test tooling for differentiable paths and fits: central finite-difference gradient and Jacobian checks with dtype-aware tolerances, 1D gradient sweeps, multi-seed closures with pulls, model-mismatch closures, and the zero expected score at the truth |
| `hepsimlib.expected` | the expected response of table libraries: energy-deposit steps to expected photoelectrons and arrival-time distributions per channel, deterministic, multilinear between cell centres (position gradients for alignment), NumPy float64 or JAX float32/float64 |
| `hepsimlib.profile` | the few per-experiment settings: allowed subsystems, the device-float and cache variables, the release URL, bank names |

Specifications: [libraries](docs/libraries.md), [component response](docs/component-response.md),
[recording component trials](docs/component-training.md), [the recorder and the capture tables](docs/recorder.md), [differentiable response and calibration](docs/calibration.md) (design).

The package depends on NumPy only; the `*_jax` helpers import JAX when called (extra `jax`). It holds no experiment
names, data or libraries: an experiment package installs its own profile once and keeps its libraries in its own
repository.

```python
from hepsimlib import profile
profile.configure(profile.Profile(name="myexp", subsystems=("calo", "tracker", "component"),
                                  dtype_env="MYEXP_DEVICE_FLOAT", cache_env="MYEXP_LIBRARY_CACHE",
                                  cache_dir="~/.cache/myexp/libraries"))

from hepsimlib.library import Axis, store
lib = store.load("calo_response_v2.npz", expect={"name": "calo.response.run1", "version": 2})
print(lib.provenance())
```

```bash
python3 -m hepsimlib.library check FILE [--name NAME] [--version N] [--content-hash HEX]
python3 -m hepsimlib.library describe FILE
python3 -m hepsimlib.library hash FILE
```

## Status

The library core carries every library of the Mu2e fast simulation; each migration onto it gave bit-identical
outputs in float32 and float64 (float64 only where a library has no float32 path). The component banks are software-tested on synthetic banks (rigid
covariance of the whole correlated bundle, repeated trials and visits, empty outcomes, refusal before drawing, volume
mapping, two-component handoff, float32 and float64); their physics validation against full Geant4 on moved and
rotated components is open work ([component response](docs/component-response.md)).

Provenance: extracted from `mu2e-fastsim-core` (commit a058b04) and generalized (see NOTICE).

## Install and test

```bash
python -m pip install -e '.[test,capture]'        # add ,jax for the JAX twins
python -m pytest -q
```

The tests are offline and synthetic. CI (`.github/workflows/ci.yml`) runs them on every push to main and every pull
request, on Python 3.10 to 3.14.
