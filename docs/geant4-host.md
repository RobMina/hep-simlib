# A bank inside Geant4 (the C++ host)

A component bank (`hepsimlib.component`, [libraries](libraries.md)) has two hosts: the Python replay
(`hepsimlib.component.replay`, NumPy, float64 and float32) and the C++ host described here, which loads the same
`.npz` file and replays it inside a Geant4 job through the fast-simulation hooks. The validation of the C++ host is
[A4](../validation/a4/README.md).

## Files

| File | What it is | Needs |
|---|---|---|
| `cpp/include/hepsimlib/npz.hh` | `.npz` / `.npy` reader (stored or deflated members, zip64) | zlib |
| `cpp/include/hepsimlib/sha256.hh`, `json.hh` | SHA-256 and a small JSON reader for the manifest | - |
| `cpp/include/hepsimlib/bank.hh` | `hepsimlib::Bank`: load with every check, entrance match, trial choice, float64 replay | zlib |
| `cpp/include/hepsimlib/g4/BankModel.hh` | `hepsimlib::BankModel` (a `G4VFastSimulationModel`), `DepositSink`, `BankPhysics` | Geant4 >= 11.2 |
| `cpp/tools/bank_tools.cc` | `hepsimlib-bank` (check a bank, print its identity) and `hepsimlib-replay` (replay a file of entrances) | zlib |
| `recorder/src/hybrid.cc` | `hepsimlib-hybrid`: the toy job of A4 (component plus blocks, Geant4 or bank) | Geant4 |

All headers are header-only C++17. The recorder's CMake builds the tools and `hepsimlib-hybrid` next to
`hepsimlib-recorder`; the image installs all four.

## Loading

`Bank::Load(path)` reads the file and refuses it, listing every problem, unless:

- the embedded manifest (`__manifest__`) is a replay bank (`kind` "replay") and declares exactly the arrays present,
  with their dtypes and shapes;
- the content hash recomputed with the definition of `hepsimlib.library.hashing.content_hash` (array names sorted;
  for each, its canonical-JSON head and its bytes, length-prefixed) equals the manifest's;
- the tables pass the structural checks of `hepsimlib.component.Library` (offsets, ancestry, root species, entrances
  inward on the envelope, exits on a face, steps inside, non-negative deposits and lengths, ordered times, one exit
  record per exited particle, no step after a particle's first exit).

A bank that loads in Python loads in C++ with the same content hash, and the reverse (tests/test_cpp_bank.py).

## Replay

`Bank::Matches(pdg, position, momentum, energy)` returns the trials whose recorded local entrance equals the state:
species exactly, position within 1e-7 mm, momentum and total energy within 1e-9 + 1e-12 relative, in bank order.
`Bank::Choose(matches, u)` takes `floor(u * n)`. `Bank::Replay` is `hepsimlib.component.replay` in float64 with the
uniforms given: same refusals, trials, track IDs and rows, reals within rounding of the placement arithmetic.

## In a Geant4 job

```cpp
// detector construction: the component's G4Box envelope is the region's root logical volume
auto bank = std::make_shared<const hepsimlib::Bank>(hepsimlib::Bank::Load("bank.npz"));
auto* region = new G4Region("component");
region->AddRootLogicalVolume(envelopeLV);
// ConstructSDandField (one model per thread):
new hepsimlib::BankModel("bank", region, bank, &sink /* a hepsimlib::DepositSink */, volumeIds /* optional */);
// physics list:
physicsList->RegisterPhysics(new hepsimlib::BankPhysics);   // or G4FastSimulationPhysics for the bank's species
```

- **Trigger.** A particle of a bank species on an envelope face moving inward (its state from `G4FastTrack`, in the
  envelope's own frame, so any placement works) is a lookup. If the bank has its exact state the model triggers;
  otherwise Geant4 transports it as usual and the model counts a refusal (`BankModel::Counts()`).
- **DoIt.** One `G4UniformRand()` picks the trial. Every deposit row goes to `DepositSink::Deposit` with global
  positions, momenta and times (entrance time plus the recorded relative time) and the local volume index or
  `volumeIds[local]`; an experiment's sink turns them into its hits. The incoming particle continues from its
  recorded exit (or is killed if it did not exit); every other exit is created as a Geant4 secondary at its global
  position and time, with its recorded momentum.
- **Re-entrant exits.** Geant4 records an exit on the last, boundary-limited step, and multiple scattering can turn
  its direction back inward. Such an exit is created as recorded and Geant4 transports it inside the component
  (`Counts().reentrant`).
- **Precision.** float64 only (Geant4's). The float32 path is a property of the Python and device host.
- **Field.** Banks are recorded field-free; the envelope must sit where the field can be neglected.
