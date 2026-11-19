# A4: the bank inside Geant4

The second host of a component bank. A3 tests a bank replayed by the Python host (`hepsimlib.component.replay`)
against full Geant4. A4 tests the C++ host: a Geant4 fast-simulation model (`hepsimlib::BankModel`,
`cpp/include/hepsimlib/`) that loads the same `.npz` bank, checks it against the same content hash, and replays it
inside a Geant4 job, so that the rest of the job (here, two downstream calorimeter blocks) stays full Geant4.

Question: does a Geant4 job with the component replaced by its bank give the same downstream response as the full
Geant4 job, and does the C++ host do exactly what the Python host does? Everything is public: the toy module of A3,
two CsI blocks, FTFP_BERT, the seeds, the bank of A3 v2's set A2 and the analysis.

## The host

- **Reader.** `hepsimlib/npz.hh` reads the NumPy `.npz` (zip, stored or deflated, zip64) and its `.npy` arrays;
  `hepsimlib/bank.hh` checks the embedded manifest (kind, declared arrays, dtypes and shapes) and recomputes the
  content hash with the definition of `hepsimlib.library.hashing.content_hash`, then re-runs the structural checks of
  `hepsimlib.component.Library`. A bank that fails any check is refused with every problem listed. Dependencies:
  the C++17 standard library and zlib.
- **Replay core** (Geant4-free, `hepsimlib::Bank`): the same entrance match (species exactly; position, momentum and
  total energy within the bank's state tolerances), the same trial choice (one uniform per entrance,
  `floor(u * matches)` over the matching trials in bank order), the same placement arithmetic and track-ID
  allocation as `replay()` in float64. `hepsimlib-replay` runs it on a file of entrances and uniforms so the Python
  tests can compare it with `replay()` row for row.
- **Geant4 model** (`hepsimlib::BankModel`, a `G4VFastSimulationModel` on the component's envelope region). The
  trigger takes the entering particle's local state from the fast track (the envelope's own frame, so any placement
  works); a particle on an envelope face moving inward is an entrance and counts as a lookup; it triggers only if
  the bank has its exact state. Otherwise Geant4 transports it as usual (refused, counted). On a trigger the model
  draws one trial, hands every deposit row to a user `DepositSink` (global positions and times, receiving volume
  IDs), continues the incoming particle from its recorded exit (or kills it if it did not exit), and creates every
  other exit as a Geant4 secondary at its global position and time. Geant4 owns everything outside the envelope;
  an exit whose recorded direction points back inward (multiple scattering on the boundary-limited last step) is
  transported by Geant4 inside the component and counted.
- **Precision.** Geant4 is double precision; the C++ host is float64 only. The float32 path stays a property of the
  Python and device host (A3 tests both there).

## Design

Geometry: the A3 module (`../a3/slab_module.gdml`) and two CsI blocks fixed to it in its local frame (they move with
the module): `floor`, half sizes (250, 50, 200) mm at (0, -100, 0), 20 mm below the module's -y face, and `wall`,
half sizes (50, 45, 200) mm at (-270, 0, 0), 20 mm beyond its -x face. Vacuum elsewhere. Entrances: the 99 exact
states of A3 v2 (`../a3/campaign_v2.json`), 1 um approach. The bank: `bank_A2.npz` of the A3 v2 run (1,000 trials
per state, recorded at the nominal placement).

| Arm | Module | Placement | Run, seed | Events |
|---|---|---|---|---|
| G4N1 | full Geant4 | nominal | 21, 510000 | 400 per state, 10 shards |
| G4N2 | full Geant4 | nominal | 22, 520000 | 400 per state, 10 shards |
| HYN | bank | nominal | 23, 530000 | 400 per state, 10 shards |
| G4T | full Geant4 | TR | 24, 540000 | 400 per state, 10 shards |
| HYT | bank | TR | 25, 550000 | 400 per state, 10 shards |
| G4D, HYD, PUD | full / bank / full without the fast-simulation process | nominal | 26, 560000 (all three) | 5 per bank state and 50 per refused state, 1 shard |

"Full Geant4" arms carry the fast-simulation process with no model attached, as a production job with the feature
off would; PUD checks that this changes nothing. TR is A3's translated and rotated placement. Each event's random
engine is seeded from (shard seed, event number) (`hepsimlib-hybrid` docs), so two arms with the same run and seed
see the same random stream event by event. The three refused states are not in the bank: mu- at 999 MeV, e- 20 MeV
10 mm off the bank's entrance point, and a 5 MeV photon tilted 10 degrees.

Per event the application writes: the module's total deposit (bank rows plus any Geant4 steps inside the envelope),
for each block its total deposit, the number of particles entering it, the deposit-weighted centroid in the module
frame and the time of its first deposit, whether the model triggered, the trial, the match count and the uniform. For
the first 10 events of each state in shard 0 of HYN and HYT it also writes every deposit row and every exit the model
produced.

## Criteria (declared before any A4 output exists)

`criteria.json`; its git commit and sha256 are the declaration, and the record carries the sha256.

Exact checks:

- **D1 reader**: the C++ loader accepts `bank_A2.npz` and its recomputed content hash equals the manifest's (the CI
  tests also show it refuses a changed array, a missing array and a changed manifest).
- **D2 parity**: for every detailed event, the Python host given the event's logged entrance state, placement and
  uniform finds the same number of matching trials and picks the same trial, and its deposit rows and exits equal
  the model's: integers exactly, real values within 1e-9 (mm, MeV/c, MeV, ns).
- **D3 refusal**: in HYD every event of a bank state triggers once, no event of a refused state triggers, and every
  refused-state event's output row equals G4D's bit for bit (CPU time excepted).
- **D4 inert plumbing**: every G4D row equals PUD's bit for bit (CPU time excepted).
- **D5 coverage**: in HYN and HYT every event triggers once for its incoming particle.

Statistical checks, per comparison: null_g4 (G4N1 vs G4N2), nominal (G4N1 vs HYN), TR (G4T vs HYT), per class (one
class per bank state's species and energy, as A3):

- **shapes**: Bonferroni-corrected minimum p over every tested (class, observable) of the cluster-robust Hotelling
  T^2 of A3 v2 is at least 0.001. A Geant4 event is one cluster; bank events that drew the same trial are one cluster.
- **means**: for the module deposit and each block's deposit, the class furthest from 1 in errors has
  |ratio - 1| <= 3.5 cluster-robust errors.

The null_g4 comparison is the false-failure check of the test and must pass like the others. The nominal comparison
tests the host (the bank's own placement); TR tests the host and the bank's portability together, so its outcome
also depends on A3's. A criterion added or changed after the first read of A4 output is recorded in the notes, and
the status stays "tested, not closed". The CPU time per event of each arm is reported, with no criterion.

## Running

```bash
python validation/a4/campaign.py macros RUN_DIR --bank BANK.npz     # 53 shards
# in RUN_DIR, each shard (one Geant4 thread each):
hepsimlib-hybrid FTFP_BERT G4N1_000.mac
python validation/a4/analyze.py RUN_DIR --output record.json
```
