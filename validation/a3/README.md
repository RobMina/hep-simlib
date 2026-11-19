# A3: placement covariance of a component bank against full Geant4

The first public validation of `hepsimlib.component`. Question: does a bank recorded with a component at one
placement reproduce full Geant4 when the same component is moved and rotated? Everything here is public: the
component (`slab_module.gdml`, a toy scintillator module that is not any experiment's design), the recorder, the
physics list, the seeds and the analysis.

## Design

| Set | Role | Placement |
|---|---|---|
| A | the bank (recorded once) | nominal |
| B | null: the procedure itself, at the bank's own placement | nominal |
| T | held out | translated by (1234.5, -567.25, 89.125) mm |
| R1 | held out | rotated 90 deg about z |
| R2 | held out | rotated 1 rad about (1, 2, 3) |
| TR | held out | R2's rotation and a translation of (-310, 2048, -777.5) mm |

Every set records the same exact local entrance grid (`campaign.json`): 9 species and energies (mu- 1 GeV, mu+ 200 MeV,
e- 20 MeV, e+ 50 MeV, gamma 5 and 50 MeV, p 500 MeV, pi- 300 MeV, n 50 MeV) x 11 entrance points and directions on
two faces = 99 states, 200 trials each (19,800 events per set, 10 shards of 20 trials per state), each set with its
own seeds. FTFP_BERT, default cuts, Geant4 11.3.2, one thread per shard. Primaries start 1 um behind the face
(`approach_mm`), so an unstable primary practically never decays in the vacuum approach.

**Amendment before the held-out read (2026-11-19).** The first run used a 1 mm approach. One of its 118,800 events
(set TR, shard 1, event 1637: a 300 MeV pi-) decayed in flight in that millimetre of vacuum, so its primary never
entered and the recorder flagged the event; the run stopped at recording and the analysis never read any held-out
set. A 300 MeV pi- decays within 1 mm with probability about 4.4e-5 (beta gamma c tau = 23 m), about 0.6 expected
among the campaign's 13,200 pi- events. The approach is now 1 um (about 6e-4 expected in the campaign). The criteria
are unchanged.

For each set S other than A, the bank of A is replayed at the entrance states of S's primaries, in S's placement,
and compared with S's Geant4 captures in the global frame (`hepsimlib.capture.covariance`): per class (incoming
species and energy), the total deposit, the exit count, the volumes hit, and the exits' species, kinetic energy, time,
position and direction. Replays run in float64 and in float32.

What this tests: the recorder's transforms, the bank's local frame and the replay's placement arithmetic, end to end
against Geant4's own rigid-motion invariance. What it does not test: neighbouring components (exact-state replay cannot
take the continuous exit states of one module as entrances of the next), magnetic fields, or entrance states that are
not in the bank.

## Criteria (declared before any capture exists)

`criteria.json`, committed before the campaign runs; the record carries its sha256. For every set in B, T, R1, R2, TR
and both replay dtypes:

- **shapes**: Bonferroni-corrected minimum p-value of the two-sample chi2 (ROOT "UU") over every tested
  (class, observable) is at least 0.001;
- **mean_edep**: the class whose mean total deposit is furthest from the reference (in errors) has
  |ratio - 1| <= 3.5 independent-sample errors.

B is the null: the same comparison at the bank's own placement, which must pass like the others. A criterion added
or changed after the first held-out read is recorded in the notes, and the status stays "tested, not closed".

## Running

```bash
python validation/a3/campaign.py macros RUN_DIR             # 60 shards, 118,800 events in all
# in RUN_DIR, each shard (any batch system; one Geant4 thread each):
hepsimlib-recorder FTFP_BERT A_000.mac
python validation/a3/analyze.py RUN_DIR --output record.json
```

`campaign.py macros RUN_DIR --trials 3 --shards 1 --sets A,B,R2` is the pilot the CI runs on every push (the chain end
to end on real Geant4 output; too small to test the criteria). Batch sizes come from a measured pilot shard.
