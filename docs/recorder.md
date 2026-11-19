# The component recorder and the capture tables

`recorder/` holds `hepsimlib-recorder`, a small standalone Geant4 application that records complete trials of one
finite component ([recording component trials](component-training.md) states the contract), and
`hepsimlib.capture` reads, audits, accounts and converts its output into a bank
([component response](component-response.md)). Any other recorder (for example an experiment's own Geant4 framework)
can feed the same Python code by writing the same tables.

```
GDML component  ->  hepsimlib-recorder (Geant4, one thread)  ->  capture file (ROOT ntuples, directory "capture")
                ->  python -m hepsimlib.capture audit | account | bank  ->  bank .npz (hepsimlib.component)
```

## The recorder

- **Geometry**: one logical volume of a GDML file, the *envelope*, which must be a `G4Box` (a field-free box
  envelope, as the banks require). The recorder places it in a `G4_Galactic` world with a chosen translation and
  rotation, so the same component can be recorded nominal, moved and rotated. Replicas and parameterised volumes
  below the envelope are refused.
- **Physics**: any reference physics list of `G4PhysListFactory`, named on the command line; production cuts with
  the usual `/run/setCut` commands.
- **Primaries**: exact local entrance states, each a species, a point on an envelope face, an inward direction and a
  kinetic energy, repeated a given number of times. The primary starts `approach` mm (default 1) behind the face in
  vacuum, so its first step is a trivial approach step and its entrance is a Geant4 boundary crossing. For unstable
  primaries keep it short (e.g. 1 um): a decay in the approach is recorded as an error (the primary never entered).
- **Recording**: every step of every particle inside the envelope (zero-deposit steps included), the primary's
  approach steps, every track's begin, end and entrance states, every stacking classification, the creator step of
  every secondary, and the volume inventory. A track is stopped right after its first exit (the step that leaves the
  envelope is recorded, its children are kept). A secondary born on the envelope surface is owned by the component if
  it points inward and is an *exit at birth* (recorded, killed at stacking) if it points outward; a tangent one is an
  error.
- **Manifest**: per run, the Geant4 version, the physics list, the process lists of the common particles, the
  production cuts of every material-cuts couple, and the composition of every material below the envelope, with its
  FNV-1a 64-bit hash. Seeds are not in the manifest (captures of one configuration with different seeds share it);
  they are in the `runinfo` table.
- **One Geant4 thread per process** (serial run manager): batch jobs scale by running more processes.

Commands (before `/run/initialize` unless noted):

| Command | Meaning |
|---|---|
| `/recorder/gdml FILE` | GDML file with the component |
| `/recorder/envelope NAME` | logical volume of the envelope (a `G4Box`) |
| `/recorder/geometryTag TAG`, `/recorder/physicsTag TAG` | labels the bank keeps; one construction and one physics configuration per tag |
| `/recorder/translation X Y Z UNIT` | envelope position in the world |
| `/recorder/rotation R00 R01 R02 R10 R11 R12 R20 R21 R22` | envelope rotation (local to global, proper orthonormal) |
| `/recorder/output FILE` | capture file (`.root`) |
| `/recorder/run N`, `/recorder/subrun N` | the run and subrun numbers of the event identities |
| `/recorder/seed N` | the random seed (recorded in `runinfo`) |
| `/recorder/approach D UNIT` | how far behind the face the primary starts |
| `/recorder/entrance PDG LX LY LZ UX UY UZ KINETIC_MEV COUNT` | append an exact local entrance state (mm, unit direction, MeV), repeated COUNT times; events take the states in order and `/run/beamOn` must equal the total count (any time before `beamOn`) |

```bash
hepsimlib-recorder FTFP_BERT recorder/examples/slab_module.mac
```

## The capture tables (schema 1)

Every table has the event identity `run, subrun, event` (int64) and an `index` (row in the event). States are
recorded under a prefix (`begin_`, `end_`, `entrance_` for tracks, `pre_`, `post_` for steps, `state_` for
classifications) with the fields `x y z px py pz` (global, mm and MeV/c), `lx ly lz lpx lpy lpz` (in the envelope
frame), `energy kinetic mass charge time properTime weight` (MeV, ns), `volume` (id in `volumes`), `placement`,
`material`, `process` (the process that defined the step, empty if none) and `stepStatus` (`G4StepStatus`).

| Table | Fields |
|---|---|
| `events` | `schema geometryTag physicsTag envelope translation[3] rotation[9] halfSize[3] complete bankAdmissible errors nTracks nSteps nClassifications nVolumes nErrors nG4Tracks nResourceKills cpuTime realTime` |
| `tracks` | `index track parent steps secondaries pdg status reason entered ended creator stopProcess` and the `begin_`, `end_`, `entrance_` states. `reason`: 1 ended inside, 2 first exit, 4 exit at birth. `status`: the `G4TrackStatus` before the recorder's first-exit stop |
| `steps` | `index track number secondaries deposit nonIonizingDeposit length trackStatus preInside postInside firstExit` and the `pre_`, `post_` states |
| `classifications` | `index track parent pdg status killed` and the `state_` state (`status` is the track status when it was stacked) |
| `volumes` | `index id copy name logical motherLogical material solid density translation[3] rotation[9]` (the world, the envelope and everything below it; placements relative to the mother) |
| `links` | `index track parent creatorStep bornOnSurface exitAtBirth`, row-aligned with `tracks` |
| `manifest` | `linksPresent linksIDsMatch schema g4Version manifest manifestHash nLinks nBirthExits`, one row per event |
| `runinfo` | `run subrun seed events physicsList geometryTag physicsTag gdml` (the GDML file name, no directory) |

The recorder writes Geant4's own ROOT ntuples (no ROOT installation needed). They have no 64-bit or boolean columns,
so the reader (`hepsimlib.capture.tables.read`) casts integer identities to int64 and flags to bool; the audit checks
the tables after that.

## The audit, the accounting and the bank

- `audit` (strict, fail-closed): census of every table against the event counts, identities, one primary per event,
  ancestry order and secondary census, classification census, step sequence and state continuity, first-exit rules,
  envelope covariance of every state (global = rotation x local + translation), entrances and exits on the box
  boundary with a geometry-boundary status, trivial outside transport, and, with links, the creator steps,
  exit-at-birth and surface-born rules and the manifest hash. Experiment recorders add their own checks through the
  `checks` argument (e.g. a framework's own status codes).
- `account`: the per-step and per-event energy accounting of [recording component trials](component-training.md).
- `bank`: one trial per event, written with `hepsimlib.component.save_library`. All files of one bank must agree on the
  component (labels, half sizes, volume inventory) and on the manifest hash.

Ported from the Mu2e fast simulation's capture modules (mu2e-fastsim-core a058b04), where they read the Mu2eG4
recorder's output; the Mu2e-only checks (its status codes and kill process) stay there as audit checks.
