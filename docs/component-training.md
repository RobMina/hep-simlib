# Recording component trials for a bank

A bank ([component response](component-response.md)) is only as good as the Geant4 trials it is built from. This page
states what a recorder must deliver before a bank can be admissible. Every bank written by this package carries
`validation.bank_admissible = false`: passing everything below is necessary for an admissible bank, never sufficient,
and the closing validation (moved, rotated and neighbouring components against full Geant4) is still required.

## Recorder contract

One trial is one incident particle entering one finite component, followed until every particle has ended inside the
component or crossed its envelope for the first time. The recorder writes, per trial:

- **Every in-component step**, zero-deposit steps included: pre/post positions, momenta, total and kinetic energy and
  mass, global and proper times, raw and non-ionising energy deposit, true path length, material and placement
  identity, defining process and step status.
- **Full integer identities**: source, event, track and parent, with the creator step of every secondary.
- **The first boundary crossing** of each track: the exact post-step state (its direction may point back inside after
  multiple scattering), a distinct exit reason, and then stop only that branch. Children already produced are not
  killed because their parent exits.
- **Boundary-born secondaries** follow one ownership rule: owned by the component, an exit at birth, or an error.
- **Terminal states** classified separately: natural stops, decays and captures, tracks with no step, and artificial or
  resource kills. A track stopped by its own energy loss on a boundary-limited step is a natural stop. At-rest physics
  stays enabled.
- **A manifest per run**: Geant4 version, physics list, process lists, cuts and materials, with their hashes.

Any overflow, missing ancestry, unsupported classification or resource kill invalidates the trial; it is never
silently truncated or turned into a physical no-response outcome, and a rejection stops the sample (surviving trials
are never selected). No detector selection, positive-deposit threshold, neutral filter or ancestor-only particle
compression is applied. The boundary is the real finite Geant4 envelope, membership includes daughters through the
physical placement ancestry, and passive parts and interior gaps are part of the component.

## Energy accounting

Energies are total energies, so rest masses are explicit. For each step inside the component,

    R = E_pre - E_post(kept) - deposit - sum(E_birth of the step's children) + medium

where `E_post` is dropped when the step destroys the particle and the medium term is `m_e` times the net electron
number the step takes from the material, plus the rest mass of any recoil nucleus an EM step sets in motion.
Transport and EM steps must close within `rtol = 1e-9` (relative to the pre-step energy) plus `atol = 1e-9` MeV.
Hadronic steps report an explicit nuclear term and are never forced to zero. Each trial must close within `rtol` plus
`atol` per step, counting deposits, first exits, exits at birth, energy left inside and every declared term. Decays
and a few EM steps carry declared terms (each a Geant4 behaviour, not a tuned value), gated step by step at the same
tolerance. Until a bank carries the hash of such an accounting report, its energy accounting stays unrecorded.

## From trials to a bank

- One trial per recorded event; the entrance is the primary's local entrance state. The particle tree is every
  recorded track, root first, parents before children. Steps are the in-component steps (`edep` is the raw deposit)
  with times relative to the entrance; exits are the first exits and the exits at birth.
- All inputs of one bank must agree on the construction and physics labels, the half sizes, the volume inventory below
  the envelope and the Geant4 process manifest.
- `save_library` writes the bank with its manifest. Provenance a caller does not give stays `unrecorded` with a reason;
  the generator stays unrecorded until the recorder supplies physics-list, material and cut hashes.

## Validation order

1. A simple finite box in a vacuum world: exact entry and exit, backscatter, stopping, neutral transmission, empty
   response. A software fixture, not a detector bank.
2. The real component, built from the experiment's construction code, with its placement and material inventory and
   its local sensitive-volume map checked against the source.
3. Geometry covariance: translated and rotated copies under the same local field-free conditions, compared with
   independent full-Geant4 trials (exact-state replay is a software test, not this).
4. Neighbours: explicit transport of recorded exits to another component, returns included as new interactions,
   compared with full Geant4 and checked for double counting.
