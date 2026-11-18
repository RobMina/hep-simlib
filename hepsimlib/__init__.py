"""hepsimlib: checked response libraries and component interaction banks for fast simulation.

    hepsimlib.library     the library core: manifest, hashes, load checks, axes and domain accounting, registry,
                          dtype policy, run provenance, and the fidelity vocabulary (docs/libraries.md)
    hepsimlib.component   component interaction banks: whole Geant4 trials of a component replayed in any rigid
                          placement, with refusal before drawing outside the recorded states (docs/component-response.md)
    hepsimlib.profile     the per-experiment settings (subsystems, environment variables, release URL)
"""
__version__ = "0.1.0"
