"""Component captures: read, audit, energy-account and convert recorder output to banks (docs/recorder.md).

    tables      the schema-1 capture tables, reading (uproot) and helpers
    audit       the strict audit (audit, check_links)
    accounting  per-step and per-event energy accounting (report)
    bank        captures to a hepsimlib.component bank (convert, library_from_captures)

Needs the optional dependencies of the ``capture`` extra (pandas; uproot and awkward to read files).

    python -m hepsimlib.capture audit FILE --expected-events N --output audit.json
    python -m hepsimlib.capture account FILE --expected-events N --output accounting.json [--require-closure]
    python -m hepsimlib.capture bank FILE... --expected-events N --output bank.npz [--configuration NAME]
"""
