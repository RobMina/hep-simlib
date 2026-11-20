"""Expected response of table libraries for differentiable fits (spec: docs/calibration.md, 2.4).

    table   TableResponse: steps (position, energy deposit, time) -> expected photoelectrons and arrival-time
            distributions per channel, deterministic, multilinear between cell centres, host (NumPy float64) or
            device (JAX float32/float64); table_fields describes such a table for hepsimlib.library.store

NumPy and the standard library; JAX only for the device copy.
"""
from . import table  # noqa: F401
from .table import ChannelResponse, Steps, TableResponse, table_fields  # noqa: F401
