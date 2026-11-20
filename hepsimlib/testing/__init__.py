"""Test tooling for differentiable paths and calibration fits, for the tests of any package (docs/calibration.md).

    gradcheck   central finite-difference checks of gradients and Jacobians (autodiff or hand-written), dtype-aware
                steps and tolerances, 1D sweeps
    closure     multi-seed closure with pulls, model-mismatch closures, and the zero expected score at the truth

NumPy only; JAX is imported when the reference gradient is ``jax.grad`` / ``jax.jacfwd``. pytest is not needed:
the ``assert_*`` helpers raise AssertionError with a readable table.
"""
from . import gradcheck  # noqa: F401
from .closure import (ClosureResult, MeanScoreReport, ScoreReport, closure, expected_score_zero,  # noqa: F401
                      mean_score_zero, mismatch_closure)
from .gradcheck import (GradientReport, assert_gradient, assert_jacobian, check_gradient, check_jacobian,  # noqa: F401
                        gradient_sweep)

expected_gradient_zero = expected_score_zero
