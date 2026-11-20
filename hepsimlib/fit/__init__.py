"""Fitting core for differentiable responses and calibration (spec: docs/calibration.md).

    optimize     damped Gauss-Newton (Fisher scoring) and L-BFGS with fixed parameters, non-finite rollback and
                 per-parameter scale preconditioning
    objectives   least squares, Gaussian and Poisson expectations, scalar NLLs; Jacobians by jax.jacfwd at the
                 highest matmul precision, or by central differences without JAX
    channels     per-channel gain and time offset: closed-form profiling, Schur elimination of the channel blocks,
                 sum-to-zero gauge, Cramér-Rao errors with the nuisances marginalised
    fisher       covariance from the information, Schur complements, marginal covariances, transforms

NumPy and the standard library; JAX is imported only when a JAX Jacobian is asked for (extra ``jax``). A float64 JAX
path needs ``jax_enable_x64``.
"""
from . import channels, fisher, objectives, optimize  # noqa: F401
from ._backend import highest_precision, jacobian  # noqa: F401
from .channels import (ChannelBlocks, ChannelFitResult, ChannelProblem, Observed, blocks, channel_crb,  # noqa: F401
                       fit_channels, gauge, profile_log_gain, profile_offset)
from .fisher import correlation, covariance, errors, marginal_covariance, schur_complement, transform  # noqa: F401
from .objectives import gaussian, least_squares, poisson, scalar  # noqa: F401
from .optimize import FitResult, GNTerms, gauss_newton, lbfgs  # noqa: F401
