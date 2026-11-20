"""Likelihood templates for fits that stop before digitization (spec: docs/calibration.md, 2.3).

    detection   detection as a probability: log P(N >= threshold) and log P(N < threshold) for a Poisson expected
                count, stable at both ends, with analytic derivatives; a seen/silent Bernoulli NLL
    censored    the censored per-crossing (seen/silent) likelihood of a log-normal charge and a Gaussian time with
                walk, fitted widths, analytic NumPy gradient
    arrival     order-statistic (k-th arrival) timing with any delay distribution: tabulated, exponential,
                exponential-Gaussian; windowed, unconditional or conditional

Every function takes NumPy (computed in float64, needs SciPy: extra ``likelihood``) or JAX arrays (computed in
their dtype, differentiable), inferred from the inputs.
"""
from . import arrival, censored, detection  # noqa: F401
from .arrival import (ExpGaussDelay, ExponentialDelay, TabulatedDelay, arrival_log_cdf,  # noqa: F401
                      first_arrival_logpdf, first_arrival_nll)
from .censored import CrossingPrediction, Selection, censored_nll, p_selected  # noqa: F401
from .detection import (d_log_p_detect, d_log_p_miss, detection_nll, log_p_detect, log_p_miss,  # noqa: F401
                        p_detect)
