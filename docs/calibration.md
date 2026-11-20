# Differentiable response and calibration: one fitting core for every experiment

Fast simulations become calibration tools when their response is a smooth, deterministic function of the
calibration parameters. Examples are per-channel gains and time offsets, light yield, attenuation and, later,
alignment. This document fixes the experiment-neutral pieces that make such fits possible, so that every
experiment's fast simulation uses the same optimizer, uncertainties, likelihoods and tests. The experiment
keeps its detector: its tables, geometry, parameter names and data selection.

The pieces, one PR each, in dependency order:

| Module | What it holds | Depends on |
|---|---|---|
| `hepsimlib.fit` | damped Gauss-Newton and L-BFGS drivers; per-channel gain and time offset profiled in closed form; Schur-complement elimination of the per-channel blocks; Fisher matrix and Cramér-Rao bound; Jacobians | NumPy; JAX optional |
| `hepsimlib.testing` | finite-difference gradient harness; closure helpers (multi-seed pulls, model-mismatch closures); the "expected gradient is zero at truth" check | NumPy; JAX optional |
| `hepsimlib.likelihood` | detection as a probability; the censored seen/silent per-crossing likelihood with fitted widths; first-arrival (order-statistic) timing with any delay distribution | NumPy + SciPy, or JAX |
| `hepsimlib.expected` | the expected-response interface of table libraries: steps in, expected counts and arrival-time distributions per channel out, deterministic, interpolated | `hepsimlib.library`; JAX optional |

The ideas come in part from LUCiD (CIDeR-ML/LUCiD, Apache-2.0; arXiv:2602.24129), a differentiable optical simulator
for water Cherenkov detectors. They are reimplemented here, and no LUCiD code is copied.

## 1. Principles

- **Fit expectations, not samples.** A table library already integrates the optics. Its expected answer
  (expected photoelectrons and an arrival-time distribution per channel) is therefore an exactly deterministic,
  differentiable function of the parameters. Fits stop before digitization. Detection, thresholds and pulse
  finding enter as probabilities of the expected count, never as differentiated discrete code. The sampled chain
  stays the simulation; the expected forward is its mean, and that equality is tested.
- **Likelihood, not weighted residuals.** With a deterministic forward, counts use a Poisson likelihood and
  selections use censored terms. Residual weights taken from data (the Neyman form) are used only when the forward
  is itself Monte Carlo.
- **Nuisances profiled, errors marginalised.** Per-channel gains and offsets are profiled in closed form where
  the model allows it. Their blocks are eliminated by the Schur complement in every step and in the Fisher matrix,
  so the cost grows linearly with the number of channels. Reported uncertainties always have the nuisances
  marginalised.
- **NumPy first, JAX optional.** Every driver runs with NumPy models and analytic or finite-difference Jacobians.
  With JAX, the Jacobians come from `jax.jacfwd` (few parameters, many channels), and matrix products run at
  `precision="highest"`. TF32 on a GPU changes float32 Gauss-Newton gradients and Fisher matrices by factors of a
  few, so the default precision is never used for them.
- **float32 and float64.** Each function that evaluates a model, a likelihood or a response supports both dtypes,
  and the tests run both. The optimizer's own small linear algebra (parameters × parameters) is always float64 on
  the host.
- **Additive.** Nothing existing in hep-simlib changes behaviour. The new modules import nothing
  experiment-specific.

## 2. Interfaces

### 2.1 `hepsimlib.fit`

```python
res = fit.gauss_newton(objective, theta0, *, fixed=None, scale=None, max_iter=100, tol=1e-10, damping=1e-3)
res = fit.lbfgs(objective, theta0, *, fixed=None, scale=None, max_iter=500, history=10, gtol=1e-6)
# objective: value(theta), value_and_grad(theta) -> (nll, grad [p]) and, for GN, terms(theta) -> GNTerms(nll, grad, info)
# res: FitResult(theta, nll, grad, info, converged, iterations, rejected_steps, decrement, history, message)

obj = fit.least_squares(residual_fn, jac="fwd" | "fd" | callable)     # whitened residuals r(theta); fit.gaussian(...)
obj = fit.poisson(expected_fn, observed, jac=...)                      # Fisher scoring on expected counts
obj = fit.scalar(nll_fn, grad="rev" | "fd" | callable)                 # L-BFGS only
J = fit.jacobian(fn, theta, method="fwd" | "rev" | "fd")              # jacfwd under highest precision; FD without JAX

# per-channel nuisances, profiled
log_k = fit.profile_log_gain(channel, observed, expected, n_channels, family="poisson" | "gaussian", weights=None)
o     = fit.profile_offset(channel, observed_t, expected_t, n_channels, weights=None)
shift, centred = fit.gauge(values, weights=None)                       # sum-to-zero; shift goes to the global term

problem = fit.ChannelProblem(channel, n_channels, model, charge=fit.Observed(q, family="poisson"),
                             time=fit.Observed(t, sigma=...), gain=True, offset=True)
res = fit.fit_channels(problem, theta0, *, method="gn" | "lbfgs", profile="closed_form" | "schur",
                       gauge=dict(log_gain=index or None, offset=index or None), gauge_weights=None | "information",
                       fixed=..., scale=...)
# model(theta, aux) -> dict(charge=M [n], time=T [n]); coupled=True: model(theta, aux, log_gain_rows, offset_rows)
# res: theta, log_gain [C], offset [C], their errors, cov_theta, gauge_shift, gauge_nll_change

cov = fit.covariance(info, fixed=None)                                 # inverse of the free block, NaN if singular
cov, nu_err = fit.channel_crb(problem, theta, nu, gauge=...)          # theta covariance, nuisances marginalised
```

- **Gauss-Newton** is Levenberg-Marquardt-damped. A step is solved on the free parameters,
  `(H + lambda diag(H)) d = -g`, in scaled coordinates. It is accepted when the NLL is finite and decreases;
  otherwise it is rolled back and `lambda` is raised. `terms` supplies the Fisher information (Gaussian:
  `J^T J`; Poisson: `J^T diag(1/mu) J`), so Gauss-Newton is Fisher scoring.
- **L-BFGS** uses two-loop recursion with a backtracking line search, rolls back a non-finite trial, and skips
  curvature pairs with `s.y <= 0`. It is for objectives whose information matrix is not available, such as the
  censored likelihood.
- **Hygiene, common to both.** `fixed` freezes parameters at their start values, bit for bit. `scale` is the
  per-parameter preconditioning: the driver works in `u = (theta - theta0) / scale`, with
  `scale="auto"` = `diag(H0)^(-1/2)`. A non-finite value or gradient is never accepted. The history records every
  accepted step.
- **Channel model.** Rows `i` carry a channel `c(i)`, an expected charge `M_i(theta)` and/or an expected time
  `T_i(theta)`. The nuisances are a multiplicative gain `exp(g_c)` on the charge and an additive offset `o_c` on the
  time. In closed form, the gain is `sum Q / sum M` (Poisson) or `sum w Q M / sum w M^2` (Gaussian), and the offset
  is a weighted mean residual.
  - **Profiling.** When charge and time are separable (no term couples `g_c` and `o_c`), the nuisances are
    re-profiled exactly after every `theta` step (variable projection).
  - **Coupled case.** When a term couples them (time walk on the charge), each channel has a 2 × 2 block. The
    step is the full Gauss-Newton step on `(theta, g, o)`, with the per-channel blocks eliminated by the Schur
    complement: `S = H_tt - sum_c H_tc H_cc^-1 H_ct`.
- **Gauge.** If `theta` holds a global scale (or a global time) that is degenerate with the gains (offsets), it is
  frozen during the fit. At the end, `mean(g)` (`mean(o)`) is moved into it, so that the reported channel terms sum
  to zero. Weighted gauges (`gauge(..., weights=...)`) avoid the Jensen bias of an equal-weight mean log gain on
  dim channels.
- **Fisher / CRB.** The covariance of `theta` is `S^-1` (the nuisances marginalised). The nuisance errors and their
  gauge-transformed versions follow from the block formulas in `O(C)`, with no dense `(p + 2C)^2` matrix. The
  tests check them against the dense inverse.

### 2.2 `hepsimlib.testing`

```python
rep = testing.check_gradient(f, x, grad=None, *, directions="axes" | int, rel_step=None, rtol=None, atol=None)
# central differences of a scalar f against grad (or jax.grad(f)); step and tolerances default by dtype;
# rep.max_rel_error, rep.error (per direction, <= 1 passes); rep.ok; assert_gradient(...) raises with the table
rep = testing.check_jacobian(F, x, jac=None, ...)        # same for vector functions (fwd or user Jacobian)
sweep = testing.gradient_sweep(f, x, direction, span, n)  # 1D AD vs FD along a line (kinks, plateaus)

res = testing.closure(simulate, fit, truth, seeds, *, names=())   # simulate(truth, rng) -> data; fit(data) -> (est, err)
res.pulls, res.bias, res.pull_mean, res.pull_std, res.coverage(1.0); res.assert_unbiased(k=3)
res = testing.mismatch_closure(simulate_other, fit, truth, seeds)   # the generator is not the fitted model: reports, never asserts by default
testing.expected_score_zero(nll, truth, *, grad=None, info=None, tol=1e-8)   # alias expected_gradient_zero: NLL of Asimov data
testing.mean_score_zero(score, simulate, truth, seeds, *, k=4)    # the score averages to zero over sampled data
```

Pytest is not required to import `hepsimlib.testing`. The assertion helpers raise `AssertionError` with a readable
table, so any package's tests can call them.

### 2.3 `hepsimlib.likelihood`

Every function takes and returns arrays of one backend (`xp` = NumPy or `jax.numpy`, inferred from the inputs), so
JAX callers get autodiff and NumPy callers get values. Analytic NumPy gradients come with the detection
probabilities and the censored likelihood, because those are the ones NumPy users fit with.

```python
likelihood.log_p_detect(mu, threshold)          # log P(N >= threshold), N ~ Poisson(mu), integer threshold; stable at both ends
likelihood.log_p_miss(mu, threshold)            # log P(N < threshold)
likelihood.detection_nll(mu, threshold, seen)   # Bernoulli seen/silent with p = P(N >= threshold)

pred = likelihood.CrossingPrediction(mu_logq, mu_t, log_sigma_q, log_sigma_t, walk, logq_ref)
sel  = likelihood.Selection(log_threshold, t_lo, t_hi)
nll, dpred = likelihood.censored_nll(pred, sel, seen, logq, t, grad=True)   # per-row terms and d/d(pred) [n, 5]

likelihood.first_arrival_logpdf(t, t0, mu, delay, k=1)     # k-th of a Poisson(mu) number of arrivals at t0 + delay
likelihood.first_arrival_nll(t, t0, mu, delay, k=1, window=None, seen=None)   # with censoring outside the window
delay = likelihood.TabulatedDelay(edges, pdf)    # piecewise-constant density, piecewise-linear CDF
delay = likelihood.ExponentialDelay(tau);  likelihood.ExpGaussDelay(tau, sigma)    # analytic CDFs
```

- **Censored crossing** (seen or silent per expected crossing). The log charge is Normal with mean `mu_logq` and
  width `sigma_q`. The time given the charge is Normal with mean `mu_t - walk (logq - logq_ref)` and width
  `sigma_t`. A crossing is selected if `logq >= log_threshold` and `t_lo <= t <= t_hi`. A seen crossing contributes
  the joint density; a silent one contributes `log P(miss)`. `(logq, t)` is bivariate normal, so `P(miss)` is
  `Phi(a)` plus two bivariate-normal orthant probabilities (Genz's algorithm), exact at double precision with a
  closed-form gradient. The predictors come from any forward: an analytic
  surrogate today, `hepsimlib.expected` tomorrow. The widths are parameters, never constants.
- **Order statistic.** For a Poisson number of arrivals with mean `mu` and arrival-time CDF `F`, the k-th arrival
  has `P(T_k <= t) = P(N(t) >= k)`, where `N(t) ~ Poisson(mu F(t - t0))`. Its density is
  `mu f(t - t0) Pois(k - 1; mu F(t - t0))`. `k = 1` is the first arrival; `k > 1` is a photon-count threshold.

### 2.4 `hepsimlib.expected`

```python
table = expected.TableResponse.from_library(lib, yield_array="yield", delay_array="delay_cdf",
                                            position_axes=("x", "y"), delay_axis="delay")
dev = table.to_device(dtype)                                     # float32 or float64 (JAX); table itself is the NumPy host twin
resp = table.expected(steps, *, yield_scale=1.0, channel_scale=None)       # NumPy, float64
resp = dev.expected(steps, yield_scale=..., channel_scale=...)            # JAX: jit-able, differentiable
# steps: Steps(position [n, d] in the table's frame, edep [n], time [n], mask [n] optional)
resp.counts          # [C] expected photoelectrons per channel
resp.cdf(t)          # [..., C] arrival-time CDF (normalised), piecewise linear in t
resp.pdf(t)          # [..., C] its density
resp.binned(edges)   # [C, nbins] expected counts per time bin (exact integral of the CDF)
resp.cumulative(t)   # [..., C] expected photoelectrons arrived by t;  resp.channel(c): a delay object for the likelihood
arrays, fields = expected.table_fields(axes, yield_, delay_edges, delay_cdf)   # describe a table for library.store
```

- **Table.** A `hepsimlib.library` file with position axes (bin edges, as every library has). It holds
  `yield [cells..., C]`, the expected photoelectrons per unit energy deposit at the cell centre, and
  `delay_cdf [cells..., C, T + 1]`, the arrival-delay CDF on the delay-axis edges. The manifest declares
  `interpolation: {method: linear, axes: [...]}`.
- **Interpolation, not nearest bin.** The table is multilinear between cell centres. Inside the outermost half
  cells, the interpolation holds the end value (stated in the manifest). Beyond the edges, the axis `policy`
  applies (`clip`, `refuse` before any work, `nan`). Position gradients are therefore defined everywhere and are
  exact piecewise derivatives.
- **Deterministic.** No random numbers. What is interpolated is the time-resolved expected count
  `yield * F`: a step contributes `edep * sum_corners w yield F(t - t_s)`. A sampled chain that picks each
  photon's corner cell with probability `w * yield` has exactly this mean. The mixture over steps is kept exact.

## 3. What stays experiment-specific

- The tables and their axes, frames and channel numbering: the experiment's libraries, with hep-simlib manifests.
- The physics parameters and how they enter the expected forward: attenuation along a fibre, signal speed,
  SiPM saturation and cross-talk means, light yield per material.
- The selection that defines a crossing or a hit (track association, eligibility, thresholds, windows) and the
  data readers.
- The mapping from fit results to conditions databases.
- The forward's own expected mode (for example a scintillator-strip veto or a crystal calorimeter), built on
  `hepsimlib.expected` and fitted with `hepsimlib.fit` and `hepsimlib.likelihood`.

## 4. Acceptance tests

All synthetic, offline, in float32 and float64 where a dtype enters, NumPy-only and with JAX (skipped without it):

| Module | Test |
|---|---|
| fit | GN and L-BFGS reach the same optimum on least-squares, Poisson and Rosenbrock-like toys; masked parameters stay bit-identical; a step into a non-finite region is rolled back and the fit still converges; the optimum does not depend on `scale` |
| fit | closed-form profiled gains/offsets equal the joint optimum; Schur steps equal the dense GN step; the gauge moves `mean(g)` into the global term without changing the NLL |
| fit | the block covariance and the nuisance errors equal the dense inverse (rtol 1e-8, f64); CRB errors agree with the spread over 200 seeds (pull std within 1 ± 0.15) |
| fit | `jacfwd` at float32 with highest precision agrees with float64 to 1e-5 relative; FD Jacobian without JAX agrees with `jacfwd` |
| testing | the harness passes on correct gradients and fails on a gradient off by 1e-3 relative (f64), in both dtypes; closure pulls are N(0, 1) on a toy; a mismatch closure reports a bias it does not assert |
| likelihood | detection probability equals the direct Poisson sum and is stable at `mu` from 1e-6 to 1e4; censored probabilities integrate to 1 (seen density + miss); analytic gradients match FD to 1e-7 (f64); the order-statistic CDF matches a Monte Carlo of sampled arrivals; the expected gradient is zero at truth |
| expected | interpolation reproduces the nodes and is linear between them; the policy at the edges; counts and binned times equal the mean of sampled photons in a toy detector (closure); float32 and float64 agree; FD of position, yield and channel-scale gradients |

CI runs the suites on Python 3.10-3.14 with JAX where a wheel exists. Each PR also runs them in float32 and
float64 in a container with JAX.
