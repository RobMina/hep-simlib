"""Fisher matrix, covariance and the Cramér-Rao bound (docs/calibration.md, 2.1).

The covariance of an efficient estimator is the inverse of the Fisher information (the Cramér-Rao bound). With
nuisance parameters, the covariance of the parameters of interest is the inverse of the Schur complement of the
nuisance block: the nuisances are marginalised, not fixed. :func:`covariance` inverts the free block of a dense
information matrix; :func:`marginal_covariance` does the Schur step for a dense nuisance block. The per-channel
(block-diagonal) case, which scales to many channels, is in :mod:`hepsimlib.fit.channels`.
"""
import numpy as np

from .optimize import _fixed_mask


def covariance(info, fixed=None):
    """Inverse of the free block of ``info`` (Cholesky), zero rows/columns for fixed parameters; all NaN when the
    free block is not positive definite."""
    H = np.asarray(info, np.float64)
    p = H.shape[0]
    free = ~_fixed_mask(fixed, p)
    cov = np.zeros_like(H)
    Hf = H[np.ix_(free, free)]
    try:
        L = np.linalg.cholesky(Hf)
    except np.linalg.LinAlgError:
        return np.full_like(H, np.nan)
    Li = np.linalg.inv(L)
    cov[np.ix_(free, free)] = Li.T @ Li
    return cov


def errors(cov):
    """Standard errors: the square root of the diagonal (NaN where negative or NaN)."""
    d = np.diag(np.asarray(cov, np.float64))
    with np.errstate(invalid="ignore"):
        return np.where(d >= 0, np.sqrt(np.abs(d)), np.nan)


def correlation(cov):
    e = errors(cov)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(cov) / np.outer(e, e)


def schur_complement(H_tt, H_tn, H_nn):
    """``H_tt - H_tn H_nn^-1 H_nt`` (dense nuisance block)."""
    return H_tt - H_tn @ np.linalg.solve(H_nn, H_tn.T)


def marginal_covariance(info, interest):
    """Covariance of the parameters ``interest`` (indices or mask) with every other parameter marginalised:
    the inverse of the Schur complement of the remaining block (equal to the corresponding block of info^-1)."""
    H = np.asarray(info, np.float64)
    m = _fixed_mask(interest, H.shape[0])
    S = schur_complement(H[np.ix_(m, m)], H[np.ix_(m, ~m)], H[np.ix_(~m, ~m)])
    return covariance(S)


def transform(cov, T):
    """Covariance of ``T theta``: ``T cov T^T`` (a gauge or a change of reported parameters)."""
    T = np.asarray(T, np.float64)
    return T @ np.asarray(cov, np.float64) @ T.T
