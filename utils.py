"""
Utility Functions for Testing Nonlinear Shrinkage
"""

import numpy as np
from numpy.linalg import eigh, eigvalsh


def sample(Sigma, T):
    """Sample from multivariate normal distribtion"""
    N = Sigma.shape[0]
    X = np.random.multivariate_normal(np.zeros(N), Sigma, T)
    return X


def cov(X):
    """Sample covariance matrix"""
    return np.cov(X, rowvar=False, bias=True)


def eig_multiply(vecs, vals):
    """Build matrix from eigenvalues and eigenvectors"""
    return vecs.dot(np.diag(vals)).dot(vecs.T)


def eig(A, return_eigenvectors=True):
    """Eigenvalue decomposition"""
    if return_eigenvectors:
        vals, vecs = eigh(A)
        return vals[::-1], vecs[:, ::-1]
    else:
        vals = eigvalsh(A)
        return vals[::-1]


def annualize_var(lam):
    """Annualize daily variance"""
    annualize_factor = 256
    return lam * annualize_factor


def annualize_vol(lam):
    """Annualize daily vol"""
    var = annualize_var(lam)
    signs = np.sign(var)
    vols = np.sqrt(np.abs(var)) * signs
    return vols


def portfolio_var(w, C):
    """Annual Portfolio Variance"""
    return annualize_var(w.T.dot(C).dot(w))


def portfolio_vol(w, C):
    """Annual Portfolio Vol"""
    return np.sqrt(portfolio_var(w, C))


def variance_ratio(w, C0, C1):
    """Portfolio Variance Ratio"""
    return portfolio_var(w, C0) / portfolio_var(w, C1)


def true_variance_ratio(w, w0, C0):
    """True Portfolio Variance Ratio"""
    return portfolio_var(w, C0) / portfolio_var(w0, C0)


def tracking_error(w, w0, C0):
    """Portfolio Variance Ratio"""
    return portfolio_vol(w - w0, C0)
