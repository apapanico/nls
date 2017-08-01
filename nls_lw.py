"""
Shrinkage Functions for Testing Nonlinear Shrinkage
"""

from functools import lru_cache
import numpy as np

from cov import nlshrink_covariance
from utils import eig, isotonic_regression


def nls_oracle(sim, isotonic=False, **kwargs):
    """Oracle eigenvalues for LW nonlinear shrinkage"""
    T, N = sim.shape
    U = sim.U
    Sig = sim.Sigma
    d = np.zeros(N)
    for i in range(N):
        u_i = U[:, i]
        d[i] = u_i.T.dot(Sig).dot(u_i)
    d_min, d_max = sim.lam_N, sim.lam_1
    return isotonic_regression(d, y_min=d_min, y_max=d_max) if isotonic else d


def nls_asymptotic(sim, isotonic=False):
    X = sim.X
    S_lw = nlshrink_covariance(X, centered=True)
    d = eig(S_lw, return_eigenvectors=False)
    return isotonic_regression(d) if isotonic else d


def nls_loo(sim, isotonic=False, **kwargs):
    """Leave-One-Out cross-validated eigenvalues for LW nonlinear shrinkage"""
    T, _ = sim.shape
    d = nls_kfold(
        sim, K=T, isotonic=isotonic, **kwargs)
    return d


def nls_kfold(sim, K=10, isotonic=False, **kwargs):
    """K-fold cross-validated eigenvalues for LW nonlinear shrinkage"""
    d = _nls_cv(sim, K)
    d_min, d_max = sim.lam_N, sim.lam_1
    return isotonic_regression(d, y_min=d_min, y_max=d_max) if isotonic else d


@lru_cache(maxsize=32)
def _nls_cv(sim, K):
    T, N = sim.shape
    X, S = sim.X, sim.S
    m = int(T / K)
    d = np.zeros(N)
    for k in range(K):
        k_set = list(range(k * m, (k + 1) * m))
        X_k = X[k_set, :]
        S_k = (T * S - X_k.T.dot(X_k)) / (T - m)
        _, U_k = eig(S_k)
        tmp = (U_k.T.dot(X_k.T)**2).sum(axis=1)
        d += tmp / T
    return d


# Old versions are commented out but left for history for now
# def nls_oracle(X, S, U, Sigma, isotonic=False):
#     """Oracle eigenvalues for LW nonlinear shrinkage"""
#     T, N = X.shape

#     d = np.zeros(N)
#     for i in range(N):
#         u_i = U[:, i]
#         d[i] = u_i.T.dot(Sigma).dot(u_i)
#     return isotonic_regression(d) if isotonic else d


# def nls_loo(X, S, U, isotonic=False):
#     """Leave-One-Out cross-validated eigenvalues for LW nonlinear shrinkage"""
#     T, N = X.shape
#     d = np.zeros(N)
#     for k in range(T):
#         x_k = X[k, :]
#         S_k = (T * S - np.outer(x_k, x_k)) / (T - 1)
#         _, U_k = eig(S_k)
#         d += U_k.T.dot(x_k)**2 / T
#     return isotonic_regression(d) if isotonic else d


# def nls_kfold(X, S, U, K=10, isotonic=False):
#     """K-fold cross-validated eigenvalues for LW nonlinear shrinkage"""
#     T, N = X.shape
#     m = int(T / K)
#     d = np.zeros(N)
#     for k in range(K):
#         k_set = list(range(k * m, (k + 1) * m))
#         X_k = X[k_set, :]
#         S_k = (T * S - X_k.T.dot(X_k)) / (T - m)
#         _, U_k = eig(S_k)
#         tmp = (U_k.T.dot(X_k.T)**2).sum(axis=1)
#         d += tmp / T
#     return isotonic_regression(d) if isotonic else d
