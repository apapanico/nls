from tqdm import tqdm
import numpy as np

from utils import eig, isotonic_regression,cov,interpolate_zeros,lsq_regularized,nnlsq_regularized
from nls_lw import nls_kfold
from scipy.optimize import lsq_linear

def minvar_vanilla_loo_isotonic(sim, smoothing='average', nonnegative=False,
                                regularization=None):
    """
    Base Estimator 1: MinVar Leave-One-Out Joint Cross-Validation with Isotonic
    Regression

    Variants:
     + Smoothing could be average or median.
     + Nonnegativity constraint
     + Regularization: 'l2' for now, maybe others
    """
    pass


def minvar_vanilla_kfold_isotonic(sim, K, smoothing='average',
                                  nonnegative=False, regularization=None):
    """
    Base Estimator 2: MinVar $K$-Fold Cross-Validation with Isotonic Regression

    Parameters:
     + K

    Variants:
     + Smoothing could be average or median.
     + Nonnegativity constraint
     + Regularization: 'l2' for now, maybe others
    """
    pass


def minvar_joint_loo_isotonic(sim, smoothing='average', nonnegative=False,
                              regularization=None):
    """
    Base Estimator 3: MinVar Leave-One-Out Joint Cross-Validation with Isotonic
    Regression

    Parameters:
     + K

    Variants:
     + Smoothing could be average or median.
     + Nonnegativity constraint
     + Regularization: 'l2' for now, maybe others
    """
    pass


def minvar_joint_kfold_isotonic(sim, K, smoothing='average', nonnegative=False,
                                regularization=None):
    """
    Base Estimator 4: MinVar $K$-Fold Joint Cross-Validation with Isotonic
    Regression

    Parameters:
     + K

    Variants:
     + Smoothing could be average or median.
     + Nonnegativity constraint
     + Regularization: 'l2' for now, maybe others
    """
        
    T, N = sim.shape
    m = int(T / K)
    X,S = sim.X,sim.S
    P = np.zeros((N,N))
    q = np.zeros(N)

    for k in range(K):
        k_set = list(range(k * m, (k + 1) * m))
        _k = np.delete(range(T),k_set)
        X_k = X[k_set, :]
        S_k =  (T * S - X_k.T @ X_k) / (T - m) # 1/(T-m) * X[_k,:].T @ X[_k,:]
        _, U_k = eig(S_k)
        alpha_k = U_k.T @ np.ones(N)
        C_k =  U_k.T @ (1/m * X_k.T @ X_k) @ U_k
        A_k = np.diag(alpha_k)
        P   = P + (A_k @ C_k.T @ C_k @ A_k)
        q   = q + (A_k @ C_k.T @ alpha_k)
        
    if nonnegative:
        z = nnlsq_regularized(P,-q,lmbda=0)
        interpolate_zeros(z)
    else:
        z = np.linalg.solve(P,-q)
    
    d = 1/z
    d = isotonic_regression(d)
    
    return d
