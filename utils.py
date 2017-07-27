"""
Utility Functions for Testing Nonlinear Shrinkage
"""

import joblib
import numpy as np
from numpy.linalg import eigh, eigvalsh
from sklearn.isotonic import isotonic_regression as sk_isotonic_regression

from portfolios import min_var_portfolio


class Simulation(object):

    def __init__(self, Sigma, T):
        self.Sigma = Sigma
        self.tau, self.V = eig(Sigma)
        self.N = Sigma.shape[0]
        self.T = T
        self.sample()

    def sample(self):
        self.X = X = sample(self.Sigma, self.T)
        self.cov_est()
        self._hash = hash(joblib.hash(X))

    def __hash__(self):
        return self._hash

    def cov_est(self):
        self.S = S = cov(self.X)
        self.lam, self.U = eig(S)

    @property
    def shape(self):
        return self.X.shape

    @property
    def lam_1(self):
        return self.lam[0]

    @property
    def lam_N(self):
        return self.lam[-1]

    @property
    def vols(self, pop=False):
        eigvals = self.tau if pop else self.lam
        return annualize_vol(eigvals / self.N)


def sample(Sigma, T):
    """Sample from multivariate normal distribtion"""
    N = Sigma.shape[0]
    X = np.random.multivariate_normal(np.zeros(N), Sigma, T)
    X = X - X.mean()
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


def isotonic_regression(y, y_min=None, y_max=None):
    """Wrapper around SKlearn's isotonic regression"""
    return sk_isotonic_regression(y, y_min=y_min, y_max=y_max, increasing=False)


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
    """Tracking error"""
    return portfolio_vol(w - w0, C0)


def portfolio_analysis(S, Sigma, gamma, pi_true):
    pi = min_var_portfolio(S, gamma=gamma)
    out = {
        'oos_var': portfolio_var(pi, Sigma),
        'is_var': portfolio_var(pi, S),
        'forecast_var_ratio': variance_ratio(pi, S, Sigma),
        'true_var_ratio': true_variance_ratio(pi, pi_true, Sigma),
        'te': tracking_error(pi, pi_true, Sigma)
    }
    return out


### >>>>>>>>>>>> ####

def generate_simple_returns(vols,T,seed=None):
   
    # vols - list of variances for each security
    # T    - number of samples
    # seed - seed of random number generator
    
    rng  = np.random.RandomState (seed)
    N    = len(vols)
    vcov = np.diag(vols)
    
    R = rng.multivariate_normal (np.zeros(N), vcov, T) 
    
    return R
#@def

def generate_uniform_volatilities(N,seed=None):
    
    # N - number of securities
    tdays = 256 
    rng  = np.random.RandomState(seed)
    vols = rng.uniform(((5/100)**2)/tdays,((30/100)**2)/tdays,N).tolist()   # assume 5-30% fluctuation a year. Those are daily volatilities
    
    return vols
 #@def
 
def sd_annual(x):
     tdays = 256 
     return np.sqrt(x*tdays)*100
#@def
 
 ### <<<<<<<<<<<<<< ####
