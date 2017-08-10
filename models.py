"""
Covariance Models for Testing Nonlinear Shrinkage
"""

import numpy as np

from slr.factor import fm_
from utils import eig


########################
# Covariance Models
########################


def SLR_cov(N, K0=4, K1=32, K2=16, seed=23945):
    """SLR Covariance Matrix"""
    signature = dict(N=N, K0=K0, K1=K1, K2=K2, seed=seed)
    fm = fm_(**signature)
    Sigma = fm.covariance()
    tau = eig(Sigma, return_eigenvectors=False)
    return Sigma, tau


def factor_cov(N, K0=4, seed=23945):
    """Conventional Factor model covariance"""
    signature = dict(N=N, K0=K0, K1=0, K2=0, seed=seed)
    fm = fm_(**signature)
    Sigma = fm.covariance()
    tau = eig(Sigma, return_eigenvectors=False)
    return Sigma, tau


def H_cov(N, gam=1000, a=5.6e-05, b=0.015, random_eigs=True):
    """Simple truncated exponential eigenvalue model from notes
    This actually generates the covariance matrix.
    Default parameters attempt to loosely match eigenvalues of SLR.
    """
    if random_eigs:
        x = np.sort(np.random.rand(N))[::-1]
    else:
        x = np.linspace(0, 1, N)[::-1]
    tau = H_inv(x, gam=gam, a=a, b=b)
    Sigma = create_sigma(tau, random_U=True)
    return Sigma, tau

def identity_cov(N):
    """Identity covariance matrix."""
    Sigma = np.identity(N)
    tau = np.ones(N)
    return Sigma, tau


def spike_cov(N, lam_spike=2., lam_bulk=1.):
    """Spike model where top eigenvalue is separated from bulk values"""
    tau = np.full(N, lam_bulk)
    tau[0] = lam_spike
    Sigma = create_sigma(tau, random_U=True)
    return Sigma, tau


def linear_cov(N, lam_1=.16, lam_n=.05):
    """Linear decay model of eigenvalues"""
    tau = np.linspace(lam_1, lam_n, N)
    tau = tau**2 / 256
    Sigma = create_sigma(tau, random_U=True)
    return Sigma, tau
    
def uniform_cov(N,seed=None):
    tau = generate_uniform_volatilities(N,seed=seed)
    Sigma = create_sigma(tau, random_U=False)
    return Sigma,tau


# Collect covariance functions
cov_functions = {'slr': SLR_cov, 'H': H_cov, 'factor': factor_cov,
                 'identity': identity_cov, 'spike': spike_cov,
                 'linear': linear_cov,'uniform':uniform_cov}
                 
                 
########################################
# Helper functions and eigenvalue models
########################################


def generate_uniform_volatilities(N,seed=None):
    
    ''' Generates uniform daily volatilities corresponding to 
    about 5-30% fluctuations a year'''
    
    # N - number of securities
    tdays = 256 
    rng  = np.random.RandomState(seed)
    vols = rng.uniform(((5/100)**2)/tdays,((30/100)**2)/tdays,N)
    vols = np.sort(vols)[::-1]
    
    return vols
 #@def


def create_sigma(tau, random_U=True):
    N = len(tau)
    if random_U:
        U = haar_measure(N)
        sigma = U.dot(np.diag(tau)).dot(U.T)
    else:
        sigma = np.diag(tau)
    return sigma


def haar_measure(N):
    """A Random matrix distributed with Haar measure"""
    z = np.random.randn(N, N)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q, ph, q)
    return q
    
def H_inv(x, gam=1000, a=5.6e-05, b=0.015):
    """Simple truncated exponential eigenvalue model from notes.
    This function computes the inverse of H, used for generating
    eigenvalues with distribution H.
    Default parameters attempt to loosely match eigenvalues of SLR.
    """
    tau = a - np.log(1 - x * (1 - np.exp(-gam * (b - a)))) / gam
    return tau
    

def H(tau, gam=1000, a=5.6e-05, b=0.015):
    """Simple truncated exponential eigenvalue model from notes.
    This function computes H, the EDF.
    Default parameters attempt to loosely match eigenvalues of SLR.
    """
    H = (1 - np.exp(-gam * (tau - a))) / (1 - np.exp(-gam * (b - a)))
    return H


def h(tau, gam=1000, a=5.6e-05, b=0.015):
    """Simple truncated exponential eigenvalue model from notes.
    This function computes h, the density of the EDF.
    Default parameters attempt to loosely match eigenvalues of SLR.
    """
    h = gam * np.exp(-gam * (tau - a)) / (1 - np.exp(-gam * (b - a)))
    return h
