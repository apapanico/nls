from simulations import generate_simple_returns,generate_uniform_volatilities
from slr.utils import *
from shrinkage import nls_kfold_cv,isotonic_regression,minvar_nls_nlsq_new2
from models import *
import numpy as np
import scipy
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import statsmodels.api as sm

inv = np.linalg.inv

def lsq_regularized(A,b,lmbda):
    N = len(b)
    G = np.identity(N) * lmbda
    x = inv(A.T @ A + G.T @ G)  @ A.T @ b
    return x

def nnlsq_regularized(A,b,lmbda):
    ''' Non-negative least squares with regularization'''
    N = len(b)
    G = np.identity(N) * lmbda
    W = np.concatenate((A,G),axis=0)
    f = np.concatenate((b,np.zeros(N)),axis=0)
    
    x = scipy.optimize.nnls(W,f)[0]
    
    return x
    
    
def minvar_nls_oracle_reg(X, S, Sigma,lmbda):
    """Oracle eigenvalues for new MinVar nonlinear shrinkage"""
    # here Sigma us generated with one of the covariance functions from models.py
    
    T, N = X.shape
    L, U = eig(S)
    alpha = U.T.dot(np.ones(N))
    C = U.T.dot(Sigma).dot(U)
    z = nnlsq_regularized(C @ np.diag(alpha), alpha, lmbda)
    #z = lsq_regularized(C @ np.diag(alpha),alpha,lmbda)
    d = 1/z
        
    return U, d



N = 300
T = 1000
seed    = 3910
cov_fun = 'H'

# Sigma - true covariance matrix, tau - true eigenvalues 
Sigma, tau = cov_functions[cov_fun](N) 

np.random.seed(seed)
X = sample(Sigma, T)
S = cov(X)

# sample eigenvalues 
L = eig(S, return_eigenvectors=False)

U,d = minvar_nls_oracle_reg(X,S,Sigma,lmbda=0)

# plotting

plt.plot(annualize_vol(d / N), color='red')
plt.plot(annualize_vol(L / N),color='green')
plt.plot(annualize_vol(tau / N),color='blue')


vleg = mlines.Line2D([], [],
                     color='red',
                     alpha=0.5,
                     markeredgecolor='None',
                     marker='s',
                     linestyle='None',
                     markersize=10,
                     label='Minvar_raw_oracle')



zleg = mlines.Line2D([], [],
                     color='green',
                     alpha=0.5,
                     markeredgecolor='None',
                     marker='s',
                     linestyle='None',
                     markersize=10,
                     label='Sample eigenvalues')

mleg = mlines.Line2D([], [],
                     color='blue',
                     alpha=0.5,
                     markeredgecolor='None',
                     marker='s',
                     linestyle='None',
                     markersize=10,
                     label='True eigenvalues')


plt.legend(handles=[vleg,zleg,mleg],
                   numpoints=1, fancybox=True, framealpha=0.25)