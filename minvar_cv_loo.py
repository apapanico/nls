
from simulations import generate_simple_returns,generate_uniform_volatilities
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

eigh = np.linalg.eigh
eigvalsh = np.linalg.eigvalsh

def eig(A, return_eigenvectors=True):
    """Eigenvalue decomposition"""
    if return_eigenvectors:
        vals, vecs = eigh(A)
        return vals[::-1], vecs[:, ::-1]
    else:
        vals = eigvalsh(A)
        return vals[::-1]

def cov(X):
    """Sample covariance matrix"""
    return np.cov(X, rowvar=False, bias=True)

def nls_oracle_cv(X,S):
    
    T,N = X.shape
    
    U = eig(S)[1]
    
    P=np.zeros((N,N))
    q=np.zeros(N)
    
    for k in range(T):
        
        _k  = list(range(T))
        del _k[k]
        
        S_k   = cov(X[_k,:])
        _,U_k = eig(S_k)
        
        Xk      = X[k].reshape(N,1)
        C_k     = U_k.T @ Xk @ Xk.T @ U_k 
        alpha_k = U_k.T @ np.ones(N)
        A_k     = np.diag(alpha_k)
        
        P += A_k @ C_k.T @ C_k @ A_k
        q += - A_k @ C_k.T @ alpha_k
        
    #@for
    
    z = np.linalg.solve(P,-q)
    d = 1/z
    
    return U,d
#@def

# example 

N = 10
T = 1000 
  
vols = range(1,N+1)[::-1]#np.sort(generate_uniform_volatilities(N,seed=1))[::-1]
X    = generate_simple_returns(vols,T)
S    = cov(X)

# output 

_,d_cv = nls_oracle_cv(X,S)

# plot part

plt.plot(d_cv, marker='o',color='red')
plt.plot(vols, marker='o',color='blue')


vleg = mlines.Line2D([], [],
                     color='red',
                     alpha=0.5,
                     markeredgecolor='None',
                     marker='s',
                     linestyle='None',
                     markersize=10,
                     label='Minvar_loo_oracle')

mleg = mlines.Line2D([], [],
                     color='blue',
                     alpha=0.5,
                     markeredgecolor='None',
                     marker='s',
                     linestyle='None',
                     markersize=10,
                     label='True eigenvalues')

plt.legend(handles=[vleg,mleg],
                   numpoints=1, fancybox=True, framealpha=0.25)
