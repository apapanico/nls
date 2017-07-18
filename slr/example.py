import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

import factor
import permute
import ADMM

from importlib import reload
reload (factor)
reload (ADMM)
reload (permute)

from ADMM import SLR
from factor import fm_
from permute import permuteBlocks

fro = ADMM.fro

def cov2cor (cov, return_std=False):
    cov = np.asanyarray(cov)
    std_ = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std_, std_)
    if return_std:
        return corr, std_
    else:
        return corr

N = 128 # securities
T = 256 # observations

# Factor model with 4 broad factors (1 market, 1 beta + 2 style)
# 32 countries + 16 industries for 48 sparse factors 
fm = fm_(N=N, K0 = 4, K1=32, K2=16)

# Draw 250 observations of security returns
R = fm.R (T, seed = 2)  # change seed for diff sample

# Sample covariance matrix
V = R.T.dot (R) / T

# insert shrunk matrix

# Run SLR ( lam ~ sqrt (N/T) and gam ~ 1/N )
res = SLR (V, lam = 1e-3 * np.sqrt (N / T), # rank penalty
              gam = 6 / N,      # sparsity penalty 
              mu = N * fro (V), # ADMM quad penalty parameter
              rit = 100,        # print every 100 iterations 
              atol=1e-12 * N,   # abs tolerance 
              rtol=1e-12 * N)   # rel tolerance

# our (unshrunk) recovery L + S  (here S contained the diagonal D)
rL = res['L']
rS = res['S']

# Run again on insert shrunk matrix

# True covariances (sparse, low rank, diagonal)
S, L, D = fm.covariances()

# we did not extract the diagonal with SLR so add it on
S = S + D

# permutation first country and then industry
IdK = np.identity(fm.X.shape[1])
perm = permuteBlocks (fm.X, fm.KS, [1,2], IdK)

# permute for better visualization 
eS = perm.dot (S.dot (perm.T))
rS = perm.dot (rS.dot (perm.T))

print ('Relative low-rank error = ' + 
    str (np.abs ((fro (L) - fro (rL))) / fro (L)))
vals, vecs = eigh (L)
print ('Broad factor eigevalues:\n' +
    str ( np.flipud (vals[-fm.K0:]) ))

print ('\n\nRelative sparse error = ' + 
    str (np.abs ((fro (eS) - fro (rS))) / fro (eS)))
vals, vecs = eigh (S)
print ('Top 8 sparse (+D) factor eigevalues: \n' +
    str ( np.flipud (vals[-8:]) ))

# show the permuted matrices
plt.matshow (cov2cor (rS), cmap=plt.cm.gray)
plt.show (block=False)
plt.matshow (cov2cor (eS), cmap=plt.cm.gray)
plt.show ()






