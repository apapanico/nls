import sys
import time as clock
import numpy as np
from numpy import *
from numpy.linalg import norm
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from .utils import nostdout

class Term (object):
    def __init__ (self, **kwargs):
        for key, value in kwargs.items():
            setattr (self, key, value)
        #@ for
    #@ def
    
    def check (self, k):
        mit = (k == self.mit)
        r_tol = self.r_norm < self.r_atol + self.r_rtol 
        s_tol = self.s_norm < self.s_atol + self.s_rtol 
        
        return mit or (r_tol and s_tol)
    #@ def
    
    def inter (self, k):
        every_it = not bool (np.mod (k,self.rit))
        print_it = np.isfinite (self.rit)
        
        return (self.check (k) or every_it) and print_it
    #@ def

#@ class


def shrink (X, tau):
    return sign(X)*maximum(abs(X)-tau,0)

def shdiag (X, tau):
    return np.maximum (np.diag (np.diag (X)), tau)

def fro (X):
    return linalg.norm (X, 'fro')


def SLR (W,
        lam = None, 
        gam = None, 
        L0 = None,
        S0 = None,
        mu = None, 
        rit = None,  
        mit = np.inf, 
        atol = None, 
        rtol = None):
    """
    Sparse low rank decomposition using ADMM
    """
    
    N = np.min (W.shape)
    if (N != np.max (W.shape)):
        print ('Input must be a square matrix; exiting!')
        return -1
    
    # quadratic penalty
    mu = N*fro (W) if mu is None else mu
    
    # print results every rit iterations
    rit = 512**2/N if rit is None else rit

    # absolute and relative tolerances
    atol = 1e-15*N if atol is None else atol
    rtol = 1e-15*N*fro (W) if rtol is None else rtol

    # terminating conditions variables
    term = Term (r_norm = np.inf, s_norm = np.inf,
                r_atol = np.sqrt(3.0)*N*atol,
                s_atol = np.sqrt(3.0)*N*atol,
                r_rtol = rtol, s_rtol = rtol,
                mu = mu, mit = mit, rit = rit) 

    # low rank penalty
    lam = fro (W) / N if lam is None else lam
    # sparse penalty
    gam = 1 / np.sqrt (N) if gam is None else gam
    
    # recover diagonal sparse piece if gam <= 0.0
    gam = 0.0 if gam <= 0.0 else gam
    sparse_op = shrink if gam > 0.0 else shdiag

    L = np.zeros ((N,N)) if (L0 is None) else L0
    S = np.zeros ((N,N)) if (S0 is None) else S0

    Q = np.zeros ((N,N))
    V = np.zeros ((N,N))
    Z = np.zeros ((N,N))
    B = np.zeros ((N,N))
     
    OF = np.inf
    k = 0

    while not term.check (k):
        k = k + 1

        Z = Q + L + S - W 

        # Termination checks
        term.r_norm = fro (Z)
        term.s_norm = fro ((Z-B)/mu)
        term.r_rtol = rtol*term.r_norm
        term.s_rtol = rtol*fro (V/mu)
        
        B = Z
        Z = Z / 3 + V

        # Q iteration
        Q = (Q - Z) / (1 + mu)

        # L iteration
        vals, vecs = eigh (L - Z) 
        eigL = shrink (vals, lam*mu)
        L = vecs.dot (diag (eigL)).dot (vecs.T)
 
        # S iteration
        S = sparse_op (S - Z, lam*gam*mu)

        # Update multipliers
        Q_ = L + S - W
        V = V + (Q + Q_) / 3
        #V = Z
        
        # update the objective function
        dF = OF
        NC = np.sum (eigL)
        S1 = np.sum ( np.abs (S))
        OF = lam*(NC + gam*S1) + 0.5*np.sum ((Q_)**2)
        dF = np.abs (OF-dF)

        if term.inter (k):
            
            rank = np.sum (x > 0 for x in eigL)
            
            print ('Iteration ' + str (k) + ' ...')
            print ('N = ' + str (N), end='') 
            print (' lambda = ' + str (lam), end='')
            print (' and gamma = ' + repr (gam))
            print ('Lagrange penalty mu = ' + str(mu))
            print ('dF = ' + str (dF), end='')
            print (' with objective F = ' + str (OF))
            
            print('Primal residual = ' + str (term.r_norm) + 
                ' with tol ' + str (term.r_rtol + term.r_atol))

            print('Dual.p residual = ' + str (term.s_norm) + 
                ' with tol ' + str (term.s_rtol + term.s_atol))
            
   
            print('||S||_1 = ' + str (S1))
            print('||L||_* = ' + str (NC))
            
            eigS_ = np.flipud (eigsh (S, 8, which='LA', 
                return_eigenvectors=False))
            eigL_ = np.flipud (eigL)[0:8]

            np.set_printoptions (precision=4)
            
            print('Top 8 eigenvalues of L:')
            print (eigL_)
            print('Top 8 eigenvalues of S:')
            print (eigS_)
            print('Rank(L) = ' + repr(rank) + '\n\n')
 
    
    st = {}
    st['it'] = int(k)
    st['L'] = L
    st['S'] = S
    st['OF'] = float(OF)
    st['dF'] = float(dF)
    st['term'] = term
    st['rank'] = int(rank)


    return st
  

slr = nostdout(SLR)




