"""
Shrinkage Functions for Testing Nonlinear Shrinkage
"""

from tqdm import tqdm
import numpy as np
from sklearn.isotonic import isotonic_regression as sk_isotonic_regression
# import cvxopt as opt
# import cvxopt.solvers as optsolvers

from cov import nlshrink_covariance
from utils import eig,cov
import scipy as scipy




def nls_oracle(X, S, U, Sigma):
    """Oracle eigenvalues for LW nonlinear shrinkage"""
    T, N = X.shape

    d = np.zeros(N)
    for i in range(N):
        u_i = U[:, i]
        d[i] = u_i.T.dot(Sigma).dot(u_i)
    return U, d


def nls_asymptotic(X, S, U):
    S_lw = nlshrink_covariance(X, centered=True)
    d_lw = eig(S_lw, return_eigenvectors=False)
    return U, d_lw


def nls_loo(X, S, U, progress=False):
    """Leave-One-Out cross-validated eigenvalues for LW nonlinear shrinkage"""
    T, N = X.shape
    d = np.zeros(N)
    if progress:
        pbar = tqdm(total=T)
    for k in range(T):
        x_k = X[k, :]
        S_k = (T * S - np.outer(x_k, x_k)) / (T - 1)
        _, U_k = eig(S_k)
        d += U_k.T.dot(x_k)**2 / T
        if progress:
            pbar.update()
    return U, d


def nls_kfold(X, S, U, K=10, progress=False):
    """K-fold cross-validated eigenvalues for LW nonlinear shrinkage"""
    T, N = X.shape
    m = int(T / K)
    d = np.zeros(N)
    if progress:
        pbar = tqdm(total=K)
    for k in range(K):
        k_set = list(range(k * m, (k + 1) * m))
        X_k = X[k_set, :]
        S_k = (T * S - X_k.T.dot(X_k)) / (T - m)
        _, U_k = eig(S_k)
        tmp = (U_k.T.dot(X_k.T)**2).sum(axis=1)
        d += tmp / T
        if progress:
            pbar.update()
    return U, d


def minvar_nls_oracle(X, S, lam, U, Sigma, isotonic=False, trace=False,
                      upper_bound=True):
    
    """Oracle eigenvalues for new MinVar nonlinear shrinkage"""
    #lam - eigenvalues of S , U - eigenvectors of S
    T, N = X.shape
    alpha = U.T.dot(np.ones(N))
    C = U.T.dot(Sigma).dot(U)
    if isotonic:
        # Solve non-linear problem with monotonicity constraints
        d_min, d_max = lam[-1], lam[0]
        _, d_kfold = nls_kfold(X, S, U)
        d_isokfold = isotonic_regression(d_kfold)
        if trace:
            trace = np.sum(d_isokfold)
            d = minvar_nls_nlsq_multi_transformed(
                [C], [alpha], trace, d_isokfold, d_min, d_max, upper_bound)
        else:
            trace = None
            d = minvar_nls_nlsq_multi(
                [C], [alpha], trace, d_isokfold, d_min, d_max, upper_bound)

    else:
        # Solve plain vanilla linear system
        A = C * alpha
        z = np.linalg.solve(A, alpha)
        d = 1. / z
    return U, d


#### <<<<<<  #####

def minvar_nls_loo(X,S,U):
    
    T,N = X.shape
        
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


#### >>>>>>>>  ####


def minvar_nls_kfold_oracle(X, S, lam, U, Sigma, K, progress=False,
                            trace=False, upper_bound=True):
    """
    Oracle/K-fold cross-validated eigenvalues for new MinVar nonlinea shrinkage.
    """
    T, N = X.shape
    m = int(T / K)

    C_list = []
    alpha_list = []

    if progress:
        pbar = tqdm(total=K)
    for k in range(K):
        k_set = list(range(k * m, (k + 1) * m))
        X_k = X[k_set, :]
        S_k = (T * S - X_k.T.dot(X_k)) / (T - m)
        _, U_k = eig(S_k)

        C = U_k.T.dot(Sigma).dot(U_k)
        C_list.append(C)
        alpha = U_k.T.dot(np.ones(N))
        alpha_list.append(alpha)

        if progress:
            pbar.update()

    d_min, d_max = lam[-1], lam[0]
    _, d_kfold = nls_kfold(X, S, U, K)
    d_isokfold = isotonic_regression(d_kfold)
    if trace:
        trace = np.sum(d_isokfold)
        d = minvar_nls_nlsq_multi_transformed(
            C_list, alpha_list, trace, d_isokfold, d_min, d_max, upper_bound)
    else:
        trace = None
        d = minvar_nls_nlsq_multi(
            C_list, alpha_list, trace, d_isokfold, d_min, d_max, upper_bound)

    return U, d


def minvar_nls_kfold(X, S, lam, U, K, progress=False, trace=False,
                     upper_bound=True):
    """K-fold cross-validated eigenvalues for new MinVar nonlinear shrinkage"""
    T, N = X.shape
    m = int(T / K)

    C_list = []
    alpha_list = []

    if progress:
        pbar = tqdm(total=K)
    for k in range(K):
        k_set = list(range(k * m, (k + 1) * m))
        X_k = X[k_set, :]
        S_k = (T * S - X_k.T.dot(X_k)) / (T - m)
        _, U_k = eig(S_k)

        C = U_k.T.dot(X_k.T.dot(X_k)).dot(U_k)
        C_list.append(C)
        alpha = U_k.T.dot(np.ones(N))
        alpha_list.append(alpha)

        if progress:
            pbar.update()

    d_min, d_max = lam[-1], lam[0]
    _, d_kfold = nls_kfold(X, S, U, K)
    d_isokfold = isotonic_regression(d_kfold)
    if trace:
        trace = np.sum(d_isokfold)
        d = minvar_nls_nlsq_multi_transformed(
            C_list, alpha_list, trace, d_isokfold, d_min, d_max, upper_bound)
    else:
        trace = None
        d = minvar_nls_nlsq_multi(
            C_list, alpha_list, trace, d_isokfold, d_min, d_max, upper_bound)

    return U, d



def isotonic_regression(y, y_min=None, y_max=None):
    """Wrapper around SKlearn's isotonic regression"""
    return sk_isotonic_regression(y, y_min=y_min, y_max=y_max, increasing=False)


def f(d, alpha, C):
    N = len(alpha)
    A = C * alpha.reshape(1, N) * alpha.reshape(N, 1)
    dinv = 1. / d
    T1 = dinv.dot(alpha**2)
    T2 = dinv.T.dot(A.dot(dinv))
    val = .5 * (1 - T2 / T1)**2
    return val


def f_grad(d, alpha, C):
    N = len(alpha)
    A = C * alpha.reshape(1, N) * alpha.reshape(N, 1)
    dinv = 1. / d
    T1 = dinv.dot(alpha**2)
    T2 = dinv.T.dot(A.dot(dinv))

    T1_d = dinv**2 * alpha**2
    T2_d = 2. * dinv**2 * A.dot(dinv)

    val = (1 - T2 / T1) * (T2_d / T1 - T1_d * T2 / T1**2)
    return val

#### <<<<<<  ####



def minvar_nls_oracle_reg(X,S,U,Sigma,lmbda):
    """Oracle eigenvalues for new MinVar nonlinear shrinkage with regularization and non-negative lsq"""
    # here Sigma is generated with one of the covariance functions from models.py
    
    T, N = X.shape
    alpha = U.T.dot(np.ones(N))
    C = U.T.dot(Sigma).dot(U)
    z = nnlsq_regularized(C @ np.diag(alpha), alpha, lmbda)
    #z = lsq_regularized(C @ np.diag(alpha),alpha,lmbda)
    d = 1/z
        
    return U, d

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

#### >>>>>>>  ####





def minvar_nls_nlsq(C, alpha, trace, d0, d_min, d_max, upper_bound=True):
    """Solve an Non-linear LS problem via SLSQP."""

    N = len(alpha)

    def obj(d):
        return f(d, alpha, C)

    def grad(d):
        return f_grad(d, alpha, C)

    if upper_bound:
        bounds = [(d_min, d_max) for _ in range(N)]
    else:
        bounds = [(d_min, None) for _ in range(N)]

    if trace is None:
        trace_con, trace_con_grad = None, None
    else:
        def trace_con(d):
            return np.sum(d) - trace

        g_trace_con = np.ones(N)

        def trace_con_grad(d):
            return g_trace_con

    from scipy.sparse import diags

    G = diags([1, -1], [0, 1], (N - 1, N)).toarray()

    def monoton_con(d):
        return G.dot(d)

    def monoton_con_grad(d):
        return G

    from scipy.optimize.slsqp import fmin_slsqp

    x = fmin_slsqp(
        obj, d0, fprime=grad,
        f_eqcons=trace_con, fprime_eqcons=trace_con_grad,
        f_ieqcons=monoton_con, fprime_ieqcons=monoton_con_grad,
        bounds=bounds, iprint=0)

    return x


def minvar_nls_nlsq_multi(C_list, alpha_list, trace, d0,
                          d_min, d_max, upper_bound):
    """
    Solve an Non-linear LS problem via SLSQP.
    Allows for additive objective function with multiple C matrices and alpha
    vectors.
    """

    N = len(alpha_list[0])

    def obj(d):
        val = sum([
            f(d, alpha, C)
            for C, alpha in zip(C_list, alpha_list)
        ])
        return val

    def grad(d):
        val = sum([
            f_grad(d, alpha, C)
            for C, alpha in zip(C_list, alpha_list)
        ])
        return val

    if upper_bound:
        bounds = [(d_min, d_max) for _ in range(N)]
    else:
        bounds = [(d_min, None) for _ in range(N)]

    if trace is None:
        trace_con, trace_con_grad = None, None
    else:
        def trace_con(d):
            return np.sum(d) - trace

        g_trace_con = np.ones(N)

        def trace_con_grad(d):
            return g_trace_con

    from scipy.sparse import diags

    G = diags([1, -1], [0, 1], (N - 1, N)).toarray()

    def monoton_con(d):
        return G.dot(d)

    def monoton_con_grad(d):
        return G

    from scipy.optimize.slsqp import fmin_slsqp

    x = fmin_slsqp(
        obj, d0, fprime=grad,
        f_eqcons=trace_con, fprime_eqcons=trace_con_grad,
        f_ieqcons=monoton_con, fprime_ieqcons=monoton_con_grad,
        bounds=bounds, iprint=0)

    return x


def G(d):
    return -np.ediff1d(d, to_end=-d[-1])


def Ginv(z, transpose=False):
    if not transpose:
        return np.cumsum(z[::-1])[::-1]
    else:
        return np.cumsum(z)


def minvar_nls_nlsq_multi_transformed(C_list, alpha_list, trace, d0,
                                      d_min, d_max, upper_bound):
    """
    Solve an Non-linear LS problem via SLSQP.
    Allows for additive objective function with multiple C matrices and alpha
    vectors.
    Uses a transformed version of the non-linear optimization problem to get rid
    of the N-1 difference constraints.
    """

    N = len(alpha_list[0])

    rho = 1. / N

    def obj(z):
        d = rho * Ginv(z)
        val = sum([
            f(d, alpha, C)
            for C, alpha in zip(C_list, alpha_list)
        ])
        return val

    def grad(z):
        d = rho * Ginv(z)
        val = sum([
            f_grad(d, alpha, C)
            for C, alpha in zip(C_list, alpha_list)
        ])
        return rho * Ginv(val, transpose=True)

    bounds = [(0., None) for _ in range(N - 1)] + [(d_min / rho, None)]

    if trace is None:
        trace_con, trace_con_grad = None, None
    else:
        v = rho * Ginv(np.ones(N), transpose=True)

        def trace_con(z):
            return v.T.dot(z) - trace

        def trace_con_grad(z):
            return v

    if upper_bound:
        u = rho * np.ones(N)

        def ub_con(z):
            return d_max - u.T.dot(z)

        def ub_con_grad(z):
            return -u
    else:
        ub_con, ub_con_grad = None, None

    from scipy.optimize.slsqp import fmin_slsqp

    z_star = fmin_slsqp(
        obj, d0, fprime=grad,
        f_eqcons=trace_con, fprime_eqcons=trace_con_grad,
        f_ieqcons=ub_con, fprime_ieqcons=ub_con_grad,
        bounds=bounds, iprint=0)

    d_star = rho * Ginv(z_star)
    return d_star


# def isotonic_constraints(N):
#     """Generate isotonic constraints for CVXOPT"""
#     G1 = opt.spmatrix(-1.0, range(N), range(N), (N + 1, N))
#     G2 = opt.spmatrix(1.0, range(1, N + 1), range(N), (N + 1, N))
#     G = G1 + G2
#     h = opt.matrix(-1e-3, (N + 1, 1))
#     return G, h


# def isotonic_lsq(A, b, z_min, z_max, trace=None):
#     """Solve an Isotonic LS problem via a constrained QP."""

#     N = len(A)

#     P = opt.matrix(A.T.dot(A))
#     q = opt.matrix(-A.T.dot(b))

#     # Constraints Gx <= h
#     G, h = isotonic_constraints(N)
#     h[0] = -z_min
#     h[-1] = z_max

#     # print(z_min, h[0], z_max, h[-1])

#     # Constraints Fx = e
#     if trace is not None:
#         F = opt.matrix(1.0, (1, N))
#         e = opt.matrix(trace)
#     else:
#         F = None
#         e = None

#     # Solve
#     optsolvers.options['show_progress'] = False
#     sol = optsolvers.qp(P, q, G, h, F, e)

#     if sol['status'] != 'optimal':
#         warnings.warn("Convergence problem")

#     return np.array(sol['x']).ravel()



# Collect estimators functions
estimators_functions = {'nls_oracle':nls_oracle,'nls_asymptotic':nls_asymptotic,
                        'nls_loo': nls_loo, 'nls_kfold': nls_kfold, 
                        'minvar_nls_oracle': minvar_nls_oracle,
                        'minvar_nls_loo': minvar_nls_loo,
                        'minvar_nls_kfold_oracle': minvar_nls_kfold_oracle,
                        'minvar_nls_kfold': minvar_nls_kfold,
                        'minvar_nls_oracle_reg':minvar_nls_oracle_reg}
