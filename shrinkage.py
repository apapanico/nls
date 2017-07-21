"""
Shrinkage Functions for Testing Nonlinear Shrinkage
"""

from tqdm import tqdm
import numpy as np
from sklearn.isotonic import isotonic_regression as sk_isotonic_regression
# import cvxopt as opt
# import cvxopt.solvers as optsolvers


from utils import eig


# The optimal covariance matrices aren't really needed in this script.
# Much can be done with just the eigenvalue shrinkers.

# def nls(X, method='loo', isotonic=False, *args):
#     """Optimal LW NLS covariance matrix"""
#     S = cov(X)
#     _methods = {'loo': nls_loo_cv, 'kcv': nls_kfold_cv, 'oracle': nls_oracle}
#     U, d = _methods[method](X, S, *args)
#     if isotonic:
#         d = isotonic_regression(d)
#     Sigma_hat = eig_multiply(U, d)
#     return Sigma_hat


# def minvar_nls(X, method='oracle', isotonic=False, *args):
#     """Optimal MinVar NLS covariance matrix"""
#     T, N = X.shape
#     S = cov(X)
#     _methods = {'oracle': minvar_nls_oracle}
#     U, d = _methods[method](X, S, *args)
#     Sigma_hat = eig_multiply(U, d)
#     return Sigma_hat


def nls_oracle(X, S, Sigma):
    """Oracle eigenvalues for LW nonlinear shrinkage"""
    T, N = X.shape
    _, U = eig(S)

    d = np.zeros(N)
    for i in range(N):
        u_i = U[:, i]
        d[i] = u_i.T.dot(Sigma).dot(u_i)
    return U, d


def nls_loo_cv(X, S, progress=False):
    """Leave-One-Out cross-validated eigenvalues for LW nonlinear shrinkage"""
    _, U = eig(S)
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


def nls_kfold_cv(X, S, K, progress=False):
    """K-fold cross-validated eigenvalues for LW nonlinear shrinkage"""
    _, U = eig(S)
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


def minvar_nls_oracle(X, S, Sigma, isotonic=False, trace=False,
                      upper_bound=True):
    """Oracle eigenvalues for new MinVar nonlinear shrinkage"""
    T, N = X.shape
    L, U = eig(S)
    alpha = U.T.dot(np.ones(N))
    C = U.T.dot(Sigma).dot(U)
    if isotonic:
        # Solve linear system with isotonic constraints
        _, d_kfold = nls_kfold_cv(X, S, 10)
        d_isokfold = isotonic_regression(d_kfold)
        trace = np.sum(d_isokfold) if trace else None
        d_min, d_max = L[-1], L[0]
        d = minvar_nls_nlsq_new2([C], [alpha], trace,
                                 d_isokfold, d_min, d_max, upper_bound)
    else:
        # Solve plain vanilla linear system
        A = C * alpha
        z = np.linalg.solve(A, alpha)
        d = 1. / z
    return U, d


def minvar_nls_kfold_oracle(X, S, Sigma, K, progress=False,
                            trace=False, upper_bound=True):
    """Oracle/K-fold cross-validated eigenvalues for new MinVar nonlinear shrinkage"""
    T, N = X.shape
    L, U = eig(S)
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

    _, d_kfold = nls_kfold_cv(X, S, 10)
    d_isokfold = isotonic_regression(d_kfold)
    trace = np.sum(d_isokfold) if trace else None
    d_min, d_max = L[-1], L[0]
    d = minvar_nls_nlsq_new2(
        C_list, alpha_list, trace, d_isokfold, d_min, d_max, upper_bound)

    return U, d


def minvar_nls_kfold(X, S, K, progress=False, trace=False,
                     upper_bound=True):
    """K-fold cross-validated eigenvalues for new MinVar nonlinear shrinkage"""
    T, N = X.shape
    L, U = eig(S)
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

    _, d_kfold = nls_kfold_cv(X, S, 10)
    d_isokfold = isotonic_regression(d_kfold)
    trace = np.sum(d_isokfold) if trace else None
    d_min, d_max = L[-1], L[0]
    d = minvar_nls_nlsq_new2(
        C_list, alpha_list, trace, d_isokfold, d_min, d_max, upper_bound)

    return U, d


def isotonic_regression(y, y_min=None, y_max=None):
    """Wrapper around SKlearn's isotonic regression"""
    return sk_isotonic_regression(y, y_min=y_min, y_max=y_max, increasing=False)


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

# def minvar_nls_nlsq_old(C, alpha, trace, d0, d_min, d_max, bounds=True):
#     """Solve an Non-linear LS problem via SLSQP."""

#     N = len(alpha)

#     def obj(d):
#         dinv = 1. / d
#         T1 = (alpha**2 * dinv).sum()
#         T2 = (C * dinv.reshape(1, N) * dinv.reshape(N, 1) *
#               alpha.reshape(1, N) * alpha.reshape(N, 1)).sum()
#         f = (1 - T2 / T1)**2
#         return f

#     def grad(d):
#         dinv = 1. / d
#         T1 = (alpha**2 * dinv).sum()
#         T1_d = -alpha**2 * dinv**2
#         T2 = (C * dinv.reshape(1, N) * dinv.reshape(N, 1) *
#               alpha.reshape(1, N) * alpha.reshape(N, 1)).sum()
#         # T2_d = -2 * C.dot(alpha * dinv) * alpha * dinv**2
#         T2_d = -2 * (C * dinv.reshape(1, N) * dinv.reshape(N, 1)**2 *
#                      alpha.reshape(1, N) * alpha.reshape(N, 1)).sum(axis=1)
#         g = 2 * (1 - T2 / T1) * (-T2_d / T1 + T1_d * T2 / T1**2)
#         return g

#     if bounds:
#         bounds = [(d_min, d_max) for _ in range(N)]
#     else:
#         bounds = ()

#     def trace_con(d):
#         return np.sum(d) - trace

#     g_trace_con = np.ones(N)

#     def trace_con_grad(d):
#         return g_trace_con

#     from scipy.sparse import diags

#     G = diags([1, -1], [0, 1], (N - 1, N)).toarray()

#     def monoton_con(d):
#         return G.dot(d)

#     def monoton_con_grad(d):
#         return G

#     from scipy.optimize.slsqp import fmin_slsqp

#     x = fmin_slsqp(
#         obj, d0, fprime=grad,
#         f_eqcons=trace_con, fprime_eqcons=trace_con_grad,
#         f_ieqcons=monoton_con, fprime_ieqcons=monoton_con_grad,
#         bounds=bounds, iprint=0)

#     return x


def minvar_nls_nlsq(C, alpha, trace, d0, d_min, d_max, bounds=True):
    """Solve an Non-linear LS problem via SLSQP."""

    N = len(alpha)

    dinv = np.zeros(N)
    alpha_sq = alpha**2
    A = C * alpha.reshape(1, N) * alpha.reshape(N, 1)
    tmp1 = np.zeros(N)
    tmp2 = np.zeros(N)
    tmp3 = np.zeros(N)
    T1_d = np.zeros(N)
    T2_d = np.zeros(N)
    g = np.zeros(N)

    def obj(d):
        np.divide(1., d, out=dinv)
        T1 = dinv.dot(alpha_sq)
        np.dot(A, dinv, out=tmp1)
        T2 = dinv.T.dot(tmp1)
        f = (1 - T2 / T1)**2
        return f

    def grad(d):
        np.divide(1., d, out=dinv)
        T1 = dinv.dot(alpha_sq)
        np.dot(A, dinv, out=tmp1)
        T2 = dinv.T.dot(tmp1)

        np.power(dinv, 2, out=tmp2)
        np.multiply(alpha_sq, tmp2, out=T1_d)
        np.multiply(tmp1, tmp2, out=T2_d)
        np.multiply(T2_d, 2., out=T2_d)

        np.divide(T2_d, T1, out=tmp1)
        np.multiply(T1_d, T2 / T1**2, out=tmp2)
        np.subtract(tmp1, tmp2, out=tmp3)
        np.multiply(tmp3, 2 * (1 - T2 / T1), out=g)
        return g

    if bounds:
        bounds = [(d_min, d_max) for _ in range(N)]
    else:
        bounds = ()

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
        # f_eqcons=trace_con, fprime_eqcons=trace_con_grad,
        f_eqcons=None, fprime_eqcons=None,
        f_ieqcons=monoton_con, fprime_ieqcons=monoton_con_grad,
        bounds=bounds, iprint=0)

    return x


def minvar_nls_nlsq_new(C_list, alpha_list, trace, d0,
                        d_min, d_max, upper_bound):
    """Solve an Non-linear LS problem via SLSQP."""

    N = len(alpha_list[0])

    dinv = np.zeros(N)
    alpha_sq_list = [alpha**2 for alpha in alpha_list]
    A_list = [
        C * alpha.reshape(1, N) * alpha.reshape(N, 1)
        for C, alpha in zip(C_list, alpha_list)
    ]
    tmp1 = np.zeros(N)
    tmp2 = np.zeros(N)
    tmp3 = np.zeros(N)
    T1_d = np.zeros(N)
    T2_d = np.zeros(N)
    g_tmp = np.zeros(N)
    g = np.zeros(N)

    def obj(d):
        np.divide(1., d, out=dinv)
        f = 0.
        for A, alpha_sq in zip(A_list, alpha_sq_list):
            T1 = dinv.dot(alpha_sq)
            np.dot(A, dinv, out=tmp1)
            T2 = dinv.T.dot(tmp1)
            f += (1 - T2 / T1)**2
        return f

    def grad(d):
        g.fill(0.)
        np.divide(1., d, out=dinv)
        for A, alpha_sq in zip(A_list, alpha_sq_list):
            T1 = dinv.dot(alpha_sq)
            np.dot(A, dinv, out=tmp1)
            T2 = dinv.T.dot(tmp1)

            np.power(dinv, 2, out=tmp2)
            np.multiply(alpha_sq, tmp2, out=T1_d)
            np.multiply(tmp1, tmp2, out=T2_d)
            np.multiply(T2_d, 2., out=T2_d)

            np.divide(T2_d, T1, out=tmp1)
            np.multiply(T1_d, T2 / T1**2, out=tmp2)
            np.subtract(tmp1, tmp2, out=tmp3)
            np.multiply(tmp3, 2 * (1 - T2 / T1), out=g_tmp)
            np.add(g, g_tmp, out=g)
        return g

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


def minvar_nls_nlsq_new2(C_list, alpha_list, trace, d0,
                         d_min, d_max, upper_bound):
    """Solve an Non-linear LS problem via SLSQP."""

    N = len(alpha_list[0])

    dinv = np.zeros(N)
    alpha_sq_list = [alpha**2 for alpha in alpha_list]
    A_list = [
        C * alpha.reshape(1, N) * alpha.reshape(N, 1)
        for C, alpha in zip(C_list, alpha_list)
    ]
    tmp1 = np.zeros(N)
    tmp2 = np.zeros(N)
    tmp3 = np.zeros(N)
    T1_d = np.zeros(N)
    T2_d = np.zeros(N)
    g_tmp = np.zeros(N)
    g = np.zeros(N)

    def G(d):
        return np.ediff1d(d, to_end=d[-1])

    def Ginv(z, transpose=False):
        if not transpose:
            return np.cumsum(z[::-1])[::-1]
        else:
            return np.cumsum(z)

    def obj(z):
        d = Ginv(z)
        np.divide(1., d, out=dinv)
        f = 0.
        for A, alpha_sq in zip(A_list, alpha_sq_list):
            T1 = dinv.dot(alpha_sq)
            np.dot(A, dinv, out=tmp1)
            T2 = dinv.T.dot(tmp1)
            f += (1 - T2 / T1)**2
        return f

    def grad(z):
        d = Ginv(z)
        g.fill(0.)
        np.divide(1., d, out=dinv)
        for A, alpha_sq in zip(A_list, alpha_sq_list):
            T1 = dinv.dot(alpha_sq)
            np.dot(A, dinv, out=tmp1)
            T2 = dinv.T.dot(tmp1)

            np.power(dinv, 2, out=tmp2)
            np.multiply(alpha_sq, tmp2, out=T1_d)
            np.multiply(tmp1, tmp2, out=T2_d)
            np.multiply(T2_d, 2., out=T2_d)

            np.divide(T2_d, T1, out=tmp1)
            np.multiply(T1_d, T2 / T1**2, out=tmp2)
            np.subtract(tmp1, tmp2, out=tmp3)
            np.multiply(tmp3, 2 * (1 - T2 / T1), out=g_tmp)
            np.add(g, g_tmp, out=g)
        return Ginv(g, transpose=True)

    bounds = [(0., None) for _ in range(N - 1)] + [(d_min, None)]

    if trace is None:
        trace_con, trace_con_grad = None, None
    else:
        v = np.arange(1., N + 1.)

        def trace_con(z):
            return v.T.dot(z) - trace

        def trace_con_grad(z):
            return v

    if upper_bound:
        u = -np.ones(N)

        def ub_con(z):
            return d_max + u.T.dot(z)

        def ub_con_grad(z):
            return u
    else:
        ub_con, ub_con_grad = None, None

    from scipy.optimize.slsqp import fmin_slsqp

    z_star = fmin_slsqp(
        obj, d0, fprime=grad,
        f_eqcons=trace_con, fprime_eqcons=trace_con_grad,
        f_ieqcons=ub_con, fprime_ieqcons=ub_con_grad,
        bounds=bounds, iprint=0)

    d_star = Ginv(z_star)
    return d_star
