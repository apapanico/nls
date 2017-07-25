"""
Shrinkage Functions for Testing Nonlinear Shrinkage
"""

from functools import lru_cache
import numpy as np
from sklearn.isotonic import isotonic_regression as sk_isotonic_regression
# import cvxopt as opt
# import cvxopt.solvers as optsolvers


from utils import eig


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


def minvar_oracle(sim, monotonicity='constraint',
                  trace=False, upper_bound=True):
    """Oracle eigenvalues for new MinVar nonlinear shrinkage"""
    type = 'oracle'
    mono = (monotonicity == 'constraint')
    d = _minvar_nlsq(sim, None, mono=mono, trace=trace,
                     type=type, upper_bound=upper_bound)

    if monotonicity == 'isotonic':
        d_max = sim.lam_1 if upper_bound else None
        return isotonic_regression(d, y_max=d_max)
    else:
        return d


def minvar_kfold_oracle(sim, K=10, monotonicity='constraint',
                        trace=False, upper_bound=True):
    """
    Oracle/K-fold cross-validated eigenvalues for new MinVar nonlinea shrinkage.
    """
    type = 'kfold_oracle'
    mono = (monotonicity == 'constraint')
    d = _minvar_nlsq(sim, K, mono=mono, trace=trace,
                     type=type, upper_bound=upper_bound)
    if monotonicity == 'isotonic':
        d_max = sim.lam_1 if upper_bound else None
        return isotonic_regression(d, y_max=d_max)
    else:
        return d


def minvar_loo(sim, monotonicity='constraint', trace=False, upper_bound=True):
    """K-fold cross-validated eigenvalues for new MinVar nonlinear shrinkage"""
    T, _ = sim.shape
    d = minvar_kfold(sim, K=T, monotonicity=monotonicity,
                     trace=trace, upper_bound=upper_bound)
    return d


def minvar_kfold(sim, K=10, monotonicity='constraint',
                 trace=False, upper_bound=True):
    """K-fold cross-validated eigenvalues for new MinVar nonlinear shrinkage"""
    type = 'kfold'
    mono = (monotonicity == 'constraint')
    d = _minvar_nlsq(sim, K, mono=mono, trace=trace,
                     type=type, upper_bound=upper_bound)
    if monotonicity == 'isotonic':
        d_max = sim.lam_1 if upper_bound else None
        return isotonic_regression(d, y_max=d_max)
    else:
        return d


@lru_cache(maxsize=32)
def _minvar_nlsq(sim, K, mono=True, upper_bound=True, trace=True, type='kfold'):
    T, N = sim.shape
    X, Sigma, S, U = sim.X, sim.Sigma, sim.S, sim.U
    d_min, d_max = sim.lam_N, sim.lam_1

    d0 = nls_kfold(sim, 10, isotonic=mono)

    if 'kfold' in type:
        m = int(T / K)
        C = []
        alpha = []
        for k in range(K):
            k_set = list(range(k * m, (k + 1) * m))
            X_k = X[k_set, :]
            S_k = (T * S - X_k.T.dot(X_k)) / (T - m)
            _, U_k = eig(S_k)

            if 'oracle' in type:
                tmp = U_k.T.dot(Sigma).dot(U_k)
            else:
                tmp = U_k.T.dot(X_k.T.dot(X_k)).dot(U_k)
            C.append(tmp)
            alpha.append(U_k.T.dot(np.ones(N)))
    else:
        C = [U.T.dot(Sigma).dot(U)]
        alpha = [U.T.dot(np.ones(N))]

    if mono and trace:
        nlsq_solver = minvar_nlsq_multi_transformed
        args = (C, alpha, np.sum(d0), d0, d_min, d_max, upper_bound)
    else:
        nlsq_solver = minvar_nlsq_multi
        trace = np.sum(d0) if trace else None
        args = (C, alpha, trace, d0, d_min, d_max, mono, upper_bound)

    d = nlsq_solver(*args)
    return d


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


# def minvar_nlsq(C, alpha, trace, d0, d_min, d_max, upper_bound=True):
#     """Solve an Non-linear LS problem via SLSQP."""

#     N = len(alpha)

#     def obj(d):
#         return f(d, alpha, C)

#     def grad(d):
#         return f_grad(d, alpha, C)

#     if upper_bound:
#         bounds = [(d_min, d_max) for _ in range(N)]
#     else:
#         bounds = [(d_min, None) for _ in range(N)]

#     if trace is None:
#         trace_con, trace_con_grad = None, None
#     else:
#         def trace_con(d):
#             return np.sum(d) - trace

#         g_trace_con = np.ones(N)

#         def trace_con_grad(d):
#             return g_trace_con

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
#         bounds=bounds, iprint=1)

#     return x


def minvar_nlsq_multi(C, alpha, trace, d0, d_min, d_max,
                      mono, upper_bound):
    """
    Solve an Non-linear LS problem via SLSQP.

    Allows for additive objective function with multiple C matrices and alpha
    vectors.
    """

    K = len(alpha)
    N = len(alpha[0])

    def obj(d):
        val = sum([
            f(d, _alpha, _C)
            for _C, _alpha in zip(C, alpha)
        ]) / K
        return val

    def grad(d):
        val = sum([
            f_grad(d, _alpha, _C)
            for _C, _alpha in zip(C, alpha)
        ]) / K
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

    if mono:
        from scipy.sparse import diags

        G = diags([1, -1], [0, 1], (N - 1, N)).toarray()

        def monoton_con(d):
            return G.dot(d)

        def monoton_con_grad(d):
            return G
    else:
        monoton_con, monoton_con_grad = None, None

    from scipy.optimize.slsqp import fmin_slsqp

    x = fmin_slsqp(
        obj, d0, fprime=grad,
        f_eqcons=trace_con, fprime_eqcons=trace_con_grad,
        f_ieqcons=monoton_con, fprime_ieqcons=monoton_con_grad,
        bounds=bounds, iprint=1, iter=500)

    return x


def G(d):
    return -np.ediff1d(d, to_end=-d[-1])


def Ginv(z, transpose=False):
    if not transpose:
        return np.cumsum(z[::-1])[::-1]
    else:
        return np.cumsum(z)


def minvar_nlsq_multi_transformed(C, alpha, trace, d0,
                                  d_min, d_max, upper_bound):
    """
    Solve an Non-linear LS problem via SLSQP.

    Allows for additive objective function with multiple C matrices and alpha
    vectors.

    Uses a transformed version of the non-linear optimization problem to get rid
    of the N-1 difference constraints.
    """

    K = len(alpha)
    N = len(alpha[0])

    rho = 1.

    def obj(z):
        d = rho * Ginv(z)
        val = sum([
            f(d, _alpha, _C)
            for _C, _alpha in zip(C, alpha)
        ]) / K
        return val

    def grad(z):
        d = rho * Ginv(z)
        val = sum([
            f_grad(d, _alpha, _C)
            for _C, _alpha in zip(C, alpha)
        ]) / K
        return rho * Ginv(val, transpose=True)

    bounds = [(0., None) for _ in range(N - 1)] + [(d_min / rho, None)]

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
        bounds=bounds, iprint=1, iter=2000)

    d_star = rho * Ginv(z_star)
    return d_star


# def monotonic_constraints(N):
#     """Generate monotonicity constraints for CVXOPT"""
#     G1 = opt.spmatrix(-1.0, range(N), range(N), (N + 1, N))
#     G2 = opt.spmatrix(1.0, range(1, N + 1), range(N), (N + 1, N))
#     G = G1 + G2
#     h = opt.matrix(-1e-3, (N + 1, 1))
#     return G, h


# def monotonic_lsq(A, b, z_min, z_max, trace=None):
#     """Solve an Monotonicity LS problem via a constrained QP."""

#     N = len(A)

#     P = opt.matrix(A.T.dot(A))
#     q = opt.matrix(-A.T.dot(b))

#     # Constraints Gx <= h
#     G, h = monotonic_constraints(N)
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
