import warnings
import numpy as np
import cvxopt as opt
import cvxopt.solvers as optsolvers


def equal_portfolio(Sigma):
    N = Sigma.shape[0]
    return np.full(N, 1. / N)


def min_var_portfolio(Sigma, leverage=1., gamma=0.0):
    """
    Computes the minimum variance portfolio.
    Note: As the variance is not invariant with respect
    to leverage, it is not possible to construct non-trivial
    market neutral minimum variance portfolios. This is because
    the variance approaches zero with decreasing leverage,
    i.e. the market neutral portfolio with minimum variance
    is not invested at all.

    Parameters
    ----------
    Sigma: pandas.DataFrame
        Covariance matrix of asset returns.
    allow_short: bool, optional
        If 'False' construct a long-only portfolio.
        If 'True' allow shorting, i.e. negative weights.
    Returns
    -------
    weights: pandas.Series
        Optimal asset weights.
    """

    n = len(Sigma)

    P = opt.matrix(Sigma)
    q = opt.matrix(0.0, (n, 1))

    # Constraints Gx <= h
    if gamma is not None:
        # x >= 0
        G = opt.matrix(-np.identity(n))
        h = opt.matrix(gamma, (n, 1))
    else:
        G = None
        h = None

    # Constraints Ax = b
    # sum(x) = 1
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(leverage)

    # Solve
    optsolvers.options['show_progress'] = False
    sol = optsolvers.qp(P, q, G, h, A, b)

    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")

    return np.array(sol['x']).ravel()
