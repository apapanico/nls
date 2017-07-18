import warnings

import click
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh, eigvalsh
from sklearn.isotonic import isotonic_regression as sk_isotonic_regression
import cvxopt as opt
import cvxopt.solvers as optsolvers

from portfolios import min_var_portfolio
from slr.factor import fm_
from cov import nlshrink_covariance


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


def H_inv(x, gam=1000, a=5.6e-05, b=0.015):
    """Simple truncated exponential eigenvalue model from notes.

    This function computes the inverse of H, used for generating
    eigenvalues with distribution H.

    Default parameters attempt to loosely match eigenvalues of SLR.
    """
    tau = a - np.log(1 - x * (1 - np.exp(-gam * (b - a)))) / gam
    return tau


def identity(n):
    """Identity covariance matrix."""
    Sigma = np.identity(n)
    tau = np.ones(n)
    return Sigma, tau


def spike(n, lam_spike=2., lam_bulk=1.):
    """Spike model where top eigenvalue is separated from bulk values"""
    tau = np.full(n, lam_bulk)
    tau[0] = lam_spike
    Sigma = create_sigma(tau, random_U=True)
    return Sigma, tau


def linear(n, lam_1=.16, lam_n=.05):
    """Linear decay model of eigenvalues"""
    tau = np.linspace(lam_1, lam_n, n)
    tau = tau**2 / 256
    Sigma = create_sigma(tau, random_U=True)
    return Sigma, tau


# Collect covariance functions
_cov_functions = {'slr': SLR_cov, 'H': H_cov, 'factor': factor_cov,
                  'identity': identity, 'spike': spike,
                  'linear': linear}


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


########################
# Helper functions
########################


def sample(Sigma, T):
    """Sample from multivariate normal distribtion"""
    N = Sigma.shape[0]
    X = np.random.multivariate_normal(np.zeros(N), Sigma, T)
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


########################
# Shrinkage Functions
########################

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
        S_k = (T * S - X_k.T.dot(X_k)) / (T - 1)
        _, U_k = eig(S_k)
        tmp = (U_k.T.dot(X_k.T)**2).sum(axis=1)
        d += tmp / T
        if progress:
            pbar.update()
    return U, d


def minvar_nls_oracle(X, S, Sigma, isotonic=False):
    """Oracle eigenvalues for new MinVar nonlinear shrinkage"""
    T, N = X.shape
    L, U = eig(S)
    alpha = U.T.dot(np.ones(N))
    C = U.T.dot(Sigma).dot(U)
    A = C * alpha
    if isotonic:
        # Solve linear system with isotonic constraints
        d_min, d_max = L[-1], L[0]
        z = isotonic_lsq(A, alpha, d_min, d_max, trace=None)
    else:
        # Solve plain vanilla linear system
        z = np.linalg.solve(A, alpha)
    d = 1. / z
    return U, d


def isotonic_regression(y, y_min=None, y_max=None):
    """Wrapper around SKlearn's isotonic regression"""
    return sk_isotonic_regression(y, y_min=y_min, y_max=y_max, increasing=False)


def isotonic_constraints(N):
    """Generate isotonic constraints for CVXOPT"""
    G1 = opt.spmatrix(-1.0, range(N), range(N), (N + 1, N))
    G2 = opt.spmatrix(1.0, range(1, N + 1), range(N), (N + 1, N))
    G = G1 + G2
    h = opt.matrix(0., (N + 1, 1))
    return G, h


def isotonic_lsq(A, b, d_min, d_max, trace=None):
    """Solve an Isotonic LS problem via a constrained QP."""

    N = len(A)

    P = opt.matrix(A.T.dot(A))
    q = opt.matrix(-A.T.dot(b))

    # Constraints Gx <= h
    G, h = isotonic_constraints(N)
    h[0] = -1. / d_max
    h[-1] = 1 / d_min

    # Constraints Fx = e
    if trace is not None:
        F = opt.matrix(1.0, (1, N))
        e = opt.matrix(trace)
    else:
        F = None
        e = None

    # Solve
    optsolvers.options['show_progress'] = False
    sol = optsolvers.qp(P, q, G, h, F, e)

    if sol['status'] != 'optimal':
        warnings.warn("Convergence problem")

    return np.array(sol['x']).ravel()


########################
# CLI
########################


@click.group()
def cli():
    pass


@cli.command()
@click.option('--n', type=int, default=100)
@click.option('--y', type=int, default=2)
@click.option('--cov_fun', default='H')
@click.option('--loo/--no-loo', default=True)
@click.option('--K', type=int, default=10)
@click.option('--ylim', nargs=2, type=float, default=(0., .25))
@click.option('--figsize', nargs=2, type=float, default=(16, 8))
@click.option('--seed', type=int, default=3910)
def demo(n, y, cov_fun, loo, k, ylim, figsize, seed):
    """Simple demo showing the results of the various shrinkage methods"""
    np.random.seed(seed)
    T = y * n
    Sigma, tau = _cov_functions[cov_fun](n)

    X = sample(Sigma, T)
    S = cov(X)
    L = eig(S, return_eigenvectors=False)
    lam_1, lam_N = L[0], L[-1]

    fig, (ax0, ax1) = plt.subplots(figsize=figsize, ncols=2)
    ax0.plot(annualize_vol(tau / n), label='true')
    ax1.plot(annualize_vol(tau / n), label='true')
    ax0.plot(annualize_vol(L / n), label='sample')
    ax1.plot(annualize_vol(L / n), label='sample')

    # Oracle LW NLS shrinkage
    _, d_lw_oracle = nls_oracle(X, S, Sigma)
    d_isolw_oracle = isotonic_regression(d_lw_oracle)
    ax0.plot(annualize_vol(d_lw_oracle / n), label='lw oracle')
    ax1.plot(annualize_vol(d_isolw_oracle / n), label='lw oracle')

    # LW NLS shrinkage
    S_lw = nlshrink_covariance(X, centered=True)
    d_lw = eig(S_lw, return_eigenvectors=False)
    ax1.plot(annualize_vol(d_lw / n), label='lw')

    if loo:
        # LOO LW NLS shrinkage
        _, d_loo = nls_loo_cv(X, S)
        d_isoloo = isotonic_regression(d_loo)
        ax0.plot(annualize_vol(d_loo / n), label='noisy-loo')
        ax1.plot(annualize_vol(d_isoloo / n), label='isoloo')

    # K-fold LW NLS shrinkage
    _, d_kfold = nls_kfold_cv(X, S, k)
    d_isokfold = isotonic_regression(d_kfold)
    ax0.plot(annualize_vol(d_kfold / n), label='noisy-kfold')
    ax1.plot(annualize_vol(d_isokfold / n), label='isokfold')

    # MinVar NLS shrinkage
    _, d_mv_oracle = minvar_nls_oracle(X, S, Sigma)
    d_isomv_oracle = isotonic_regression(d_mv_oracle, y_min=lam_N, y_max=lam_1)
    _, d_isolsq_mv_oracle = minvar_nls_oracle(X, S, Sigma, isotonic=True)
    ax0.plot(annualize_vol(d_mv_oracle / n), label='noisy-mv_oracle')
    ax1.plot(annualize_vol(d_isomv_oracle / n), label='buggy-iso-mv_oracle')
    ax1.plot(annualize_vol(d_isolsq_mv_oracle / n), label='isolsq-mv_oracle')

    ax0.legend()
    ax1.legend()
    ax0.set_ylim(*ylim)
    ax1.set_ylim(*ylim)
    plt.show()


@cli.command()
@click.option('--m', type=int, default=100)
@click.option('--n', type=int, default=100)
@click.option('--y', type=int, default=2)
@click.option('--cov_fun', default='H')
@click.option('--lw/--no-lw', default=False)
@click.option('--loo/--no-loo', default=False)
@click.option('--K', type=int, default=10)
@click.option('--ylim', nargs=2, type=float, default=None)
@click.option('--figsize', nargs=2, type=float, default=(16, 8))
@click.option('--seed', type=int, default=3910)
def eigs(m, n, y, cov_fun, lw, loo, k, ylim, figsize, seed):
    """
    Generate box plots of the eigenvalues from the various shrinkage methods.

    Leave-One-Out cross-validated NLS and the optimal LW NLS are not turned on
    by default since they take a while to compute and 10-Fold CV NLS w/ isotonic
    regression is fast and just as good in terms of accuracy.
    """
    # Setup covariance
    np.random.seed(seed)
    T = y * n
    Sigma, tmp = _cov_functions[cov_fun](n)
    tau = annualize_vol(tmp / n)

    if ylim is None:
        ylim = (0., 2 * np.max(tau))

    names = ['sample', 'lw_oracle', 'isolw_oracle', 'kfold', 'isokfold',
             'mv_oracle', 'isolsq_mv_oracle']  # , 'isomv_oracle']
    if lw:
        names += ['lw']
    if loo:
        names += ['loo', 'isoloo']
    dfs = {
        name: pd.DataFrame(np.zeros((m, n)))
        for name in names
    }

    pbar = tqdm(total=m)
    for j in range(m):
        # Generate data
        X = sample(Sigma, T)
        S = cov(X)
        L = eig(S, return_eigenvectors=False)

        # Note: eigenvalues need to be scaled by 1 / n to convert to variance
        # Sample covariance
        dfs['sample'].iloc[j, :] = annualize_vol(L / n)

        # Oracle LW NLS shrinkage
        _, tmp = nls_oracle(X, S, Sigma)
        dfs['lw_oracle'].iloc[j, :] = annualize_vol(tmp / n)
        tmp = isotonic_regression(tmp)
        dfs['isolw_oracle'].iloc[j, :] = annualize_vol(tmp / n)

        # LW NLS shrinkage
        if lw:
            S_lw = nlshrink_covariance(X, centered=True)
            tmp = eig(S_lw, return_eigenvectors=False)
            dfs['lw'].loc[j, :] = annualize_vol(tmp / n)

        # LOO LW NLS shrinkage
        if loo:
            _, tmp = nls_loo_cv(X, S)
            dfs['loo'].iloc[j, :] = annualize_vol(tmp / n)
            tmp = isotonic_regression(tmp)
            dfs['isoloo'].iloc[j, :] = annualize_vol(tmp / n)

        # K-fold LW NLS shrinkage
        _, tmp = nls_kfold_cv(X, S, k)
        dfs['kfold'].iloc[j, :] = annualize_vol(tmp / n)
        tmp = isotonic_regression(tmp)
        dfs['isokfold'].iloc[j, :] = annualize_vol(tmp / n)

        # MinVar NLS shrinkage
        _, tmp = minvar_nls_oracle(X, S, Sigma)
        dfs['mv_oracle'].iloc[j, :] = annualize_vol(tmp / n)
        # Note: Applying isotonic regression after solving for the oracle values
        # is consistently way worse than solving the constrained LS problem so
        # it is omitted.
        # lam_1, lam_n = L[0], L[-1]
        # tmp = isotonic_regression(tmp, y_min=lam_n, y_max=lam_1)
        # dfs['isomv_oracle'].iloc[j, :] = annualize_vol(tmp / n)
        _, tmp = minvar_nls_oracle(X, S, Sigma, isotonic=True)
        dfs['isolsq_mv_oracle'].iloc[j, :] = annualize_vol(tmp / n)

        pbar.update()

    # Generate band plots for various shrinkage methods
    fig, (ax0, ax1) = plt.subplots(figsize=figsize, ncols=2)
    ax0.plot(tau, label='true')
    ax1.plot(tau, label='true')

    band_plot(dfs['sample'], ax0, 'sample')
    band_plot(dfs['sample'], ax1, 'sample')

    if lw:
        band_plot(dfs['lw'], ax1, 'lw')

    if loo:
        band_plot(dfs['loo'], ax0, 'loo')
        band_plot(dfs['isoloo'], ax1, 'isoloo')

    band_plot(dfs['kfold'], ax0, 'kfold')
    band_plot(dfs['isokfold'], ax1, 'isokfold')

    band_plot(dfs['mv_oracle'], ax0, 'mv_oracle')
    # band_plot(dfs['isomv_oracle'], ax1, 'isomv_oracle')
    band_plot(dfs['isolsq_mv_oracle'], ax1, 'isolsq_mv_oracle')

    ax0.legend()
    ax1.legend()
    ax0.set_ylim(*ylim)
    ax1.set_ylim(*ylim)

    plt.show()


@cli.command()
@click.option('--m', type=int, default=100)
@click.option('--n', type=int, default=128)
@click.option('--y', type=int, default=2)
@click.option('--cov_fun', default='H')
@click.option('--gamma', type=float, default=None)
@click.option('--lw/--no-lw', default=False)
@click.option('--loo/--no-loo', default=False)
@click.option('--K', type=int, default=10)
@click.option('--ylim', nargs=2, type=float, default=(0., 4.))
@click.option('--figsize', nargs=2, type=float, default=(16, 8))
@click.option('--seed', type=int, default=3910)
def var_ratio(m, n, y, cov_fun, gamma, lw, loo, k, ylim, figsize, seed):
    """
    Generate box plots of the variance ratios from the various shrinkage
    methods for the minimum variance portfolio (global or long-only depending
    on option flag).

    Leave-One-Out cross-validated NLS and the optimal LW NLS are not turned on
    by default since they take a while to compute and 10-Fold CV NLS w/ isotonic
    regression is fast and just as good in terms of accuracy.
    """
    np.random.seed(seed)
    T = y * n
    Sigma, tau = _cov_functions[cov_fun](n)

    names = ['sample', 'lw_oracle', 'isolw_oracle', 'kfold', 'isokfold',
             'isolsq_mv_oracle']  # , 'isomv_oracle']

    if loo:
        names.insert(1, 'loo')
        names.insert(2, 'isoloo')
    if lw:
        names.insert(1, 'lw')

    var_ratio_df = pd.DataFrame(np.zeros((m, len(names))), columns=names)
    oos_vol_df = pd.DataFrame(np.zeros((m, len(names))), columns=names)

    pbar = tqdm(total=m)
    for j in range(m):
        X = sample(Sigma, T)
        X = X - X.mean()
        S = cov(X)
        L, U = eig(S)

        # Sample covariance
        pi_S = min_var_portfolio(S, gamma=gamma)
        oos_vol_df.loc[j, 'sample'] = portfolio_vol(pi_S, Sigma)
        var_ratio_df.loc[j, 'sample'] = variance_ratio(pi_S, S, Sigma)

        # Oracle LW NLS shrinkage
        _, d_lw_oracle = nls_oracle(X, S, Sigma)
        S_lw_oracle = eig_multiply(U, d_lw_oracle)
        pi_lw_oracle = min_var_portfolio(S_lw_oracle, gamma=gamma)
        oos_vol_df.loc[j, 'lw_oracle'] = portfolio_vol(pi_lw_oracle, Sigma)
        var_ratio_df.loc[j, 'lw_oracle'] = variance_ratio(
            pi_lw_oracle, S_lw_oracle, Sigma)

        d_isolw_oracle = isotonic_regression(d_lw_oracle)
        S_isolw_oracle = eig_multiply(U, d_isolw_oracle)
        pi_isolw_oracle = min_var_portfolio(S_isolw_oracle, gamma=gamma)
        oos_vol_df.loc[j, 'isolw_oracle'] = portfolio_vol(
            pi_isolw_oracle, Sigma)
        var_ratio_df.loc[j, 'isolw_oracle'] = variance_ratio(
            pi_isolw_oracle, S_isolw_oracle, Sigma)

        # LW NLS shrinkage
        if lw:
            S_lw = nlshrink_covariance(X, centered=True)
            pi_lw = min_var_portfolio(S_lw, gamma=gamma)
            var_ratio_df.loc[j, 'lw'] = variance_ratio(pi_lw, S_lw, Sigma)

        # LOO LW NLS shrinkage
        if loo:
            _, d_loo = nls_loo_cv(X, S)
            S_loo = eig_multiply(U, d_loo)
            pi_loo = min_var_portfolio(S_loo, gamma=gamma)
            oos_vol_df.loc[j, 'loo'] = portfolio_vol(pi_loo, Sigma)
            var_ratio_df.loc[j, 'loo'] = variance_ratio(pi_loo, S_loo, Sigma)

            d_isoloo = isotonic_regression(d_loo)
            S_isoloo = eig_multiply(U, d_isoloo)
            pi_isoloo = min_var_portfolio(S_isoloo, gamma=gamma)
            oos_vol_df.loc[j, 'isoloo'] = portfolio_vol(pi_isoloo, Sigma)
            var_ratio_df.loc[j, 'isoloo'] = variance_ratio(
                pi_isoloo, S_isoloo, Sigma)

        # K-fold LW NLS shrinkage
        _, d_kfold = nls_kfold_cv(X, S, k)
        S_kfold = eig_multiply(U, d_kfold)
        pi_kfold = min_var_portfolio(S_kfold, gamma=gamma)
        oos_vol_df.loc[j, 'kfold'] = portfolio_vol(pi_kfold, Sigma)
        var_ratio_df.loc[j, 'kfold'] = variance_ratio(pi_kfold, S_kfold, Sigma)

        d_isokfold = isotonic_regression(d_kfold)
        S_isokfold = eig_multiply(U, d_isokfold)
        pi_isokfold = min_var_portfolio(S_isokfold, gamma=gamma)
        oos_vol_df.loc[j, 'isokfold'] = portfolio_vol(pi_isokfold, Sigma)
        var_ratio_df.loc[j, 'isokfold'] = variance_ratio(
            pi_isokfold, S_isokfold, Sigma)

        # MinVar NLS shrinkage
        _, d_mv_oracle = minvar_nls_oracle(X, S, Sigma)
        # Note: the raw oracle values for MinVar shrinkage are likely to
        # produce negative eigenvalues, which means the minimum variance
        # portfolio cannot be reasonably computed.  Computing variance
        # ratios for the MinVar shrinkage only works with some kind of
        # modification to the raw values.

        # Note: Applying isotonic regression after solving for the oracle values
        # is consistently way worse than solving the constrained LS problem so
        # it is omitted.
        # lam_1, lam_n = L[0], L[-1]
        # d_isomv_oracle = isotonic_regression(
        #     d_mv_oracle, y_min=lam_n, y_max=lam_1)
        # S_isomv_oracle = eig_multiply(U, d_isomv_oracle)
        # pi_isomv_oracle = min_var_portfolio(S_isomv_oracle, gamma=gamma)
        # oos_vol_df.loc[j, 'isomv_oracle'] = portfolio_vol(
        #     pi_isomv_oracle, Sigma)
        # var_ratio_df.loc[j, 'isomv_oracle'] = variance_ratio(
        #     pi_isomv_oracle, S_isomv_oracle, Sigma)

        _, d_isolsq_mv_oracle = minvar_nls_oracle(
            X, S, Sigma, isotonic=True)
        S_isolsq_mv_oracle = eig_multiply(U, d_isolsq_mv_oracle)
        pi_isolsq_mv_oracle = min_var_portfolio(S_isolsq_mv_oracle, gamma=gamma)
        oos_vol_df.loc[j, 'isolsq_mv_oracle'] = portfolio_vol(
            pi_isolsq_mv_oracle, Sigma)
        var_ratio_df.loc[j, 'isolsq_mv_oracle'] = variance_ratio(
            pi_isolsq_mv_oracle, S_isolsq_mv_oracle, Sigma)

        pbar.update()

    fig, ax = plt.subplots(figsize=figsize, ncols=2)
    var_ratio_df.boxplot(ax=ax[0])
    oos_vol_df.boxplot(ax=ax[1])
    # ax.legend()
    plt.show()


def band_plot(df, ax, label):
    """Generate banded plot"""
    m = df.median()
    q0 = df.quantile(.05)
    q1 = df.quantile(.95)
    t = list(range(len(m)))
    line = ax.plot(t, m, label=label)
    clr = line[0].get_color()
    ax.plot(t, q0.values, '--', color=clr, label=None)
    ax.plot(t, q1.values, '--', color=clr, label=None)
    ax.fill_between(t, q0, q1, facecolor=clr, alpha=.1)


if __name__ == "__main__":
    cli()
