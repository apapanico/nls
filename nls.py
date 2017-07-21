"""
CLI Script for Testing Nonlinear Shrinkage
"""

import matplotlib
matplotlib.use('qt5agg')
import click
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from cov import nlshrink_covariance

from portfolios import min_var_portfolio
from utils import *
from models import *
from shrinkage import *


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
    T = y * n
    Sigma, tau = cov_functions[cov_fun](n)

    np.random.seed(seed)
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
    _, d_isonlsq_mv_oracle = minvar_nls_oracle(X, S, Sigma, isotonic=True)
    ax0.plot(annualize_vol(d_mv_oracle / n), label='noisy-mv_oracle')
    ax1.plot(annualize_vol(d_isomv_oracle / n), label='buggy-iso-mv_oracle')
    ax1.plot(annualize_vol(d_isonlsq_mv_oracle / n), label='isolsq-mv_oracle')

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

    names = ['true', 'sample', 'lw_oracle', 'isolw_oracle', 'kfold', 'isokfold',
             'mv_oracle', 'isonlsq_mv_oracle', 'isonlsq_mv_kfold']
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
        # Build Model
        if cov_fun in ['slr', 'factor']:
            fm_seed = np.random.randint(1, 2**32 - 1)
            Sigma, tmp = cov_functions[cov_fun](n, seed=fm_seed)
        else:
            Sigma, tmp = cov_functions[cov_fun](n)
        dfs['true'].iloc[j, :] = tau = annualize_vol(tmp / n)

        if ylim is None:
            ylim = (0., 2 * np.max(tau))

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
        dfs['isonlsq_mv_oracle'].iloc[j, :] = annualize_vol(tmp / n)

        _, tmp = minvar_nls_kfold(X, S, k)
        dfs['isonlsq_mv_kfold'].iloc[j, :] = annualize_vol(tmp / n)

        pbar.update()

    # Generate band plots for various shrinkage methods
    fig, (ax0, ax1, ax2) = plt.subplots(figsize=figsize, ncols=3)
    band_plot(dfs['true'], ax0, 'true')
    band_plot(dfs['true'], ax1, 'true')
    band_plot(dfs['true'], ax2, 'true')

    band_plot(dfs['sample'], ax0, 'sample')
    band_plot(dfs['sample'], ax1, 'sample')
    band_plot(dfs['sample'], ax2, 'sample')

    if lw:
        band_plot(dfs['lw'], ax1, 'lw')

    if loo:
        band_plot(dfs['loo'], ax0, 'loo')
        band_plot(dfs['isoloo'], ax1, 'isoloo')

    band_plot(dfs['kfold'], ax0, 'kfold')
    band_plot(dfs['isokfold'], ax1, 'isokfold')

    band_plot(dfs['mv_oracle'], ax0, 'mv_oracle')
    # band_plot(dfs['isomv_oracle'], ax1, 'isomv_oracle')
    band_plot(dfs['isonlsq_mv_oracle'], ax2, 'isonlsq_mv_oracle')
    band_plot(dfs['isonlsq_mv_kfold'], ax2, 'isonlsq_mv_kfold')

    ax0.legend()
    ax1.legend()
    ax2.legend()
    ax0.set_ylim(*ylim)
    ax1.set_ylim(*ylim)
    ax2.set_ylim(*ylim)

    plt.show()


@cli.command()
@click.option('--m', type=int, default=100)
@click.option('--n', type=int, default=128)
@click.option('--y', type=int, default=2)
@click.option('--cov_fun', default='H')
@click.option('--gamma', type=float, default=None)
@click.option('--lw/--no-lw', default=False)
@click.option('--loo/--no-loo', default=False)
@click.option('--K', type=int, default=20)
@click.option('--ylim', nargs=2, type=float, default=(0., 4.))
@click.option('--figsize', nargs=2, type=float, default=(12, 12))
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

    names = ['sample', 'lw_oracle', 'isolw_oracle', 'kfold', 'isokfold',
             'isonlsq_mv_oracle', 'isonlsq_mv_kfold']  # , 'isomv_oracle']

    if loo:
        names.insert(1, 'loo')
        names.insert(2, 'isoloo')
    if lw:
        names.insert(1, 'lw')

    forecast_var_ratio_df = pd.DataFrame(
        np.zeros((m, len(names))), columns=names)
    oos_var_df = pd.DataFrame(np.zeros((m, len(names))), columns=names)
    is_var_df = pd.DataFrame(np.zeros((m, len(names))), columns=names)
    true_var_ratio_df = pd.DataFrame(np.zeros((m, len(names))), columns=names)
    te_df = pd.DataFrame(np.zeros((m, len(names))), columns=names)

    pbar = tqdm(total=m)
    for j in range(m):
        # Build Model
        if cov_fun in ['slr', 'factor']:
            fm_seed = np.random.randint(1, 2**32 - 1)
            Sigma, tau = cov_functions[cov_fun](n, seed=fm_seed)
        else:
            Sigma, tau = cov_functions[cov_fun](n)
        pi_true = min_var_portfolio(Sigma, gamma=gamma)

        # Generate Data
        X = sample(Sigma, T)
        X = X - X.mean()
        S = cov(X)
        L, U = eig(S)

        # Sample covariance
        pi_S = min_var_portfolio(S, gamma=gamma)
        oos_var_df.loc[j, 'sample'] = portfolio_var(pi_S, Sigma)
        is_var_df.loc[j, 'sample'] = portfolio_var(pi_S, S)
        forecast_var_ratio_df.loc[j, 'sample'] = variance_ratio(pi_S, S, Sigma)
        true_var_ratio_df.loc[j, 'sample'] = true_variance_ratio(
            pi_S, pi_true, Sigma)
        te_df.loc[j, 'sample'] = tracking_error(pi_S, pi_true, Sigma)

        # Oracle LW NLS shrinkage
        _, d_lw_oracle = nls_oracle(X, S, Sigma)
        S_lw_oracle = eig_multiply(U, d_lw_oracle)
        pi_lw_oracle = min_var_portfolio(S_lw_oracle, gamma=gamma)
        oos_var_df.loc[j, 'lw_oracle'] = portfolio_var(pi_lw_oracle, Sigma)
        is_var_df.loc[j, 'lw_oracle'] = portfolio_var(pi_lw_oracle, S_lw_oracle)
        forecast_var_ratio_df.loc[j, 'lw_oracle'] = variance_ratio(
            pi_lw_oracle, S_lw_oracle, Sigma)
        true_var_ratio_df.loc[j, 'lw_oracle'] = true_variance_ratio(
            pi_lw_oracle, pi_true, Sigma)
        te_df.loc[j, 'lw_oracle'] = tracking_error(
            pi_lw_oracle, pi_true, Sigma)

        d_isolw_oracle = isotonic_regression(d_lw_oracle)
        S_isolw_oracle = eig_multiply(U, d_isolw_oracle)
        pi_isolw_oracle = min_var_portfolio(S_isolw_oracle, gamma=gamma)
        oos_var_df.loc[j, 'isolw_oracle'] = portfolio_var(
            pi_isolw_oracle, Sigma)
        is_var_df.loc[j, 'isolw_oracle'] = portfolio_var(
            pi_isolw_oracle, S_isolw_oracle)
        forecast_var_ratio_df.loc[j, 'isolw_oracle'] = variance_ratio(
            pi_isolw_oracle, S_isolw_oracle, Sigma)
        true_var_ratio_df.loc[j, 'isolw_oracle'] = true_variance_ratio(
            pi_isolw_oracle, pi_true, Sigma)
        te_df.loc[j, 'isolw_oracle'] = tracking_error(
            pi_isolw_oracle, pi_true, Sigma)

        # LW NLS shrinkage
        if lw:
            S_lw = nlshrink_covariance(X, centered=True)
            pi_lw = min_var_portfolio(S_lw, gamma=gamma)
            forecast_var_ratio_df.loc[
                j, 'lw'] = variance_ratio(pi_lw, S_lw, Sigma)

        # LOO LW NLS shrinkage
        if loo:
            _, d_loo = nls_loo_cv(X, S)
            S_loo = eig_multiply(U, d_loo)
            pi_loo = min_var_portfolio(S_loo, gamma=gamma)
            oos_var_df.loc[j, 'loo'] = portfolio_var(pi_loo, Sigma)
            forecast_var_ratio_df.loc[
                j, 'loo'] = variance_ratio(pi_loo, S_loo, Sigma)

            d_isoloo = isotonic_regression(d_loo)
            S_isoloo = eig_multiply(U, d_isoloo)
            pi_isoloo = min_var_portfolio(S_isoloo, gamma=gamma)
            oos_var_df.loc[j, 'isoloo'] = portfolio_var(pi_isoloo, Sigma)
            forecast_var_ratio_df.loc[j, 'isoloo'] = variance_ratio(
                pi_isoloo, S_isoloo, Sigma)

        # K-fold LW NLS shrinkage
        _, d_kfold = nls_kfold_cv(X, S, k)
        S_kfold = eig_multiply(U, d_kfold)
        pi_kfold = min_var_portfolio(S_kfold, gamma=gamma)
        oos_var_df.loc[j, 'kfold'] = portfolio_var(pi_kfold, Sigma)
        is_var_df.loc[j, 'kfold'] = portfolio_var(pi_kfold, S_kfold)
        forecast_var_ratio_df.loc[j, 'kfold'] = variance_ratio(
            pi_kfold, S_kfold, Sigma)
        true_var_ratio_df.loc[j, 'kfold'] = true_variance_ratio(
            pi_kfold, pi_true, Sigma)
        te_df.loc[j, 'kfold'] = tracking_error(pi_kfold, pi_true, Sigma)

        d_isokfold = isotonic_regression(d_kfold)
        S_isokfold = eig_multiply(U, d_isokfold)
        pi_isokfold = min_var_portfolio(S_isokfold, gamma=gamma)
        oos_var_df.loc[j, 'isokfold'] = portfolio_var(pi_isokfold, Sigma)
        is_var_df.loc[j, 'isokfold'] = portfolio_var(pi_isokfold, S_isokfold)
        forecast_var_ratio_df.loc[j, 'isokfold'] = variance_ratio(
            pi_isokfold, S_isokfold, Sigma)
        true_var_ratio_df.loc[j, 'isokfold'] = true_variance_ratio(
            pi_isokfold, pi_true, Sigma)
        te_df.loc[j, 'isokfold'] = tracking_error(
            pi_isokfold, pi_true, Sigma)

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
        # oos_var_df.loc[j, 'isomv_oracle'] = portfolio_var(
        #     pi_isomv_oracle, Sigma)
        # forecast_var_ratio_df.loc[j, 'isomv_oracle'] = variance_ratio(
        #     pi_isomv_oracle, S_isomv_oracle, Sigma)

        _, d_isonlsq_mv_oracle = minvar_nls_oracle(
            X, S, Sigma, isotonic=True)
        S_isonlsq_mv_oracle = eig_multiply(U, d_isonlsq_mv_oracle)
        pi_isonlsq_mv_oracle = min_var_portfolio(
            S_isonlsq_mv_oracle, gamma=gamma)
        oos_var_df.loc[j, 'isonlsq_mv_oracle'] = portfolio_var(
            pi_isonlsq_mv_oracle, Sigma)
        is_var_df.loc[j, 'isonlsq_mv_oracle'] = portfolio_var(
            pi_isonlsq_mv_oracle, S_isonlsq_mv_oracle)
        forecast_var_ratio_df.loc[j, 'isonlsq_mv_oracle'] = variance_ratio(
            pi_isonlsq_mv_oracle, S_isonlsq_mv_oracle, Sigma)
        true_var_ratio_df.loc[j, 'isonlsq_mv_oracle'] = true_variance_ratio(
            pi_isonlsq_mv_oracle, pi_true, Sigma)
        te_df.loc[j, 'isonlsq_mv_oracle'] = tracking_error(
            pi_isonlsq_mv_oracle, pi_true, Sigma)

        _, d_isonlsq_mv_kfold = minvar_nls_kfold(X, S, k)
        S_isonlsq_mv_kfold = eig_multiply(U, d_isonlsq_mv_kfold)
        pi_isonlsq_mv_kfold = min_var_portfolio(
            S_isonlsq_mv_kfold, gamma=gamma)
        oos_var_df.loc[j, 'isonlsq_mv_kfold'] = portfolio_var(
            pi_isonlsq_mv_kfold, Sigma)
        is_var_df.loc[j, 'isonlsq_mv_kfold'] = portfolio_var(
            pi_isonlsq_mv_kfold, S_isonlsq_mv_kfold)
        forecast_var_ratio_df.loc[j, 'isonlsq_mv_kfold'] = variance_ratio(
            pi_isonlsq_mv_kfold, S_isonlsq_mv_kfold, Sigma)
        true_var_ratio_df.loc[j, 'isonlsq_mv_kfold'] = true_variance_ratio(
            pi_isonlsq_mv_kfold, pi_true, Sigma)
        te_df.loc[j, 'isonlsq_mv_kfold'] = tracking_error(
            pi_isonlsq_mv_kfold, pi_true, Sigma)

        pbar.update()

    fig, ax = plt.subplots(figsize=figsize, ncols=5)
    forecast_var_ratio_df.boxplot(ax=ax[0])
    true_var_ratio_df.boxplot(ax=ax[1])
    oos_var_df.boxplot(ax=ax[2])
    is_var_df.boxplot(ax=ax[3])
    te_df.boxplot(ax=ax[4])

    ax[0].set_title('Forecast Variance Ratios')
    ax[1].set_title('True Variance Ratios')
    ax[2].set_title('Out-of-Sample Variance')
    ax[3].set_title('In-Sample Variance')
    ax[4].set_title('Tracking Error to True MinVar')

    ylim = (.5 * min(is_var_df.values.min(), oos_var_df.values.min()),
            2 * max(is_var_df.values.max(), oos_var_df.values.max()))
    ax[0].set_ylim((0, 1.5))
    ax[1].set_ylim((0, 3))
    ax[2].set_ylim(ylim)
    ax[3].set_ylim(ylim)
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
