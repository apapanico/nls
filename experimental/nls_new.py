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
import yaml

from cov import nlshrink_covariance

from portfolios import min_var_portfolio
from utils import *
from models import *
from shrinkage_new import *
from shrinkage_new import _nls_cv, _minvar_nlsq

from classes import Simulation


def parse_kwarg_input(s):
    if '-' in s:
        cov_fun, kwargs = s.split('-')
        cov_fun = cov_fun.lower()
        kwargs = yaml.load(kwargs.replace(':', ': '))
        if kwargs is None:
            kwargs = {}
    else:
        cov_fun = s.lower()
        kwargs = {}
    return cov_fun, kwargs


class FunctionKwargParamType(click.ParamType):
    name = 'mixed'

    def convert(self, value, param, ctx):
        try:
            return parse_kwarg_input(value)
        except Exception:
            self.fail('%s is not a valid input' % value, param, ctx)


FUNCTION_KWARG = FunctionKwargParamType()


@click.group()
def cli():
    pass


@cli.command()
@click.option('--N', 'N', type=int, default=100)
@click.option('--y', type=int, default=2)
@click.option('--cov_fun', 'cov_fun_kwargs',
              type=FUNCTION_KWARG, default='slr-{}')
@click.option('--loo/--no-loo', default=True)
@click.option('--K', 'K', type=int, default=10)
@click.option('--ylim', nargs=2, type=float, default=(0., .25))
@click.option('--figsize', nargs=2, type=float, default=(16, 8))
@click.option('--seed', type=int, default=3910)
@click.option('--trace/--no-trace', default=True)
@click.option('--upper_bound/--no-upper_bound', default=True)
def demo(N, y, cov_fun_kwargs, loo, K, ylim, figsize, seed, trace, upper_bound):
    """Simple demo showing the results of the various shrinkage methods"""

    T = y * N
    cov_fun, cov_kwargs = cov_fun_kwargs
    Sigma, tau = cov_functions[cov_fun](N, seed=seed, **cov_kwargs)

    np.random.seed(seed)
    sim = Simulation(Sigma, T)

    fig, (ax0, ax1) = plt.subplots(figsize=figsize, ncols=2)
    # ax0.plot(annualize_vol(tau / N), label='true')
    # ax1.plot(annualize_vol(tau / N), label='true')
    # ax0.plot(annualize_vol(lam / N), label='sample')
    # ax1.plot(annualize_vol(lam / N), label='sample')

    # Oracle LW NLS shrinkage
    # d_lw_oracle = nls_oracle(sim)
    # d_isolw_oracle = nls_oracle(sim, isotonic=True)
    # ax0.plot(annualize_vol(d_lw_oracle / N), label='lw oracle')
    # ax1.plot(annualize_vol(d_isolw_oracle / N), label='lw iso oracle')

    # # LW NLS shrinkage
    # S_lw = nlshrink_covariance(X, centered=True)
    # d_lw = eig(S_lw, return_eigenvectors=False)
    # ax1.plot(annualize_vol(d_lw / N), label='lw')

    # if loo:
    #     # LOO LW NLS shrinkage
    #     _, d_loo = nls_loo_cv(X, S, U)
    #     d_isoloo = isotonic_regression(d_loo)
    #     ax0.plot(annualize_vol(d_loo / N), label='noisy-loo')
    #     ax1.plot(annualize_vol(d_isoloo / N), label='isoloo')

    # K-fold LW NLS shrinkage
    # d_lw_loo = nls_loo(sim)
    # d_lw_isoloo = nls_loo(sim, isotonic=True)
    # ax0.plot(annualize_vol(d_lw_loo / N), label='lw_kfold')
    # ax1.plot(annualize_vol(d_lw_isoloo / N), label='lw_isoloo')

    d_lw_kfold = nls_kfold(sim, K)
    d_lw_isokfold = nls_kfold(sim, K, isotonic=True)
    ax0.plot(annualize_vol(d_lw_kfold / N), label='lw_kfold')
    ax1.plot(annualize_vol(d_lw_isokfold / N), label='lw_isokfold')

    # MinVar NLS shrinkage
    d_mv_oracle = minvar_oracle(
        sim, monotonicity=None, trace=trace, upper_bound=upper_bound)
    d_mv_mono_oracle = minvar_oracle(
        sim, monotonicity='constraint', trace=trace, upper_bound=upper_bound)
    d_mv_iso_oracle = minvar_oracle(
        sim, monotonicity='isotonic', trace=trace, upper_bound=upper_bound)

    ax0.plot(annualize_vol(d_mv_oracle / N), label='mv_oracle')
    ax1.plot(annualize_vol(d_mv_mono_oracle / N), label='mv_mono_oracle')
    ax1.plot(annualize_vol(d_mv_iso_oracle / N), label='mv_iso_oracle')

    d_mv_loo = minvar_loo(
        sim, monotonicity=None, trace=trace, upper_bound=upper_bound)
    d_mv_mono_loo = minvar_loo(
        sim, monotonicity='constraint', trace=trace, upper_bound=upper_bound)
    d_mv_iso_loo = minvar_loo(
        sim, monotonicity='isotonic', trace=trace, upper_bound=upper_bound)

    ax0.plot(annualize_vol(d_mv_loo / N), label='mv_loo')
    ax1.plot(annualize_vol(d_mv_mono_loo / N), label='mv_mono_loo')
    ax1.plot(annualize_vol(d_mv_iso_loo / N), label='mv_iso_loo')

    ax0.legend()
    ax1.legend()
    # ax0.set_ylim(*ylim)
    # ax1.set_ylim(*ylim)
    plt.show()


@cli.command()
@click.option('--m', type=int, default=100)
@click.option('--n', type=int, default=100)
@click.option('--y', type=int, default=2)
@click.option('--cov_fun', default='slr')
@click.option('--lw/--no-lw', default=False)
@click.option('--loo/--no-loo', default=False)
@click.option('--K', type=int, default=10)
@click.option('--ylim', nargs=2, type=float, default=None)
@click.option('--figsize', nargs=2, type=float, default=(16, 12))
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
        lam, U = eig(S)

        # Note: eigenvalues need to be scaled by 1 / n to convert to variance
        # Sample covariance
        dfs['sample'].iloc[j, :] = annualize_vol(lam / n)

        # Oracle LW NLS shrinkage
        _, tmp = nls_oracle(X, S, U, Sigma)
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
            _, tmp = nls_loo_cv(X, S, U)
            dfs['loo'].iloc[j, :] = annualize_vol(tmp / n)
            tmp = isotonic_regression(tmp)
            dfs['isoloo'].iloc[j, :] = annualize_vol(tmp / n)

        # K-fold LW NLS shrinkage
        _, tmp = nls_kfold_cv(X, S, U, k)
        dfs['kfold'].iloc[j, :] = annualize_vol(tmp / n)
        tmp = isotonic_regression(tmp)
        dfs['isokfold'].iloc[j, :] = annualize_vol(tmp / n)

        # MinVar NLS shrinkage
        _, tmp = minvar_nls_oracle(X, S, lam, U, Sigma)
        dfs['mv_oracle'].iloc[j, :] = annualize_vol(tmp / n)
        # Note: Applying isotonic regression after solving for the oracle values
        # is consistently way worse than solving the constrained LS problem so
        # it is omitted.
        # lam_1, lam_n = lam[0], lam[-1]
        # tmp = isotonic_regression(tmp, y_min=lam_n, y_max=lam_1)
        # dfs['isomv_oracle'].iloc[j, :] = annualize_vol(tmp / n)
        _, tmp = minvar_nls_oracle(X, S, lam, U, Sigma, isotonic=True)
        dfs['isonlsq_mv_oracle'].iloc[j, :] = annualize_vol(tmp / n)

        _, tmp = minvar_nls_kfold(X, S, lam, U, k)
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


def portfolio_analysis(S, Sigma, gamma, pi_true):
    pi = min_var_portfolio(S, gamma=gamma)
    out = {
        'oos_var': portfolio_var(pi, Sigma),
        'is_var': portfolio_var(pi, S),
        'forecast_var_ratio': variance_ratio(pi, S, Sigma),
        'true_var_ratio': true_variance_ratio(pi, pi_true, Sigma),
        'te': tracking_error(pi, pi_true, Sigma)
    }
    return out


@cli.command()
@click.option('--M', 'M', type=int, default=100)
@click.option('--N', 'N', type=int, default=128)
@click.option('--y', type=int, default=2)
@click.option('--cov_fun', default='slr')
@click.option('--gamma', type=float, default=None)
@click.option('--lw/--no-lw', default=False)
@click.option('--loo/--no-loo', default=False)
@click.option('--K', 'K', type=int, default=20)
@click.option('--ylim', nargs=2, type=float, default=(0., 4.))
@click.option('--figsize', nargs=2, type=float, default=(12, 16))
@click.option('--seed', type=int, default=3910)
def var_ratio(M, N, y, cov_fun, gamma, lw, loo, K, ylim, figsize, seed):
    """
    Generate box plots of the variance ratios from the various shrinkage
    methods for the minimum variance portfolio (global or long-only depending
    on option flag).

    Leave-One-Out cross-validated NLS and the optimal LW NLS are not turned on
    by default since they take a while to compute and 10-Fold CV NLS w/ isotonic
    regression is fast and just as good in terms of accuracy.
    """
    np.random.seed(seed)
    T = y * N

    names = ['sample', 'lw_oracle', 'lw_iso_oracle', 'lw_kfold', 'lw_isokfold',
             'mv_isonlsq_oracle', 'mv_isonlsq_kfold']  # , 'isomv_oracle']

    if loo:
        names.insert(1, 'lw_loo')
        names.insert(2, 'lw_isoloo')
    if lw:
        names.insert(1, 'lw')

    empty_df = pd.DataFrame(np.zeros((M, len(names))), columns=names)

    dfs = {
        'oos_var': empty_df,
        'is_var': empty_df.copy(),
        'forecast_var_ratio': empty_df.copy(),
        'true_var_ratio': empty_df.copy(),
        'te': empty_df.copy(),
    }
    # forecast_var_ratio_df = pd.DataFrame(
    #     np.zeros((M, len(names))), columns=names)
    # oos_var_df = pd.DataFrame(np.zeros((M, len(names))), columns=names)
    # is_var_df = pd.DataFrame(np.zeros((M, len(names))), columns=names)
    # true_var_ratio_df = pd.DataFrame(np.zeros((M, len(names))), columns=names)
    # te_df = pd.DataFrame(np.zeros((M, len(names))), columns=names)

    pbar = tqdm(total=M)

    results = []
    for j in range(M):
        # Build Model
        if cov_fun in ['slr', 'factor']:
            fm_seed = np.random.randint(1, 2**32 - 1)
            Sigma, tau = cov_functions[cov_fun](N, seed=fm_seed)
        else:
            Sigma, tau = cov_functions[cov_fun](N)
        pi_true = min_var_portfolio(Sigma, gamma=gamma)

        # Generate Data
        X = sample(Sigma, T)
        X = X - X.mean()
        S = cov(X)
        lam, U = eig(S)

        # Sample covariance
        name = 'sample'
        result = portfolio_analysis(S, Sigma, gamma, pi_true)
        results.append({name: result})
        for key in result:
            dfs[key].loc[j, name] = result[key]

        # Oracle LW NLS shrinkage
        name = 'lw_oracle'
        _, d_lw_oracle = nls_oracle(X, S, U, Sigma)
        S_lw_oracle = eig_multiply(U, d_lw_oracle)
        result = portfolio_analysis(S_lw_oracle, Sigma, gamma, pi_true)
        results.append({name: result})
        for key in result:
            dfs[key].loc[j, name] = result[key]

        name = 'lw_iso_oracle'
        d_lw_iso_oracle = isotonic_regression(d_lw_oracle)
        S_lw_iso_oracle = eig_multiply(U, d_lw_iso_oracle)
        result = portfolio_analysis(S_lw_iso_oracle, Sigma, gamma, pi_true)
        results.append({name: result})
        for key in result:
            dfs[key].loc[j, name] = result[key]

        # LW NLS shrinkage
        if lw:
            name = 'lw'
            S_lw = nlshrink_covariance(X, centered=True)
            result = portfolio_analysis(S_lw, Sigma, gamma, pi_true)
            results.append({name: result})
            for key in result:
                dfs[key].loc[j, name] = result[key]

        # LOO LW NLS shrinkage
        if loo:
            name = 'lw_loo'
            _, d_lw_loo = nls_loo_cv(X, S, U)
            S_lw_loo = eig_multiply(U, d_lw_loo)
            result = portfolio_analysis(S_lw_loo, Sigma, gamma, pi_true)
            results.append({name: result})
            for key in result:
                dfs[key].loc[j, name] = result[key]

            name = 'lw_isoloo'
            d_lw_isoloo = isotonic_regression(d_lw_loo)
            S_lw_isoloo = eig_multiply(U, d_lw_isoloo)
            result = portfolio_analysis(S_lw_isoloo, Sigma, gamma, pi_true)
            results.append({name: result})
            for key in result:
                dfs[key].loc[j, name] = result[key]

        # K-fold LW NLS shrinkage
        name = 'lw_kfold'
        _, d_lw_kfold = nls_kfold_cv(X, S, U, K)
        S_lw_kfold = eig_multiply(U, d_lw_kfold)
        result = portfolio_analysis(S_lw_kfold, Sigma, gamma, pi_true)
        results.append({name: result})
        for key in result:
            dfs[key].loc[j, name] = result[key]

        name = 'lw_isokfold'
        d_lw_isokfold = isotonic_regression(d_lw_kfold)
        S_lw_isokfold = eig_multiply(U, d_lw_isokfold)
        result = portfolio_analysis(S_lw_isokfold, Sigma, gamma, pi_true)
        results.append({name: result})
        for key in result:
            dfs[key].loc[j, name] = result[key]

        # MinVar NLS shrinkage
        _, d_mv_oracle = minvar_nls_oracle(X, S, lam, U, Sigma)
        # Note: the raw oracle values for MinVar shrinkage are likely to
        # produce negative eigenvalues, which means the minimum variance
        # portfolio cannot be reasonably computed.  Computing variance
        # ratios for the MinVar shrinkage only works with some kind of
        # modification to the raw values.

        # Note: Applying isotonic regression after solving for the oracle values
        # is consistently way worse than solving the constrained LS problem so
        # it is omitted.
        # d_mv_iso_oracle = isotonic_regression(d_mv_oracle)
        # S_mv_iso_oracle = eig_multiply(U, d_mv_iso_oracle)

        name = 'mv_isonlsq_oracle'
        _, d_mv_isonlsq_oracle = minvar_nls_oracle(
            X, S, lam, U, Sigma, isotonic=True)
        S_mv_isonlsq_oracle = eig_multiply(U, d_mv_isonlsq_oracle)
        result = portfolio_analysis(S_mv_isonlsq_oracle, Sigma, gamma, pi_true)
        results.append({name: result})
        for key in result:
            dfs[key].loc[j, name] = result[key]

        name = 'mv_isonlsq_kfold'
        _, d_mv_isonlsq_kfold = minvar_nls_kfold(X, S, lam, U, K)
        S_d_mv_isonlsq_kfold = eig_multiply(U, d_mv_isonlsq_kfold)
        result = portfolio_analysis(S_d_mv_isonlsq_kfold, Sigma, gamma, pi_true)
        results.append({name: result})
        for key in result:
            dfs[key].loc[j, name] = result[key]

        pbar.update()

    fig, ax = plt.subplots(figsize=figsize, ncols=5)
    fig.suptitle("Shrinkage Performance: N={}".format(N))
    dfs['forecast_var_ratio'].boxplot(ax=ax[0])
    dfs['true_var_ratio'].boxplot(ax=ax[1])
    dfs['oos_var'].boxplot(ax=ax[2])
    dfs['is_var'].boxplot(ax=ax[3])
    dfs['te'].boxplot(ax=ax[4])

    ax[0].set_title('Forecast Variance Ratios')
    ax[1].set_title('True Variance Ratios')
    ax[2].set_title('Out-of-Sample Variance')
    ax[3].set_title('In-Sample Variance')
    ax[4].set_title('Tracking Error to True MinVar')

    ax[0].set_ylim((0, 2.))
    ax[1].set_ylim((0, 3.))

    ylim = (.5 * min(dfs['is_var'].values.min(), dfs['oos_var'].values.min()),
            2 * max(dfs['is_var'].values.max(), dfs['oos_var'].values.max()))
    ax[2].set_ylim(ylim)
    ax[3].set_ylim(ylim)
    fig.autofmt_xdate(rotation=90)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=.22,
                        top=0.9, wspace=.36, hspace=.2)
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
