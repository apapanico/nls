"""
"""

import logging
import warnings

import numpy as np
from sklearn.covariance import ShrunkCovariance
from sklearn.utils import check_array

from .emp import empirical_covariance
from .shrink import shrunk_covariance
from .cv import cv_covariance


LOGGER = logging.getLogger(__name__)


def lw_const_corr_covariance(X, centered=False, return_shrink=False):
    """Estimates the shrunk Ledoit-Wolf covariance matrix with constant
    correlation target.
    Read more in the :ref:`User Guide <shrunk_covariance>`.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data from which to compute the covariance estimate
    assume_centered : boolean, default=False
        If True, data are not centered before computation.
        Useful to work with data whose mean is significantly equal to
        zero but is not exactly zero.
        If False, data are centered before computation.
    block_size : int, default=1000
        Size of the blocks into which the covariance matrix will be split.
        This is purely a memory optimization and does not affect results.
    Returns
    -------
    shrunk_cov : array-like, shape (n_features, n_features)
        Shrunk covariance.
    shrinkage : float
        Coefficient in the convex combination used for the computation
        of the shrunk estimate.
    Notes
    -----
    The regularized (shrunk) covariance is:
    (1 - shrinkage)*cov
      + shrinkage * Const_Corr_Target

    where Const_Corr_Target is the constant correlation target of Ledoit,
    Olivier and Michael Wolf (2004b). “Honey, I shrunk the sample covariance
    matrix”. In: The Journal of Portfolio Management 30.4, pp. 110–119.
    """
    X = np.asarray(X)
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        if not centered:
            X = X - X.mean()
        return np.atleast_2d((X ** 2).mean()), 0.
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")
        n_samples = 1
        n_features = X.size
    else:
        n_samples, n_features = X.shape

    # get Ledoit-Wolf shrinkage
    emp_cov = empirical_covariance(X, centered=centered)
    target = lw_const_corr_target(X, emp_cov, centered=centered)
    shrinkage = lw_const_corr_shrinkage(X, emp_cov, target, centered=centered)
    cov = shrunk_covariance(emp_cov, target, shrinkage)

    if return_shrink:
        return cov, shrinkage
    else:
        return cov


class ConstantCorrelationShrunkCovariance(ShrunkCovariance):

    def fit(self, X, y=None):
        """ Fits the shrunk covariance model
        according to the given training data and parameters.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : not used, present for API consistence purpose.
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        X = check_array(X)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        emp_cov = empirical_covariance(X - self.location_, centered=True)
        target = lw_const_corr_target(
            X - self.location_, emp_cov, centered=True)
        covariance = shrunk_covariance(emp_cov, target, self.shrinkage)
        self._set_covariance(covariance)

        return self


class LedoitWolfConstantCorrelationShrunkCovariance(ShrunkCovariance):

    def fit(self, X, y=None):
        """ Fits the shrunk covariance model
        according to the given training data and parameters.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : not used, present for API consistence purpose.
        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X)
        X = check_array(X)
        if self.assume_centered:
            self.location_ = np.zeros(X.shape[1])
        else:
            self.location_ = X.mean(0)
        covariance, shrinkage = lw_const_corr_covariance(
            X - self.location_, assume_centered=True, return_shrink=True)

        self.shrinkage_ = shrinkage
        self._set_covariance(covariance)

        return self


def cv_const_corr_shrunk_covariance(X, centered=False, n_splits=100,
                                    test_size=.1, random_state=42,
                                    shrinkages=np.logspace(-2, 0, 30),
                                    return_shrink=False):
    LOGGER.info("computing optimally shrunk covariance via cross-validation")
    cov, shrinkage = cv_covariance(
        X, cov_model=ConstantCorrelationShrunkCovariance, centered=centered,
        n_splits=n_splits, test_size=test_size, random_state=random_state,
        shrinkages=shrinkages, return_shrink=True)

    if return_shrink:
        return cov, shrinkage
    else:
        return cov


def lw_const_corr_target(X, emp_cov, centered=False):
    X = np.asarray(X)
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        warnings.warn("Only one feature available. ")
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.shape[0] == 1:
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")
    n_samples, n_features = X.shape

    # optionally center data
    if not centered:
        X = X - X.mean(0)

    D = np.sqrt(np.diag(emp_cov))
    D_inv = 1. / D

    N = n_features**2 - n_features
    corr = D_inv.reshape((-1, 1)) * emp_cov * D_inv
    r_bar = (np.sum(corr) - np.sum(np.diag(corr))) / N

    target = D.reshape((-1, 1)) * np.full((n_features, n_features), r_bar) * D
    target.flat[::n_features + 1] = np.diag(emp_cov)

    return target


def lw_const_corr_shrinkage(X, emp_cov, target, centered=False):
    X = np.asarray(X)
    # for only one feature, the result is the same whatever the shrinkage
    if len(X.shape) == 2 and X.shape[1] == 1:
        warnings.warn("Only one feature available. ")
    if X.ndim == 1:
        X = np.reshape(X, (1, -1))

    if X.shape[0] == 1:
        warnings.warn("Only one sample available. "
                      "You may want to reshape your data array")
    n_samples, n_features = X.shape

    # optionally center data
    if not centered:
        X = X - X.mean(0)

    D = np.sqrt(np.diag(emp_cov))
    D_inv = 1. / D

    N = n_features**2 - n_features
    corr = D_inv.reshape((-1, 1)) * emp_cov * D_inv
    r_bar = (np.sum(corr) - np.sum(np.diag(corr))) / N

    X_sq = X**2
    X_cu = X**3
    s_11 = emp_cov
    s_22 = np.dot(X_sq.T, X_sq) / X_sq.shape[0]
    s_31 = np.dot(X_cu.T, X) / X_sq.shape[0]
    s_ii = np.diag(s_11).reshape((-1, 1))
    s_jj = np.diag(s_11)

    pi = s_22 - s_11**2
    pi_hat = np.sum(pi)

    theta_ii = s_31 - s_ii * s_11
    theta_ii.flat[::n_features + 1] = 0.
    tmp1 = np.sqrt(s_jj / s_ii) * theta_ii
    rho_hat = np.sum(np.diag(pi)) + r_bar * np.sum(tmp1)

    gamma_hat = np.sum((target - emp_cov)**2)

    kappa_hat = (pi_hat - rho_hat) / gamma_hat
    delta_hat = max(0., min(kappa_hat / n_samples, 1.))
    return delta_hat
