"""
=======================================================================
Shrinkage covariance estimation: LedoitWolf, OAS, and Cross Validation
=======================================================================

When working with covariance estimation, the usual approach is to use a
maximum likelihood estimator, such as the class
sklearn.covariance.EmpiricalCovariance. It is unbiased, i.e. it converges to
the true (population) covariance when given many observations. However, it can
also be beneficial to regularize it, in order to reduce its variance; this, in
turn, introduces some bias. This example illustrates the simple regularization
used in shrunk covariance estimators. In particular, it focuses on how to set
the amount of regularization, i.e. how to choose the bias-variance trade- off.

Here we compare 3 approaches:

* A close formula proposed by Ledoit and Wolf to compute the asymptotically
  optimal regularization parameter (minimizing a MSE criterion), yielding the
  sklearn.covariance.LedoitWolf covariance estimate.

* An improvement of the Ledoit-Wolf shrinkage, the sklearn.covariance.OAS,
  proposed by Chen et al. Its convergence is significantly better under the
  assumption that the data are Gaussian, in particular for small samples.

* Setting the parameter by cross-validating the likelihood on three folds
  according to a grid of potential shrinkage parameters.

"""

import logging

import numpy as np
from sklearn.covariance import (ShrunkCovariance,
                                ledoit_wolf as sk_lw_covariance,
                                oas as sk_oas_covariance)

from .emp import empirical_covariance
from .shrink import shrunk_covariance
from .cv import cv_covariance


LOGGER = logging.getLogger(__name__)


def lw_covariance(X, centered=False, block_size=1000, return_shrink=False):
    LOGGER.info("computing Ledoit-Wolf optimally shrunk covariance")
    cov, shrink = sk_lw_covariance(
        X, assume_centered=centered, block_size=block_size)
    if return_shrink:
        return cov, shrink
    else:
        return cov


def diag_shrunk_covariance(X, centered=False, shrinkage=0.1):
    LOGGER.info("computing shrunk covariance with delta={}".format(shrinkage))
    emp_cov = empirical_covariance(X, assume_centered=centered)
    n_features = emp_cov.shape[0]
    target = np.trace(emp_cov) / n_features * np.identity(n_features)
    cov = shrunk_covariance(emp_cov, target, shrinkage)
    return cov


def oas_covariance(X, centered=False, return_shrink=False):
    LOGGER.info("computing OAS optimally shrunk covariance")
    cov, shrink = sk_oas_covariance(X, assume_centered=centered,)
    if return_shrink:
        return cov, shrink
    else:
        return cov


def cv_diag_shrunk_covariance(X, centered=False, n_splits=100,
                              test_size=.1, random_state=42,
                              shrinkages=np.logspace(-2, 0, 30),
                              return_shrink=False):
    LOGGER.info("computing optimally shrunk covariance via cross-validation")
    cov, cv_shrink = cv_covariance(
        X, cov_model=ShrunkCovariance, centered=centered, n_splits=n_splits,
        test_size=test_size, random_state=random_state, shrinkages=shrinkages,
        return_shrink=True)

    _, lw_shrink = lw_covariance(X, centered=centered, return_shrink=True)
    _, oas_shrink = oas_covariance(X, centered=centered, return_shrink=True)
    LOGGER.info("CV optimal shrinkage: {}".format(cv_shrink))
    LOGGER.info("LW optimal shrinkage: {}".format(lw_shrink))
    LOGGER.info("OAS optimal shrinkage: {}".format(oas_shrink))

    return cov
