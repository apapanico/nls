# """
# =======================================================================
# Shrinkage covariance estimation: LedoitWolf, OAS, and Cross Validation
# =======================================================================

# When working with covariance estimation, the usual approach is to use a
# maximum likelihood estimator, such as the class
# sklearn.covariance.EmpiricalCovariance. It is unbiased, i.e. it converges to
# the true (population) covariance when given many observations. However, it can
# also be beneficial to regularize it, in order to reduce its variance; this, in
# turn, introduces some bias. This example illustrates the simple regularization
# used in shrunk covariance estimators. In particular, it focuses on how to set
# the amount of regularization, i.e. how to choose the bias-variance trade- off.

# Here we compare 3 approaches:

# * A close formula proposed by Ledoit and Wolf to compute the asymptotically
#   optimal regularization parameter (minimizing a MSE criterion), yielding the
#   sklearn.covariance.LedoitWolf covariance estimate.

# * An improvement of the Ledoit-Wolf shrinkage, the sklearn.covariance.OAS,
#   proposed by Chen et al. Its convergence is significantly better under the
#   assumption that the data are Gaussian, in particular for small samples.

# * Setting the parameter by cross-validating the likelihood on three folds
#   according to a grid of potential shrinkage parameters.

# """

# import logging
# import warnings

# import numpy as np
# from sklearn.covariance import (ShrunkCovariance,
#                                 empirical_covariance as sk_empirical_covariance,
#                                 ledoit_wolf as sk_lw_covariance,
#                                 shrunk_covariance as sk_shrunk_covariance,
#                                 oas as sk_oas_covariance)
# from sklearn.model_selection import GridSearchCV, ShuffleSplit
# from sklearn.utils import check_array


# __all__ = [
#     'cov_mats',
#     'empirical_covariance',
#     'diag_shrunk_covariance',
#     'lw_covariance',
#     'oas_covariance',
#     'lw_const_corr_covariance',
#     'cv_diag_shrunk_covariance',
#     'cv_const_corr_shrunk_covariance'
# ]

# LOGGER = logging.getLogger(__name__)

# DiagShrunkCovariance = ShrunkCovariance
# sk_diag_shrunk_covariance = sk_shrunk_covariance


# def shrunk_covariance(emp_cov, target, shrinkage):
#     return shrinkage * emp_cov + (1 - shrinkage) * target


# def empirical_covariance(X, centered=False):
#     LOGGER.info("computing empirical covariance")
#     return sk_empirical_covariance(X, assume_centered=centered)


# def lw_covariance(X, centered=False, block_size=1000, return_shrink=False):
#     LOGGER.info("computing Ledoit-Wolf optimally shrunk covariance")
#     cov, shrink = sk_lw_covariance(
#         X, assume_centered=centered, block_size=block_size)
#     if return_shrink:
#         return cov, shrink
#     else:
#         return cov


# def diag_shrunk_covariance(X, centered=False, shrinkage=0.1):
#     LOGGER.info("computing shrunk covariance with delta={}".format(shrinkage))
#     emp_cov = empirical_covariance(X, assume_centered=centered)
#     cov = sk_diag_shrunk_covariance(emp_cov, shrinkage=shrinkage)
#     return cov


# def oas_covariance(X, centered=False, return_shrink=False):
#     LOGGER.info("computing OAS optimally shrunk covariance")
#     cov, shrink = sk_oas_covariance(X, assume_centered=centered,)
#     if return_shrink:
#         return cov, shrink
#     else:
#         return cov


# def cv_covariance(X, cov_model=DiagShrunkCovariance,
#                   centered=False, n_splits=100,
#                   test_size=.1, random_state=42,
#                   shrinkages=np.logspace(-2, 0, 30),
#                   return_shrink=False):
#     tuned_parameters = [{'shrinkage': shrinkages}]
#     cv = ShuffleSplit(n_splits=n_splits, test_size=test_size,
#                       random_state=random_state)
#     cov_cv = GridSearchCV(
#         cov_model(assume_centered=centered),
#         tuned_parameters,
#         cv=cv
#     )
#     cov_cv.fit(X)
#     best_est = cov_cv.best_estimator_
#     cov = best_est.covariance_

#     if return_shrink:
#         cv_shrink = best_est.shrinkage
#         return cov, cv_shrink
#     else:
#         return cov


# def cv_diag_shrunk_covariance(X, centered=False, n_splits=100,
#                               test_size=.1, random_state=42,
#                               shrinkages=np.logspace(-2, 0, 30),
#                               return_shrink=False):
#     LOGGER.info("computing optimally shrunk covariance via cross-validation")
#     cov, cv_shrink = cv_covariance(
#         X, cov_model=DiagShrunkCovariance, centered=centered, n_splits=n_splits,
#         test_size=test_size, random_state=random_state, shrinkages=shrinkages,
#         return_shrink=True)

#     _, lw_shrink = lw_covariance(X, centered=centered, return_shrink=True)
#     _, oas_shrink = oas_covariance(X, centered=centered, return_shrink=True)
#     LOGGER.info("CV optimal shrinkage: {}".format(cv_shrink))
#     LOGGER.info("LW optimal shrinkage: {}".format(lw_shrink))
#     LOGGER.info("OAS optimal shrinkage: {}".format(oas_shrink))

#     return cov


# def lw_const_corr_target(X, emp_cov, centered=False):
#     X = np.asarray(X)
#     # for only one feature, the result is the same whatever the shrinkage
#     if len(X.shape) == 2 and X.shape[1] == 1:
#         warnings.warn("Only one feature available. ")
#     if X.ndim == 1:
#         X = np.reshape(X, (1, -1))

#     if X.shape[0] == 1:
#         warnings.warn("Only one sample available. "
#                       "You may want to reshape your data array")
#     n_samples, n_features = X.shape

#     # optionally center data
#     if not centered:
#         X = X - X.mean(0)

#     D = np.sqrt(np.diag(emp_cov))
#     D_inv = 1. / D

#     N = n_features**2 - n_features
#     corr = D_inv.reshape((-1, 1)) * emp_cov * D_inv
#     r_bar = (np.sum(corr) - np.sum(np.diag(corr))) / N

#     target = D.reshape((-1, 1)) * np.full((n_features, n_features), r_bar) * D
#     target.flat[::n_features + 1] = np.diag(emp_cov)

#     return target


# def lw_const_corr_shrinkage(X, emp_cov, target, centered=False):
#     X = np.asarray(X)
#     # for only one feature, the result is the same whatever the shrinkage
#     if len(X.shape) == 2 and X.shape[1] == 1:
#         warnings.warn("Only one feature available. ")
#     if X.ndim == 1:
#         X = np.reshape(X, (1, -1))

#     if X.shape[0] == 1:
#         warnings.warn("Only one sample available. "
#                       "You may want to reshape your data array")
#     n_samples, n_features = X.shape

#     # optionally center data
#     if not centered:
#         X = X - X.mean(0)

#     D = np.sqrt(np.diag(emp_cov))
#     D_inv = 1. / D

#     N = n_features**2 - n_features
#     corr = D_inv.reshape((-1, 1)) * emp_cov * D_inv
#     r_bar = (np.sum(corr) - np.sum(np.diag(corr))) / N

#     X_sq = X**2
#     X_cu = X**3
#     s_11 = emp_cov
#     s_22 = np.dot(X_sq.T, X_sq) / X_sq.shape[0]
#     s_31 = np.dot(X_cu.T, X) / X_sq.shape[0]
#     s_ii = np.diag(s_11).reshape((-1, 1))
#     s_jj = np.diag(s_11)

#     pi = s_22 - s_11**2
#     pi_hat = np.sum(pi)

#     theta_ii = s_31 - s_ii * s_11
#     theta_ii.flat[::n_features + 1] = 0.
#     tmp1 = np.sqrt(s_jj / s_ii) * theta_ii
#     rho_hat = np.sum(np.diag(pi)) + r_bar * np.sum(tmp1)

#     gamma_hat = np.sum((target - emp_cov)**2)

#     kappa_hat = (pi_hat - rho_hat) / gamma_hat
#     delta_hat = max(0., min(kappa_hat / n_samples, 1.))
#     return delta_hat


# def test_lw_const_corr_shrinkage(X, s, centered=False):
#     X = np.asarray(X)
#     # for only one feature, the result is the same whatever the shrinkage
#     if len(X.shape) == 2 and X.shape[1] == 1:
#         warnings.warn("Only one feature available. ")
#     if X.ndim == 1:
#         X = np.reshape(X, (1, -1))

#     if X.shape[0] == 1:
#         warnings.warn("Only one sample available. "
#                       "You may want to reshape your data array")
#     n_samples, n_features = X.shape

#     # optionally center data
#     if not centered:
#         X = X - X.mean(0)

#     c = 2 / ((n_features - 1) * n_features)
#     r_bar = 0.
#     for i in range(n_features - 1):
#         for j in range(i + 1, n_features):
#             r_bar += c * s[i, j] / np.sqrt(s[i, i] * s[j, j])

#     target = np.zeros_like(s)
#     for i in range(n_features):
#         for j in range(i, n_features):
#             if i == j:
#                 target[i, i] = s[i, i]
#             else:
#                 target[i, j] = target[j, i] = r_bar * np.sqrt(s[i, i] * s[j, j])

#     pi = np.zeros_like(s)
#     for i in range(n_features):
#         for j in range(i, n_features):
#             pi[i, j] = pi[j, i] = ((X[:, i] * X[:, j] - s[i, j])**2).mean()

#     pi_hat = pi.sum()

#     theta_ii = np.zeros_like(s)
#     for i in range(n_features):
#         tmp = X[:, i]**2 - s[i, i]
#         for j in range(n_features):
#             if i == j:
#                 continue
#             else:
#                 tmp2 = (tmp * (X[:, i] * X[:, j] - s[i, j])).mean()
#                 theta_ii[i, j] = tmp2

#     theta_jj = np.zeros_like(s)
#     for j in range(n_features):
#         tmp = X[:, j]**2 - s[j, j]
#         for i in range(n_features):
#             if i == j:
#                 continue
#             else:
#                 tmp2 = (tmp * (X[:, i] * X[:, j] - s[i, j])).mean()
#                 theta_jj[i, j] = tmp2

#     rho_hat = 0.
#     for i in range(n_features):
#         for j in range(n_features):
#             if i == j:
#                 rho_hat += pi[i, i]
#             else:
#                 tmp1 = np.sqrt(s[j, j] / s[i, i]) * theta_ii[i, j]
#                 tmp2 = np.sqrt(s[i, i] / s[j, j]) * theta_jj[i, j]
#                 rho_hat += r_bar / 2. * (tmp1 + tmp2)

#     gamma_hat = np.sum((target - s)**2)

#     kappa_hat = (pi_hat - rho_hat) / gamma_hat
#     delta_hat = max(0., min(kappa_hat / n_samples, 1.))
#     return target, delta_hat


# def lw_const_corr_covariance(X, centered=False, return_shrink=False):
#     """Estimates the shrunk Ledoit-Wolf covariance matrix with constant
#     correlation target.
#     Read more in the :ref:`User Guide <shrunk_covariance>`.
#     Parameters
#     ----------
#     X : array-like, shape (n_samples, n_features)
#         Data from which to compute the covariance estimate
#     assume_centered : boolean, default=False
#         If True, data are not centered before computation.
#         Useful to work with data whose mean is significantly equal to
#         zero but is not exactly zero.
#         If False, data are centered before computation.
#     block_size : int, default=1000
#         Size of the blocks into which the covariance matrix will be split.
#         This is purely a memory optimization and does not affect results.
#     Returns
#     -------
#     shrunk_cov : array-like, shape (n_features, n_features)
#         Shrunk covariance.
#     shrinkage : float
#         Coefficient in the convex combination used for the computation
#         of the shrunk estimate.
#     Notes
#     -----
#     The regularized (shrunk) covariance is:
#     (1 - shrinkage)*cov
#       + shrinkage * Const_Corr_Target

#     where Const_Corr_Target is the constant correlation target of Ledoit,
#     Olivier and Michael Wolf (2004b). “Honey, I shrunk the sample covariance
#     matrix”. In: The Journal of Portfolio Management 30.4, pp. 110–119.
#     """
#     X = np.asarray(X)
#     # for only one feature, the result is the same whatever the shrinkage
#     if len(X.shape) == 2 and X.shape[1] == 1:
#         if not centered:
#             X = X - X.mean()
#         return np.atleast_2d((X ** 2).mean()), 0.
#     if X.ndim == 1:
#         X = np.reshape(X, (1, -1))
#         warnings.warn("Only one sample available. "
#                       "You may want to reshape your data array")
#         n_samples = 1
#         n_features = X.size
#     else:
#         n_samples, n_features = X.shape

#     # get Ledoit-Wolf shrinkage
#     emp_cov = empirical_covariance(X, centered=centered)
#     target = lw_const_corr_target(X, emp_cov, centered=centered)
#     shrinkage = lw_const_corr_shrinkage(X, emp_cov, target, centered=centered)
#     shrunk_cov = shrunk_covariance(emp_cov, target, shrinkage)

#     return shrunk_cov, shrinkage


# class ConstantCorrelationShrunkCovariance(ShrunkCovariance):

#     def fit(self, X, y=None):
#         """ Fits the shrunk covariance model
#         according to the given training data and parameters.
#         Parameters
#         ----------
#         X : array-like, shape = [n_samples, n_features]
#             Training data, where n_samples is the number of samples
#             and n_features is the number of features.
#         y : not used, present for API consistence purpose.
#         Returns
#         -------
#         self : object
#             Returns self.
#         """
#         X = check_array(X)
#         X = check_array(X)
#         if self.assume_centered:
#             self.location_ = np.zeros(X.shape[1])
#         else:
#             self.location_ = X.mean(0)
#         emp_cov = empirical_covariance(X - self.location_, centered=True)
#         target = lw_const_corr_target(
#             X - self.location_, emp_cov, centered=True)
#         covariance = shrunk_covariance(emp_cov, target, self.shrinkage)
#         self._set_covariance(covariance)

#         return self


# class LedoitWolfConstantCorrelationShrunkCovariance(ShrunkCovariance):

#     def fit(self, X, y=None):
#         """ Fits the shrunk covariance model
#         according to the given training data and parameters.
#         Parameters
#         ----------
#         X : array-like, shape = [n_samples, n_features]
#             Training data, where n_samples is the number of samples
#             and n_features is the number of features.
#         y : not used, present for API consistence purpose.
#         Returns
#         -------
#         self : object
#             Returns self.
#         """
#         X = check_array(X)
#         X = check_array(X)
#         if self.assume_centered:
#             self.location_ = np.zeros(X.shape[1])
#         else:
#             self.location_ = X.mean(0)
#         covariance, shrinkage = lw_const_corr_covariance(
#             X - self.location_, assume_centered=True, return_shrink=True)

#         self.shrinkage_ = shrinkage
#         self._set_covariance(covariance)

#         return self


# def cv_const_corr_shrunk_covariance(X, centered=False, n_splits=100,
#                                     test_size=.1, random_state=42,
#                                     shrinkages=np.logspace(-2, 0, 30),
#                                     return_shrink=False):
#     LOGGER.info("computing optimally shrunk covariance via cross-validation")
#     cov, cv_shrink = cv_covariance(
#         X, cov_model=ConstantCorrelationShrunkCovariance, centered=centered,
#         n_splits=n_splits, test_size=test_size, random_state=random_state,
#         shrinkages=shrinkages, return_shrink=True)

#     _, lw_shrink = lw_covariance(X, centered=centered, return_shrink=True)
#     _, oas_shrink = oas_covariance(X, centered=centered, return_shrink=True)
#     LOGGER.info("CV optimal shrinkage: {}".format(cv_shrink))
#     LOGGER.info("LW optimal shrinkage: {}".format(lw_shrink))
#     LOGGER.info("OAS optimal shrinkage: {}".format(oas_shrink))

#     return cov


# cov_mats = {
#     'emp': empirical_covariance,
#     'diag_shrunk': diag_shrunk_covariance,
#     'lw': lw_covariance,
#     'oas': oas_covariance,
#     'const_corr': lw_const_corr_covariance,
#     'diag_cv': cv_diag_shrunk_covariance,
#     'const_corr_cv': cv_const_corr_shrunk_covariance
# }
