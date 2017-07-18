

import numpy as np
from sklearn.model_selection import GridSearchCV, ShuffleSplit


def cv_covariance(X, cov_model,
                  centered=False, n_splits=100,
                  test_size=.1, random_state=42,
                  shrinkages=np.logspace(-2, 0, 30),
                  return_shrink=False):
    tuned_parameters = [{'shrinkage': shrinkages}]
    cv = ShuffleSplit(n_splits=n_splits, test_size=test_size,
                      random_state=random_state)
    cov_cv = GridSearchCV(
        cov_model(assume_centered=centered),
        tuned_parameters,
        cv=cv
    )
    cov_cv.fit(X)
    best_est = cov_cv.best_estimator_
    cov = best_est.covariance_

    if return_shrink:
        cv_shrink = best_est.shrinkage
        return cov, cv_shrink
    else:
        return cov
