import logging

from sklearn.covariance import empirical_covariance as sk_empirical_covariance


LOGGER = logging.getLogger(__name__)


def empirical_covariance(X, centered=False):
    LOGGER.info("computing empirical covariance")
    return sk_empirical_covariance(X, assume_centered=centered)
