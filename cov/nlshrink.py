import logging
import numpy as np

from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

import sys


class DummyFile(object):
    def write(self, x):
        pass


def nostdout(f):

    def wrapper(*args, **kwargs):
        save_stdout = sys.stdout
        sys.stdout = DummyFile()
        out = f(*args, **kwargs)
        sys.stdout = save_stdout
        return out

    return wrapper


r_nlshrink = importr('nlshrink')


LOGGER = logging.getLogger(__name__)


def nlshrink_covariance(X, centered=False):
    LOGGER.info("computing Ledoit-Wolf non-linear shrinkage covariance")
    if not centered:
        X = X - X.mean()

    f = nostdout(r_nlshrink.nlshrink_cov)
    numpy2ri.activate()
    cov = np.asarray(f(X))
    numpy2ri.deactivate()
    return cov
