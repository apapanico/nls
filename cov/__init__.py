

from .emp import empirical_covariance
from .diag import (lw_covariance, oas_covariance,
                   cv_diag_shrunk_covariance)
from .constcorr import (lw_const_corr_covariance,
                        cv_const_corr_shrunk_covariance)
from .nlshrink import nlshrink_covariance

__all__ = [
    'cov_mats',
    'empirical_covariance',
    'lw_covariance',
    'oas_covariance',
    'lw_const_corr_covariance',
    'cv_diag_shrunk_covariance',
    'cv_const_corr_shrunk_covariance'
]


cov_mats = {
    'emp': empirical_covariance,
    'lw': lw_covariance,
    'oas': oas_covariance,
    'diag_cv': cv_diag_shrunk_covariance,
    'cc': lw_const_corr_covariance,
    'cc_cv': cv_const_corr_shrunk_covariance,
    'nlshrink': nlshrink_covariance
}
