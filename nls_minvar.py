

def minvar_vanilla_loo_isotonic(sim, smoothing='average', nonnegative=False,
                                regularization=None):
    """
    Base Estimator 1: MinVar Leave-One-Out Joint Cross-Validation with Isotonic
    Regression

    Variants:
     + Smoothing could be average or median.
     + Nonnegativity constraint
     + Regularization: 'l2' for now, maybe others
    """
    pass


def minvar_vanilla_kfold_isotonic(sim, K, smoothing='average',
                                  nonnegative=False, regularization=None):
    """
    Base Estimator 2: MinVar $K$-Fold Cross-Validation with Isotonic Regression

    Parameters:
     + K

    Variants:
     + Smoothing could be average or median.
     + Nonnegativity constraint
     + Regularization: 'l2' for now, maybe others
    """
    pass


def minvar_joint_loo_isotonic(sim, smoothing='average', nonnegative=False,
                              regularization=None):
    """
    Base Estimator 3: MinVar Leave-One-Out Joint Cross-Validation with Isotonic
    Regression

    Parameters:
     + K

    Variants:
     + Smoothing could be average or median.
     + Nonnegativity constraint
     + Regularization: 'l2' for now, maybe others
    """
    pass


def minvar_joint_kfold_isotonic(sim, K, smoothing='average', nonnegative=False,
                                regularization=None):
    """
    Base Estimator 4: MinVar $K$-Fold Joint Cross-Validation with Isotonic
    Regression

    Parameters:
     + K

    Variants:
     + Smoothing could be average or median.
     + Nonnegativity constraint
     + Regularization: 'l2' for now, maybe others
    """
    pass
