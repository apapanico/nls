
def shrunk_covariance(emp_cov, target, shrinkage):
    return shrinkage * emp_cov + (1 - shrinkage) * target
