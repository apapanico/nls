import joblib
from utils import eig, cov, sample, annualize_vol


class Simulation(object):

    def __init__(self, Sigma, T):
        self.Sigma = Sigma
        self.tau, self.V = eig(Sigma)
        self.N = Sigma.shape[0]
        self.T = T
        self.sample()

    def sample(self):
        self.X = X = sample(self.Sigma, self.T)
        self.cov_est()
        self._hash = hash(joblib.hash(X))

    def __hash__(self):
        return self._hash

    def cov_est(self):
        self.S = S = cov(self.X)
        self.lam, self.U = eig(S)

    @property
    def shape(self):
        return self.X.shape

    @property
    def lam_1(self):
        return self.lam[0]

    @property
    def lam_N(self):
        return self.lam[-1]

    @property
    def vols(self, pop=False):
        eigvals = self.tau if pop else self.lam
        return annualize_vol(eigvals / self.N)
