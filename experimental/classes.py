import joblib
from utils import eig, cov, sample, annualize_vol
from models import * 


class Simulation(object):

    def __init__(self, Sigma, T):
        
        ''' Simulation class with given covariance matrix. Stores 
        popualtion eigenvalues and eigenvectors (tau, V respectively)
        ,number of data points (T) and features (N) '''
        
        self.Sigma = Sigma
        self.tau, self.V = eig(Sigma)
        self.N = Sigma.shape[0]
        self.T = T
        self.seed = None
        
    def sample(self):
        
        ''' Samples from multivariate normal distribution and 
         creates a matrix of returns given a population cov matrix '''
    
        self.X = X = sample(self.Sigma, self.T,self.seed)
        self.cov_est()
        self._hash = hash(joblib.hash(X))

    def __hash__(self):
        return self._hash

    def cov_est(self):
        
        ''' Calculates sample eigenvalues and eigenvectors from 
        matrix of returns X ''' 
        
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
        
        ''' Returns annualized population/sample 
        volatilities from eigenvalues'''
        
        eigvals = self.tau if pop else self.lam
        return annualize_vol(eigvals / self.N)
