from utils import *
from models import *
# from nls_minvar_new import minvar_joint_kfold_isotonic
from nls_minvar import minvar_joint_kfold_isotonic
from experimental.classes import Simulation

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd
from cvxpy import *

seed = 1532
gamma = None


def isotonic_regression(y, y_min=None, y_max=None):
  """Wrapper around SKlearn's isotonic regression"""
  return sk_isotonic_regression(y, y_min=y_min, y_max=y_max, increasing=False)


'''Set parameters '''

cov_model = 'H'
N = 100
T = 200
K = 10


'''Simulate some data '''
Sigma = cov_functions[cov_model](N)[0]
sim = Simulation(Sigma, T)
sim.seed = seed

sim.sample()

''' Do the Cross Validation part '''

T, N = sim.shape
m = int(T / K)
X, S = sim.X, sim.S

A = []
b = []

for k in range(K):
  k_set = list(range(k * m, (k + 1) * m))
  _k = np.delete(range(T), k_set)
  X_k = X[k_set, :]
  S_k = 1 / (T - m) * X[_k, :].T @ X[_k, :]  # (T * S - X_k.T @ X_k) / (T - m)
  _, U_k = eig(S_k)
  alpha_k = U_k.T @ np.ones(N)
  C_k = U_k.T @ (1 / m * X_k.T @ X_k) @ U_k
  A_k = np.diag(alpha_k)
  A.append(C_k @ A_k)
  b.append(alpha_k)

''' obtain stacked A and b matrices '''

A = np.vstack(A)  # (NxK) x N
b = np.hstack(b)  # (NxK) x 1


''' set monotonicity = isotonic constraint '''

G = (np.diag(np.ones(N), k=0) - np.diag(np.ones(N - 1), k=1))
Ginv = np.triu(np.ones((N, N)))

''' set trace constraint '''

Sinv = np.linalg.inv(S)
trace = sum(np.diag(Sinv))

''' Use Gershgorin circle theorem to estimate the circle in which the
    eigenvalues should lie: 
        
    |lambda-A_ii| <= sum(A_ij): i!=j, i = 1...n '''

''' calculate sample and population matrix statistics '''

av_sinv = np.mean(eig(Sinv)[0])
sd_sinv = np.std(eig(Sinv)[0])

av_s = np.mean(sim.lam)
sd_s = np.std(sim.lam)

av_t = np.mean(sim.tau)
sd_t = np.std(sim.tau)

''' create cvxpy problem statetement '''

x = Variable(N)
objective = Minimize(sum_squares(A @ x - b))
constraints = [
    G @ x >= 0,
    x >= 0,
    sum(x) == trace
]
prob = Problem(objective, constraints)
result = prob.solve()

z = np.array(x.value).ravel()[::-1]

print(np.any(z < 0.))

''' scale 1/z to sample mean and std '''

d_new = 1 / z
# d_new = (d_new - np.mean(d_new)) / (np.std(d_new))
# d_new = (d_new * sd_s) + av_s

''' calculate the condition number of the the matrix:
    it is >>1 '''

np.abs(max(d_new) / min(d_new))

''' calculate d with previous estimator '''

d_old = minvar_joint_kfold_isotonic(sim, K, nonnegative=True)

''' plot spectrum '''

plt.plot(d_new, color='orange')
plt.plot(d_old, color='red')
plt.plot(sim.tau, color='black')
plt.plot(sim.lam, color='gray')
plt.title('Spectrum')

patch1 = mpatches.Patch(color='orange', label='New joint estimator')
patch2 = mpatches.Patch(color='red', label='Old joint estimator')
patch3 = mpatches.Patch(color='black', label='True values')
patch4 = mpatches.Patch(color='gray', label='Sample values')

plt.legend(handles=[patch1, patch2, patch3, patch4])

plt.show()

''' calculate covariance matrices '''

Sigma_est_old = eig_multiply(sim.U, d_old)
Sigma_est_new = eig_multiply(sim.U, d_new)

'''Problem: we get singular covariance matrix with negative diagonals

   Solutions: 
       1) Regularize covariance matrix(done):
            choose lambda such that the quadratic loss function R is minimized ? 
       2) Zero elements of covariance matrix:
            + for every a_i,a_j : i,j = 1...N, i<j , calculate correlation r_ij between features
            + if t-stat for r_ij is not significant - zero the element
       3) Diagonalize covariance matrix  

    Most likely 2,3 won`t help since the diagonals are negative

 '''

'''Solution 1'''
lmbda = 0.001545
I = np.diag(np.ones(N))
Sigma_est_new = (Sigma_est_new + lmbda * I)

''' check results in terms of variance ratios '''

pi_true = min_var_portfolio(Sigma, gamma=gamma)

var_new = portfolio_analysis(Sigma_est_new, Sigma, gamma, pi_true)
var_old = portfolio_analysis(Sigma_est_old, Sigma, gamma, pi_true)
var_sample = portfolio_analysis(S, Sigma, gamma, pi_true)

''' print portfolio stat '''

stat = pd.DataFrame(np.zeros((5, 3)),
                    index=['forecast_var_ratio', 'is_var',
                           'oos_var', 'te', 'true_var_ratio'],
                    columns=['New estimator', 'Old estimator', 'Sample estimator'])

stat['New estimator'] = [var_new['forecast_var_ratio'], var_new['is_var'],
                         var_new['oos_var'], var_new['te'], var_new['true_var_ratio']]
stat['Old estimator'] = [var_old['forecast_var_ratio'], var_old['is_var'],
                         var_old['oos_var'], var_old['te'], var_old['true_var_ratio']]
stat['Sample estimator'] = [var_sample['forecast_var_ratio'], var_sample['is_var'],
                            var_sample['oos_var'], var_sample['te'], var_sample['true_var_ratio']]

print(stat)


#np.abs(Sigma_est_new).sum(axis=1) - np.abs(np.diag(Sigma_est_new))
