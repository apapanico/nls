from models import *
import shrinkage
import shrinkage_new
from utils import *
import numpy as np
import time
import sys
import matplotlib.pyplot as plt
from classes import Simulation

f = shrinkage.f


def str_to_bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False


def compare(d1, d2, alpha, C):
    print("l2 err:", np.linalg.norm(d1 - d2))
    print("discrepancy:", np.max(np.abs(d1 - d2)))
    f1 = sum([f(d1, _alpha, _C) for _C, _alpha in zip(C, alpha)])
    print("f eval:", f1)
    f2 = sum([f(d2, _alpha, _C) for _C, _alpha in zip(C, alpha)])
    print("f eval:", f2)


N, upper_bound = int(sys.argv[1]), str_to_bool(sys.argv[2])
print(N, upper_bound)

y = 2

T = y * N
Sigma, tau = SLR_cov(N, seed=3823)

np.random.seed(4328)

sim = Simulation(Sigma, T)
X, Sigma, S, lam, U = sim.X, sim.Sigma, sim.S, sim.lam, sim.U


K = 10

if K == 1:
    alpha = [U.T.dot(np.ones(N))]
    C = [U.T.dot(Sigma).dot(U)]
else:
    m = int(T / K)

    C = []
    alpha = []
    for k in range(K):
        k_set = list(range(k * m, (k + 1) * m))
        X_k = X[k_set, :]
        S_k = (T * S - X_k.T.dot(X_k)) / (T - m)
        _, U_k = eig(S_k)
        C.append(U_k.T.dot(Sigma).dot(U_k))
        alpha.append(U_k.T.dot(np.ones(N)))

_, tmp = shrinkage.nls_kfold(X, S, U, K=10)
d0 = shrinkage.isotonic_regression(tmp)
d0_new = shrinkage_new.nls_kfold(sim, 10, isotonic=True)


trace = np.sum(d0)
d_min, d_max = lam[-1], lam[0]


t = time.time()
d_multi = shrinkage.minvar_nls_nlsq_multi(
    C, alpha, trace, d0, d_min, d_max, upper_bound)
print("elapsed: {:.2f}".format(time.time() - t))

t = time.time()
d_multi_new = shrinkage_new.minvar_nlsq_multi(
    C, alpha, trace, d0, d_min, d_max, True, upper_bound)
print("elapsed: {:.2f}".format(time.time() - t))


t = time.time()
d_transform = shrinkage.minvar_nls_nlsq_multi_transformed(
    C, alpha, trace, d0, d_min, d_max, upper_bound)
print("elapsed: {:.2f}".format(time.time() - t))

t = time.time()
d_transform_new = shrinkage_new.minvar_nlsq_multi_transformed(
    C, alpha, trace, d0, d_min, d_max, upper_bound)
print("elapsed: {:.2f}".format(time.time() - t))

compare(d0, d0_new, alpha, C)
compare(d_multi, d_multi_new, alpha, C)
compare(d_transform, d_multi, alpha, C)
compare(d_transform, d_multi_new, alpha, C)
compare(d_transform, d_transform_new, alpha, C)
compare(d_transform_new, d_multi, alpha, C)
compare(d_transform_new, d_multi_new, alpha, C)


compare(d_multi, d0, alpha, C)
compare(d_multi_new, d0, alpha, C)
compare(d_transform, d0, alpha, C)
compare(d_transform_new, d0, alpha, C)

# d2 = minvar_nls_nlsq_multi_transformed(
#     C, alpha, trace, d0, d_min, d_max, upper_bound)
# print("elapsed: {:.2f}".format(time.time() - t))

# print("discrepancy:", np.max(np.abs(d1 - d2)))

# print("f eval:", f(d1, alpha, C))
# print("f eval:", f(d2, alpha, C))

# t = np.linspace(0., 1., len(d1))

# plt.plot(t, tau)
# plt.plot(t, d0)
# plt.plot(t, d1)
# plt.plot(t, d2)
# # plt.plot(t, d3)
# plt.show()
