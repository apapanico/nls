

from cvxpy import *
import numpy as np
import scipy.optimize
from sklearn.isotonic import isotonic_regression as sk_isotonic_regression


def isotonic_regression(y, y_min=None, y_max=None):
    """Wrapper around SKlearn's isotonic regression"""
    return sk_isotonic_regression(y, y_min=y_min, y_max=y_max, increasing=False)


np.set_printoptions(precision=5, suppress=True)
np.random.seed(193)

N = 20
K = 10

C_mats = [np.random.randn(N, N) for _ in range(K)]
z = np.linspace(1., 0., N)
alpha_vecs = [C.dot(z) + .5 * np.random.randn(N) for C in C_mats]

P = sum([C.T.dot(C) for C in C_mats])
q = -sum([C.T.dot(alpha) for C, alpha in zip(C_mats, alpha_vecs)])

print("LS/NNLS/Iso demo with numpy, scipy, and cvxpy")
print("=============================================")

print("True z (ordered and decreasing)")
print(z)
input()


# LS Solution
print("Solving basic LS problem with P and q (note the solution is not ordered or non-negative)")
z_ls = np.linalg.solve(P, -q)
print('z_ls:\n', z_ls)
input()

# LS Solution (alternate)
print("Solving basic LS problem with lsq_linear and stacked matrix and vector")
A = np.vstack(C_mats)
b = np.hstack(alpha_vecs)
z_ls_alt = scipy.optimize.lsq_linear(A, b).x
print('z_ls_alt:\n', z_ls_alt)
input()

# LS Solution (cvxpy)
print("Solving basic LS problem with CVXPY")
x = Variable(N)
objective = Minimize(sum_squares(A * x - b))
constraints = []
prob = Problem(objective, constraints)
result = prob.solve()
z_ls_cvx = np.array(x.value).ravel()
print('z_ls_cvx:\n', z_ls_cvx)
input()

# NNLS Solution
print("Solving NNLS problem (this is done incorrectly, stacked matrix/vector should be used as in next attempt)")
z_nnls = scipy.optimize.nnls(P, -q)[0]
print('z_nnls:\n', z_nnls)
input()

# NNLS Solution (alternate)
print("Solving NNLS problem with stacked matrix and vector")
z_nnls_alt = scipy.optimize.nnls(A, b)[0]
print('z_nnls_alt:\n', z_nnls_alt)
input()

# NNLS Solution (cvxpy)
print("Solving NNLS problem with CVXPY")
x = Variable(N)
objective = Minimize(sum_squares(A * x - b))
constraints = [0 <= x]
prob = Problem(objective, constraints)
result = prob.solve()
z_nnls_cvx = np.array(x.value).ravel()
mask = z_nnls_cvx < 1e-8
z_nnls_cvx[mask] = 0.
print('z_nnls_cvx:\n', z_nnls_cvx)
input()

# Isotonic LS Solution
print("Solving LS problem, post-process Isotonic regression")
z_ls_iso = isotonic_regression(z_ls)
print('z_ls_iso:\n', z_ls_iso)
input()

# Isotonic NNLS Solution
print("Solving NNLS problem, post-process Isotonic regression")
z_nnls_cvx_iso = isotonic_regression(z_nnls_cvx)
print('z_nnls_cvx_iso:\n', z_nnls_cvx_iso)
input()

# NNLS Solution w/ Monotonic constraint (cvxpy)
print("Solving LS problem with Isotonic constraint")
G = (np.diag(np.ones(N), k=0) - np.diag(np.ones(N - 1), k=1))
Ginv = np.triu(np.ones((N, N)))

x = Variable(N)
objective = Minimize(sum_squares(A * x - b))
constraints = [G * x >= 0]
prob = Problem(objective, constraints)
result = prob.solve()
z_ls_cvx_mono = np.array(x.value).ravel()
print('z_ls_cvx_mono:\n', z_ls_cvx_mono)
input()

# NNLS Solution w/ Monotonic constraint 2 (cvxpy)
print("Solving LS problem with Isotonic constraint, alternative")
A2 = A.dot(Ginv)

x = Variable(N)
objective = Minimize(sum_squares(A2 * x - b))
constraints = [x >= 0]
prob = Problem(objective, constraints)
result = prob.solve()
z_diff = np.array(x.value).ravel()
z_ls_cvx_mono2 = Ginv.dot(z_diff)
print('z_ls_cvx_mono2:\n', z_ls_cvx_mono2)
input()

# NNLS Solution w/ Monotonic constraint via scipy.optimize.nnls
print("Solving LS problem with Isotonic constraint via scipy.optimize.nnls")
z_diff = scipy.optimize.nnls(A2, b)[0]
z_ls_mono = Ginv.dot(z_diff)
print('z_ls_mono:\n', z_ls_mono)
input()


print("Compare monotonic solution optimality")

print("Basic LS (Note its optimal wrt f but not the constraint)")
f, sat = np.linalg.norm(A.dot(z_ls_iso) - b), np.sum(G.dot(z_ls_iso) < 0.) == 0
print("f:", f, "Gz >= 0", sat)

print("NNLS, post-process Iso (Note it's not optimal)")
f, sat = np.linalg.norm(A.dot(z_nnls_cvx_iso) -
                        b), np.sum(G.dot(z_nnls_cvx_iso) < 0.) == 0
print("f:", f, "Gz >= 0", sat)

print("NNLS w/ Iso constraint")
f, sat = np.linalg.norm(A.dot(z_ls_cvx_mono) -
                        b), np.sum(G.dot(z_ls_cvx_mono) < 0.) == 0
print("f:", f, "Gz >= 0", sat)

print("NNLS w/ Iso constraint (alternative)")
f, sat = np.linalg.norm(A.dot(z_ls_cvx_mono2) -
                        b), np.sum(G.dot(z_ls_cvx_mono2) < 0.) == 0
print("f:", f, "Gz >= 0", sat)

print("NNLS w/ Iso constraint (alternative #2)")
f, sat = np.linalg.norm(A.dot(z_ls_mono) -
                        b), np.sum(G.dot(z_ls_mono) < 0.) == 0
print("f:", f, "Gz >= 0", sat)

input()

print("Compare Isotonic regression and monotone constraint (note the slight difference")
print("NNLS, post-process Iso\n", z_nnls_cvx_iso)
print('NNLS w/ Iso constraint\n', z_ls_cvx_mono)

print("...exiting")
