import numpy as np
import matplotlib.pyplot as plt


def H(N, gam=2., a=1, b=2):
    tau = np.linspace(a, b, N)
    H = (1 - np.exp(-gam * (tau - a))) / (1 - np.exp(-gam * (b - a)))
    return H


def h(N, gam=2., a=1, b=2):
    tau = np.linspace(a, b, N)
    h = gam * np.exp(-gam * (tau - a)) / (1 - np.exp(-gam * (b - a)))
    return h


def H_inv(N, gam=2., a=1, b=2):
    x = np.linspace(0, 1, N)
    tau = a - np.log(1 - x * (1 - np.exp(-gam * (b - a)))) / gam
    return tau


def Sigma(tau, random_U=True):
    N = len(tau)
    if random_U:
        U = haar_measure(N)
        sigma = U.dot(np.diag(tau)).dot(U.T)
    else:
        sigma = np.diag(tau)
    return sigma


def haar_measure(N):
    """A Random matrix distributed with Haar measure"""
    z = np.random.randn(N, N)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.absolute(d)
    q = np.multiply(q, ph, q)
    return q


def sample(sig, T):
    N = sig.shape[0]
    X = np.random.multivariate_normal(np.zeros(N), sig, T)
    return X


def eig(sig, vecs=False):
    if vecs:
        return np.linalg.eigh(sig)
    else:
        return np.linalg.eigvalsh(sig)


def cov(X):
    return np.cov(X, rowvar=False)


def cov_eigs(sig):
    vals = np.sort(eig(sig))
    return vals


def min_var(sig):
    n = sig.shape[0]
    w = np.linalg.solve(sig, np.ones(n))
    w /= np.ones(n).dot(w)
    return w


def rotate_portfolio(w, sig):
    D, U = eig(sig, vecs=True)
    return U.T.dot(w)


if __name__ == "__main__":

    y = 2
    N = 100
    T = 100

    tau = H_inv(N, gam=5)
    sig = Sigma(tau)
    X = sample(sig, y * N)

    lam = cov_eigs(cov(X))

    fig, ax = plt.subplots()
    plt.plot(tau, '.', label="True Eigenvalues")
    plt.plot(lam, 'x', label="Sample Eigenvalues")
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    plt.plot(np.linspace(0, 1, N), tau, '.', label="True Eigenvalues")
    for N, mark, ms in [(50, 'x', 6), (500, 'o', 4), (1000, '+', 2)]:
        tau = H_inv(N, gam=5)
        sig = Sigma(tau)
        X = sample(sig, y * N)
        lam = cov_eigs(cov(X))
        plt.plot(np.linspace(0, 1, N), lam, mark, ms=ms,
                 label="Sample Eigenvalues: N = {}".format(N))
    plt.legend()
    plt.title('Sample Eigenvalues vs True Eigenvalues')
    plt.show()

    fig, ax = plt.subplots()
    N = 50
    tau = H_inv(N, gam=5)
    sig = Sigma(tau)
    plt.plot(tau, '.', label="True Eigenvalues")
    for T, mark, ms in [(100, 'x', 6), (500, 'o', 4), (5000, '+', 2)]:
        X = sample(sig, T)
        lam = cov_eigs(cov(X))
        plt.plot(lam, mark, ms=ms, label="Sample Eigenvalues: N = {}".format(N))
    plt.legend()
    plt.title('Sample Eigenvalues vs True Eigenvalues')
    plt.show()

    # fig, ax = plt.subplots(ncols=2)
    # sig = Sigma(tau, random_U=False)
    # w = min_var(sig)
    # alpha = rotate_portfolio(w, sig)
    # ax[0].plot(np.linspace(0, 1, N), w, '.', label="True")
    # ax[1].plot(np.linspace(0, 1, N), alpha, '.', label="True")

    # fig, ax = plt.subplots()
    # ax.plot(np.linspace(0, 1, N), 1 / tau / np.sum(1 / tau), '.', label="True")
    # for N, mark, ms in [(50, 'x', 6), (500, 'o', 4), (1000, '+', 2)]:
    #     tau = H_inv(N, gam=5)
    #     sig = Sigma(tau)
    #     X = sample(sig, y * N)
    #     w_hat = min_var(cov(X))
    #     alpha_hat = rotate_portfolio(w_hat, cov(X))
    #     ax[0].plot(np.linspace(0, 1, N), w_hat, mark, ms=ms,
    #                label="N = {}".format(N))
    #     ax[1].plot(np.linspace(0, 1, N), alpha_hat, mark, ms=ms,
    #                label="N = {}".format(N))
    # plt.legend()
    # ax[0].set_title('Min Var Portfolio Weights True vs Sample')
    # ax[1].set_title('Rotated Min Var Portfolio Weights True vs Sample')
    # plt.show()

    for N in [50, 500, 1000]:
        x = np.linspace(0, 1, N)
        fig, ax = plt.subplots()
        tau = H_inv(N, gam=5)
        sig = Sigma(tau, random_U=True)
        D, U = eig(sig, vecs=True)
        w = min_var(sig)
        alpha = rotate_portfolio(w, sig)
        b = U.T.dot(np.ones(N))
        ax.plot(x, N * alpha, 'o', label="Normalized Weights")
        ax.plot(x, (b / tau) / (np.sum(b**2 / tau) / N), '.', label="Predicted")
        ax.set_title("N = {}".format(N))
        ax.legend()

        plt.show(block=False)
