import sys
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from dataset_gen_gaussian import kernel_cos, kernel_rbf
sys.path.append('../')


'''
    This script shows how to sample GP with mean and kernel.
'''


def main(n=100):
    x = np.linspace(0, 1., n)
    # cov = kernel_cos(x, k=5.0) + np.eye(n)*1e-10
    cov = kernel_rbf(x, x, sig=2., l=0.2) + np.eye(n)*1e-10
    plt.imshow(cov); plt.colorbar();plt.tight_layout(); plt.show()
    L = np.linalg.cholesky(cov)
    # L_inv = numpy.linalg.inv(L)
    for i in range(8):
        rand_x = np.random.randn(len(x))[:, np.newaxis]
        y_ = L @ rand_x
        plt.plot(x, y_)
    plt.show()

    # add the first constraints to the regression
    n_train = 1
    x_train = np.array([0]).reshape(-1, 1)
    y_train = np.array([0]).reshape(-1, 1)

    k_train = kernel_rbf(x_train, x_train, sig=2., l=0.2)
    L = np.linalg.cholesky(k_train + np.eye(n_train)*1e-10)
    k_half = kernel_rbf(x, x_train, sig=2., l=0.2)
    v = np.linalg.solve(L, k_half.T)
    kernel_train = cov - v.T@v

    L = np.linalg.cholesky(kernel_train + np.eye(n)*1e-10)

    for i in range(2):
        rand_x = np.random.randn(len(x))[:, np.newaxis]
        y_ = L @ rand_x
        plt.plot(x, y_, label='home-made')

    for i in range(2):
        y_ = np.random.multivariate_normal(mean=np.zeros(n), cov=kernel_train)
        plt.plot(x, y_, 'r.', label='numpy-func')
    plt.legend(); plt.tight_layout();plt.show()


if __name__ == '__main__':
    main()


