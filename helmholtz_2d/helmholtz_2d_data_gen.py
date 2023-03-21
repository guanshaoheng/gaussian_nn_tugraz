import os.path

import matplotlib.pyplot as plt
import numpy as np
from kernel_2d import kernel_bessel_2d, input_2d
from utils_general.utils import echo, check_mkdir


def data_gen(n=20, save_path='xy_data', gaussian_flag=True, noise=0.1):
    x, X, Y = input_2d(num=n)
    if gaussian_flag:
        kernel = kernel_bessel_2d(x=x, k=2.0)
        y = np.random.multivariate_normal(mean=np.zeros(n**2), cov=kernel)
    else:
        omega = 2.0
        phi = 0.2
        A = 1.0
        y = A * np.sin(omega * (x[:, 0] + x[:, 1]) + phi)

    # save the data
    y_noise = y + np.random.normal(0, 1, len(y))*noise
    data_save(x=x, y=y, y_noise=y_noise)

    # plot the data Truth
    plt.contourf(X, Y, y.reshape(n, n))
    plt.colorbar()
    plt.tight_layout()
    fig_path_name = os.path.join(save_path, 'contourf_helmholtz_Truth.png')
    plt.savefig(fig_path_name, dpi=200)
    echo('fig saved as %s' % fig_path_name)
    plt.close()

    # plot the data noise
    plt.contourf(X, Y, y_noise.reshape(n, n))
    plt.colorbar()
    plt.tight_layout()
    fig_path_name = os.path.join(save_path, 'contourf_helmholtz_noise.png')
    plt.savefig(fig_path_name, dpi=200)
    echo('fig saved as %s' % fig_path_name)
    plt.close()
    return


def data_save(x, y, y_noise, save_path='xy_data'):
    check_mkdir(save_path)
    x_name = os.path.join(save_path, 'x.npy')
    np.save(x_name, x)
    np.save(os.path.join(save_path, 'y.npy'), y)
    np.save(os.path.join(save_path, 'y_noise.npy'), y_noise)
    index = np.random.permutation(np.arange(len(x)))
    np.save(os.path.join(save_path, 'index.npy'), index)
    echo('Data saved as %s' % x_name)


def data_load(save_path = 'xy_data'):
    x = np.load(os.path.join(save_path, 'x.npy'))
    y = np.load(os.path.join(save_path, 'y.npy'))
    y_noise = np.load(os.path.join(save_path, 'y_noise.npy'))
    index = np.load(os.path.join(save_path, 'index.npy'))
    return x, y, y_noise, index


if __name__ == '__main__':
    data_gen(gaussian_flag=True)

