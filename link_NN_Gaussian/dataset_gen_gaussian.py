import numpy as np
import matplotlib.pyplot as plt
from utils_general.utils import check_mkdir, echo
import os


def gaussian_data_gen(
        xmin=0., xmax=1.0, k=3.*np.pi, num_points=100, noise=0.1):
    x = np.linspace(xmin, xmax, num_points)
    K = kernel_cos(x=x, k=k)
    # plt.imshow(K); plt.colorbar();plt.show()

    y = np.random.multivariate_normal(mean=np.zeros_like(x), cov=K)
    y_noise = y + np.random.normal(0, 1., len(y))*(np.max(y) - np.min(y))*0.5 * noise
    return x.reshape(-1, 1), y.reshape(-1, 1), y_noise.reshape(-1, 1)


def kernel_cos(x, k=1.):
    '''
            cos(k(x, x'))
    :param x:  in shape of (num_points,)
    :param k:
    :return:
    '''
    x = x.reshape(-1, 1)
    K = np.cos(k*(x-x.T))
    return K


def kernel_rbf(x1, x2, sig=1.0, l=1.0):
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)
    K = sig**2 * np.exp(-(x1 - x2.T)**2/(2*l**2))
    return K

def show_the_gp(x, y, y_noise):
    # x, y, y_noise = gaussian_data_gen()
    plt.plot(x, y, label='Truth')
    plt.plot(x, y_noise, label='Noise')
    plt.show()


def data_save(x, y, y_noise, save_path='xy_data'):
    check_mkdir(save_path)
    np.save(os.path.join(save_path, 'x.npy'), x)
    np.save(os.path.join(save_path, 'y.npy'), y)
    np.save(os.path.join(save_path, 'y_noise.npy'), y_noise)
    index = np.random.permutation(np.arange(len(x)))
    np.save(os.path.join(save_path, 'index.npy'), index)


def data_load(save_path = 'xy_data'):
    x = np.load(os.path.join(save_path, 'x.npy'))
    y = np.load(os.path.join(save_path, 'y.npy'))
    y_noise = np.load(os.path.join(save_path, 'y_noise.npy'))
    index = np.load(os.path.join(save_path, 'index.npy'))
    return x, y, y_noise, index


if __name__ == '__main__':
    x, y, y_noise = gaussian_data_gen()
    data_save(x, y, y_noise)
    show_the_gp(x, y, y_noise)
