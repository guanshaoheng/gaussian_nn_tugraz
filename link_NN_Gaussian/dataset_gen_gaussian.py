import numpy as np
import matplotlib.pyplot as plt


def gaussian_data_gen(
        xmin=0., xmax=np.pi*2., k=1.0, num_points=100, num_samples=1):
    x = np.linspace(xmin, xmax, num_points)
    K = kernel_cos(x=x, k=k)
    y_list = []
    for i in range(num_samples):
        y_list.append(np.random.multivariate_normal(mean=np.zeros_like(x), cov=K))
    return x, y_list


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


def show_the_gp():
    x, y_list = gaussian_data_gen(num_samples=5)
    for i in range(len(y_list)):
        plt.plot(x, y_list[i])
    plt.show()


if __name__ == '__main__':
    show_the_gp()