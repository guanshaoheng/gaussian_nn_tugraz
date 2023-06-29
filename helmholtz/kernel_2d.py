import numpy as np
from bessel_func import bessel_func
import matplotlib.pyplot as plt


def kernel_bessel_2d(x, k):
    r'''
     this kernel func is referred to the paper
         [1] C. Albert, “Physics-informed transfer path analysis with parameter
             estimation using Gaussian processes,” Proc. Int. Congr. Acoust., vol. 2019-Septe, no. 1, pp. 459–466, 2019, doi: 10.18154/RWTH-CONV-238988.
    :param x:  in shape of (num_samples, (input_1, input_2))
    :param k:  $J_0( k \| x-x' \| )$
    :return:
    '''
    distance = cal_distance(x)
    kernel = bessel_func(v=0, x=k*distance)
    # plt.imshow(kernel); plt.colorbar();plt.show()
    return kernel


def cal_distance(x):
    '''
        calculate the distance of the input
    :param x: in shape of (num_samples, (input_1, input_2))
    :return:
    '''
    n = len(x)
    distance = np.zeros(shape=[n, n])
    for i in range(n):
        for j in range(n):
            distance[i, j] = np.linalg.norm(x[i] - x[j])
    return distance


def input_2d(min_=-1, max_=1, num=11):
    x = np.linspace(min_, max_, num)
    X = np.meshgrid(x, x)
    xy = np.concatenate((X[0].reshape(-1, 1), X[1].reshape(-1, 1)), axis=1)
    return xy, X[0], X[1]


def main():
    x = input_2d()
    kernel = kernel_bessel_2d(x=x, k=2.)
    plt.imshow(kernel); plt.colorbar(); plt.show()


if __name__ == '__main__':
    main()


