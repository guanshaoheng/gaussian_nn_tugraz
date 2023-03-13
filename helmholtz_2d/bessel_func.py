import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv


def bessel_func(v, x):
    '''

    :param v:  number of the alpha
    :param x:  vector of the input
    :return:
    '''
    return jv(v, x)


if __name__ == '__main__':
    x = np.linspace(0, 10, 100)
    num = bessel_func(v=0, x=x)
    plt.plot(x, num); plt.show()