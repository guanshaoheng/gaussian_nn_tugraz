import numpy as np
import math
from scipy.special import jv
import matplotlib.pyplot as plt

'''
    this is refereed to 
    https://stackoverflow.com/questions/74839356/how-to-write-bessel-function-using-power-series-method-in-python-without-sympy
'''

''' 
    Tk / Tk-1 = - (X/2)^2 /(k(k+N))
'''

def bessel_n_func(N, X):
    # First term
    X_temp = 0.5*X.copy()
    Term = pow(X_temp, N) / np.math.factorial(N)
    Sum = Term.copy()
    # print(Sum)

    # Next terms
    X_temp *= -X_temp
    for k in range(1, 50):
        Term *= X_temp / (k * (k + N))
        Sum += Term
        # print(Sum)
    return Sum

x = np.linspace(0, 10, 200)
y = bessel_n_func(X=x, N=0)
plt.plot(x, y)
plt.show()

print('number from %f' % jv(0, 5))


def residual_bessel(x):
    t = x**2
    x2 = x**2
    sum_ = x**2
    for n in range(1, 1000):
        t *= -x2/4.*(4*n**2+x2)/(4*(n-1)**2 + x2)/n**2
        sum_ += t
    return sum_

resi = residual_bessel(1.)
x = np.linspace(0.1, 10, 20)
resi = [residual_bessel(i) for i in x]

print()




