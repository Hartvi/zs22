
from scipy.spatial.transform import Rotation
from scipy import integrate as itg
from scipy import optimize as opt
from typing import *
import numpy as np
import math as mt
from matplotlib import pyplot as plt
import mpmath as mpm

from class_w_units import Orientation

npr = np.array
vized = lambda x: np.vectorize(x, [float])

PI = np.pi
TOOPI = 2*PI
ISOTROPIC_AMPLITUDE_2D = 1/TOOPI          # for convenience the integral over the circle is 1
ISOTROPIC_AMPLITUDE_3D = 1/2/mt.sqrt(PI)  # for convenience the integral over the surface is 1


def polar2euclid(theta, r):
    sins = np.sin(theta)
    coss = np.cos(theta)
    return r*coss, r*sins


def euclid2polar(x, y):
    return np.linalg.norm([x-y], axis=0), np.atan2([y, x], axis=0)


def normalized_to_2pi(func):
    my_integral, err = itg.quad(func, 0, 2*PI)
    return lambda x: func(x) / my_integral


def figure8(balance, squeeze=0, squeeze_limit=0.9):

    print('balance', balance, 'squeeze', squeeze)
    # assert 0<=balance<=1, "balance must be in interval [0,1]"
    balance = np.clip(balance, 0, 1)
    counter_balance = balance - 1
    squeeze_limit = np.clip(squeeze_limit, 0, 1)
    squeeze = np.clip(squeeze, 0, squeeze_limit)
    squeeze *= PI
    half_squeeze = 0.5*squeeze
    limits = [half_squeeze, PI-half_squeeze, PI+half_squeeze, TOOPI-half_squeeze]
    speed_up_factor = PI/(PI-squeeze)

    def get_angle(x):
        if x < limits[0]:
            angle = 0
        elif x < limits[1]:
            angle = speed_up_factor*(x-half_squeeze)
        elif x < limits[2]:
            angle = PI
        elif x < limits[3]:
            angle = speed_up_factor*(x-squeeze-half_squeeze)
        else:
            angle = TOOPI
        return angle

    def func(x):
        angle = get_angle(x)
        if x < PI:
            return balance*mt.sin(angle)
        else:
            return counter_balance*mt.sin(angle)
    return func

def const_func(x):
    if isinstance(x, np.ndarray):
        return npr([0.5/PI]*len(x))
    else:
        return 0.5/PI


def create_func(func):
    return normalized_to_2pi(vized(func))


def normalized_plot(ax, x, y):
    ax.plot(x/np.max(x), y)


def cumul_int(y, x):
    return itg.cumulative_trapezoid(y, x, initial=0)


def plot_func(func, samples=100):
    interval = TOOPI
    lin = np.linspace(0, interval, samples)
    flin = func(lin)
    clin = const_func(lin)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.axis('equal')
    # normalized_plot(ax, lin, flin)
    # normalized_plot(ax, lin, cumul_int(flin, lin))
    xs, ys = polar2euclid(lin, flin)
    ax.plot(xs, ys)
    xs, ys = polar2euclid(lin, clin)
    ax.plot(xs, ys)
    xs, ys = polar2euclid(lin, flin/clin)
    ax.plot(xs, ys)
    plt.show()
    # ax = fig.add_subplot(projection='3d')


if __name__ == '__main__':
    

    optres = opt.fmin(lambda x: (4.8 - create_func(figure8(x[0], x[1]))(PI/2)/ISOTROPIC_AMPLITUDE_2D)**2, x0=npr([0.5, 0.3]))
    # TODO HPBW CRITERION
    figure8_antenna_func = create_func(figure8(optres[0], squeeze=optres[1], squeeze_limit=0.99))
    print("max forward gain", figure8_antenna_func(PI/2)/ISOTROPIC_AMPLITUDE_2D)
    print("balance:", optres[0], " squeeze:", optres[1])
    plot_func(figure8_antenna_func)
    # double_func = lambda x: figure8_antenna_func(x)*2
    # plot_func(double_func)
    # print(optres)
    # plt.plot(lin, flin/clin)
