
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

PI = np.pi
TOOPI = 2*PI

def polar2euclid(theta, r):
    sins = np.sin(theta)
    coss = np.cos(theta)
    return r*coss, r*sins


def euclid2polar(x, y):
    return np.linalg.norm([x-y], axis=0), np.atan2([y, x], axis=0)


def normalized_to_2pi(func):
    my_integral, err = itg.quad(func, 0, 2*PI)
    return lambda x: func(x) / my_integral


def figure8(balance, squeeze_angle=0):

    # assert 0<=balance<=1, "balance must be in interval [0,1]"
    half_squeeze = 0.5*squeeze_angle
    limits = [half_squeeze, PI-half_squeeze, PI+half_squeeze, TOOPI-half_squeeze]
    speed_up_factor = PI/(PI-squeeze_angle)
    def get_angle(x):
        if limits[0] < x < limits[1]:
            angle = speed_up_factor*(x-half_squeeze)
        elif limits[2] < x < limits[3]:
            angle = speed_up_factor*(x-half_squeeze)
        else:
            angle = 0
        return angle

    def func(x):
        angle = get_angle(x)
        # angle=x
        # print(angle)
        return (1 - (x > PI)*balance)*abs(mt.sin(angle))
    # return lambda x: (1 - (x < PI)*balance)*np.sin(x)
    return func

def const_func(x):
    if isinstance(x, np.ndarray):
        return npr([0.5/PI]*len(x))
    else:
        return 0.5/PI


if __name__ == '__main__':
    vized = lambda x: np.vectorize(x, [float])
    
    some_func = vized(figure8(0.8, squeeze_angle=np.deg2rad(30)))
    figure8_antenna_func = normalized_to_2pi(some_func)
    # figure8_antenna_func = (some_func)
    const_antenna_func = const_func
    samples = 100
    interval = 2*PI
    lin = np.linspace(0, interval, samples)
    flin = figure8_antenna_func(lin)
    clin = const_func(lin)

    # print(normalized_to_2pi(vized(figure8(0.9)))(PI/2))
    optres = opt.fmin(lambda x: (2.19 - normalized_to_2pi(vized(figure8(*x)))(PI/2)/const_antenna_func(PI/2))**2, x0=npr([1, 0]))
    print(optres[0])
    # plt.plot(lin, flin/clin)
    plt.plot(lin, flin)
    plt.plot(lin, itg.cumulative_trapezoid(flin, lin, initial=0))
    xs, ys = polar2euclid(lin, flin)
    plt.plot(xs, ys)
    plt.show()
