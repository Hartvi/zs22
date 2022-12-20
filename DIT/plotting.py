
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
HALFPI = PI/2
SQRT2 = mt.sqrt(2)
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

    # print('balance', balance, 'squeeze', squeeze)
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
    ax.plot(lin, flin)
    ax.plot(lin, cumul_int(flin, lin))
    xs, ys = polar2euclid(lin, flin)
    ax.plot(xs, ys)
    xs, ys = polar2euclid(lin, clin)
    ax.plot(xs, ys)
    # xs, ys = polar2euclid(lin, flin/clin)
    # ax.plot(xs, ys)
    plt.show()
    # ax = fig.add_subplot(projection='3d')


def find_func(hpbw, max_amplitude):

    hhpbw = hpbw/2
    print("PI-hpbw/2:", (HALFPI - hhpbw)/PI*180)
    half_max = max_amplitude/SQRT2
    def err_func(arg):
        err = 0.0
        new_func = lambda x: (create_func(figure8(arg[0], arg[1]))(x))
        # err = new_func(HALFPI)
        # err = (err - max_amplitude)**2
        err += (new_func(HALFPI - hhpbw) - half_max)**2
        # err += 0.001*(new_func(3*HALFPI))**2
        return err
    return err_func


def figureO(squeeze, squeeze_limit=0.9, balance=0.95):
    return figure8(balance, squeeze, squeeze_limit)


def find_figO(hpbw, balance=0.95):
    hhpbw = hpbw/2
    sqrt_balance = 1 - (1-balance)**2
    # print("PI-hpbw/2:", (HALFPI - hhpbw)/PI*180)
    def err_func(arg):
        new_func = lambda x: figureO(arg, balance=balance)(x)
        err = (new_func(HALFPI) - new_func(HALFPI - hhpbw)*2)**2
        return err
    fv = vized(err_func)
    resolution = 101
    l = np.linspace(0, 1, resolution)
    fvl = fv(l)
    opt_i = np.argmin(fvl)
    opt_res = opt_i/resolution
    print('min err:', fvl[opt_i], "max/2 - ampl(hpbw")
    print("found ratio:", opt_res, 'max/ampl(hpbw)')
    
    return vized(lambda x: figureO(opt_res, balance=balance)(x))


def maxify_func(func, amplitude):
    k = amplitude/func(HALFPI)
    return lambda x: k*func(x)

if __name__ == '__main__':    

    max_amplitude = 4.8
    hpbw = 120.0/180.0*PI
    antenna_func = find_figO(hpbw, balance=0.95)
    antenna_func = maxify_func(antenna_func, max_amplitude)
    print("max val:", antenna_func(HALFPI), ' half val:', antenna_func(HALFPI-hpbw/2))
    plot_func(antenna_func)

    hpbw = 60.0/180.0*PI
    antenna_func = find_figO(hpbw, balance=0.95)
    antenna_func = maxify_func(antenna_func, max_amplitude)
    print("max val:", antenna_func(HALFPI), ' half val:', antenna_func(HALFPI-hpbw/2))
    plot_func(antenna_func)

