
from scipy.spatial.transform import Rotation
from scipy import integrate as itg
from scipy import optimize as opt
from typing import *
import numpy as np
import math as mt
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import itertools as it

from class_w_units import Orientation
from friis_utils import *

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


def figure8(balance, squeeze=0, squeeze_limit=0.9, is_sine=True):

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
        # x = x
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
        if is_sine:
            if x < PI:
                return balance*mt.sin(angle)
            else:
                return counter_balance*mt.sin(angle)
        else:
            if x < HALFPI or x > HALFPI+PI:
                return balance*mt.cos(angle)
            else:
                return counter_balance*mt.cos(angle)
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


def find_func(hpbw, max_amplitude, is_sine=True):

    hhpbw = hpbw/2
    print("PI-hpbw/2:", (HALFPI - hhpbw)/PI*180)
    half_max = max_amplitude/SQRT2
    def err_func(arg):
        err = 0.0
        new_func = lambda x: (create_func(figure8(arg[0], arg[1], is_sine=is_sine))(x))
        # err = new_func(HALFPI)
        # err = (err - max_amplitude)**2
        err += (new_func(HALFPI - hhpbw) - half_max)**2
        # err += 0.001*(new_func(3*HALFPI))**2
        return err
    return err_func


def figureO(squeeze, squeeze_limit=0.9, balance=0.95, is_sine=True):
    return figure8(balance, squeeze, squeeze_limit, is_sine=is_sine)


def find_figO(hpbw, balance=0.95, is_sine=True):
    hhpbw = hpbw/2
    sqrt_balance = 1 - (1-balance)**2
    # print("PI-hpbw/2:", (HALFPI - hhpbw)/PI*180)
    def err_func(arg):
        new_func = lambda x: figureO(arg, balance=balance)(x)
        err = (new_func(HALFPI) - new_func(HALFPI - hhpbw)*2)**2  # HALF AMPLITUDE => SQRT TO GET POWER RATIO
        return err
    fv = vized(err_func)
    resolution = 1001
    l = np.linspace(0, 1, resolution)
    fvl = fv(l)
    opt_i = np.argmin(fvl)
    opt_res = opt_i/resolution
    # print('min err:', fvl[opt_i])
    # print("found result:", opt_res)
    
    return vized(lambda x: figureO(opt_res, balance=balance, is_sine=is_sine)(x)/balance)


def maxify_func(func, amplitude):
    k = amplitude/func(HALFPI)
    return lambda x: k*func(x)


def cosinify(func):
    return lambda x: func(x+HALFPI)


if __name__ == '__main__':    

    max_amplitude = 4.8
    hpbw = 120.0/180.0*PI
    antenna_func_z = find_figO(hpbw, balance=0.95)
    # make func have a preset forward max gain
    # antenna_func_z = maxify_func(antenna_func_z, max_amplitude)
    # print("max val:", antenna_func_z(HALFPI), ' half val:', antenna_func_z(HALFPI-hpbw/2))
    # plot_func(antenna_func_z)

    hpbw = 60.0/180.0*PI
    antenna_func_x = find_figO(hpbw, balance=0.95)
    # antenna_func_x = maxify_func(antenna_func_x, max_amplitude)
    # print("max val:", antenna_func_x(HALFPI), ' half val:', antenna_func_x(HALFPI-hpbw/2))
    antenna_func_x = cosinify(antenna_func_x)
    # plot_func(antenna_func_x)
    # exit(1)

    antenna_func_y = vized(lambda _: 1.0)
    funcs = npr([antenna_func_x, antenna_func_y, antenna_func_z])
    antenna = Antenna(915e6, funcs, np.eye(3), base_gains=npr([np.sqrt(4.8), 1, np.sqrt(4.8)]))
    number_of_sections = 40
    cube_side = 1
    x_range = (-2*cube_side, 2*cube_side)
    y_range = (0, 4*cube_side)
    z_range = (-2*cube_side, 2*cube_side)
    eval_space_range_x = np.linspace(*x_range, number_of_sections)
    eval_space_range_y = np.linspace(*y_range, number_of_sections//2)
    eval_space_range_z = np.linspace(*z_range, number_of_sections)
    eval_positions = npr(list(it.product(eval_space_range_x, eval_space_range_y, eval_space_range_z))).T
    cube_range = np.max(eval_positions)
    cube_range = (-cube_range, cube_range)
    print('eval positions', eval_positions.shape[1])
    # eval_positions = npr([eval_space_range]*3)

    # print(eval_positions.shape)
    # Y is forward, X is right, Z is up
    # antenna_forward = antenna.eval(npr([0.1,0.1,0.05]), include_amplitude=False)
    # print("antenna_forward:", antenna_forward[0])
    antenna_forward, polarized_vectors = antenna.eval(eval_positions, logify=False, include_amplitude=True)
    coeff = 30
    shell_thickness = 0.05**2*0.1
    shell_beginning = 0.05**2*0.1
    shell_indices = (antenna_forward < ((shell_beginning+shell_thickness)))&(antenna_forward > (shell_beginning))
    shell_positions = eval_positions[:, shell_indices]
    xs,ys,zs = shell_positions
    print('shell contains', shell_positions.shape[1], 'points')
    # print('log intensities', antenna_forward[shell_indices])
    plot_3d = True
    if plot_3d:
        fig = plt.figure()
        # fig.canvas.manager.full_screen_toggle()
        ax = fig.add_subplot(projection='3d')
        ax.plot([0, 0], [0, 0], [-0.15, 0.15], color=(0, 0.7, 0.7), lw=2/cube_side)
        myline, = ax.plot(xs, ys, zs, marker='.', markersize=1, color=(1,0.0,0.0), lw=0)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # ax.axis('equal')
        ax.set_xlim3d(x_range)
        ax.set_ylim3d(y_range)
        ax.set_zlim3d(z_range)
        
        # adjust the main plot to make room for the sliders
        fig.subplots_adjust(left=0.25, bottom=0.25)

        # Make a horizontal slider to control the frequency.
        axfreq = fig.add_axes([0.25, 0.1, 0.65, 0.03])
        valmax = -3
        valmin = -6
        freq_slider = Slider(
            ax=axfreq,
            label='Proportional value',
            valmin=valmin,
            valmax=valmax,
            valinit=-4,
        )

        def update(val):
            shell_thickness = 10**((0.5+val))
            shell_beginning = 10**((1+val))
            shell_indices = (antenna_forward < ((shell_beginning+shell_thickness)))&(antenna_forward > (shell_beginning))
            shell_positions = eval_positions[:, shell_indices]
            xs,ys,zs = shell_positions
            myline.set_xdata(xs)
            myline.set_ydata(ys)
            myline.set_3d_properties(zs, zdir='z')
            cols = npr([3*np.abs((val - valmax)/(valmax - valmin))]*3) - npr([0, 1, 2])
            cols = np.clip(cols,0,1)
            myline.set(color=cols)
            fig.canvas.draw_idle()


        # register the update function with each slider
        freq_slider.on_changed(update)

        plt.show()
    
    # lnsp = np.linspace(0, TOOPI, 100)
    # sine = np.sin(lnsp)
    # print("RMS: ", np.sqrt(np.mean(sine**2)))

