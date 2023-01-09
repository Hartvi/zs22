
from scipy.spatial.transform import Rotation
from scipy import integrate as itg
from scipy import optimize as opt
from typing import *
import numpy as np
import math as mt
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import itertools as it

from friis_utils import *



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


def plot_antenna_characteristic_3d(antenna: Antenna):
    # antenna = Antenna(915e6, funcs, np.eye(3), base_gains=npr([np.sqrt(4.8), 1, np.sqrt(4.8)]))
    number_of_sections = 40
    cube_side = 1
    x_range = (-1*cube_side, 1*cube_side)
    y_range = (0, 2*cube_side)
    z_range = (-1*cube_side, 1*cube_side)
    x_range = (-1*cube_side, 1*cube_side)
    y_range = (-1*cube_side, 1*cube_side)
    z_range = (-1*cube_side, 1*cube_side)
    eval_space_range_x = np.linspace(*x_range, number_of_sections)
    eval_space_range_y = np.linspace(*y_range, number_of_sections//2)
    eval_space_range_z = np.linspace(*z_range, number_of_sections)
    eval_positions = npr(list(it.product(eval_space_range_x, eval_space_range_y, eval_space_range_z))).T
    cube_range = np.max(eval_positions)
    cube_range = (-cube_range, cube_range)
    # print('eval positions', eval_positions.shape[1])
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
        ax.set_title('Patch antenna characteristic')
        
        # adjust the main plot to make room for the sliders
        fig.subplots_adjust(left=0.25, bottom=0.25)

        # Make a horizontal slider to control the frequency.
        axfreq = fig.add_axes([0.5, 0.05, 0.3, 0.03])
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


if __name__ == '__main__':    

    # max_power = 4.8
    # hpbw = 120.0/180.0*PI
    # antenna_func_z = find_figO(hpbw, balance=0.95)

    # hpbw = 62.0/180.0*PI
    # antenna_func_x = find_figO(hpbw, balance=0.95)
    # antenna_func_x = cosinify(antenna_func_x)

    # antenna_func_y = vized(lambda _: 1.0)
    # funcs = npr([antenna_func_x, antenna_func_y, antenna_func_z])
    # antenna = Antenna(915e6, funcs, np.eye(3), base_gains=npr([np.sqrt(max_power), 1, np.sqrt(max_power)]))
    
    # hpbw = 80.0/180.0*PI
    # antenna_func_z = find_figO(hpbw, balance=1)
    # plot_func(antenna_func_z)
    # exit(1)
    norot = np.eye(3)
    zrot = npr([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    antenna = PatchAntenna(norot)
    # antenna = RodAntenna(norot)
    plot_antenna_characteristic_3d(antenna=antenna)
    
    # lnsp = np.linspace(0, TOOPI, 100)
    # sine = np.sin(lnsp)
    # print("RMS: ", np.sqrt(np.mean(sine**2)))

