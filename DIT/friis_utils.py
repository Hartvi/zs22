from scipy.spatial.transform import Rotation
import numpy as np
import math as mt

from scipy import integrate as itg
from scipy import optimize as opt
from typing import *
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

c = 299792458.0


class Antenna:
    def __init__(self, frequency, power_funcs, antenna_rotation, base_gain) -> None:
        self.frequency = frequency
        self.wave_length = c/frequency
        # should be a list of the power funcs for each axis xyz
        self.power_funcs = power_funcs
        self.eval_power_funcs = lambda x: npr([power_funcs[0](x[0]), power_funcs[1](x[1]), power_funcs[2](x[2])])
        self.antenna_rotation = antenna_rotation
        # vector points up to correspond to default antenna rotation
        self.polarization_vector = antenna_rotation @ npr([[0, 0, 1]]).T
        self.direction_vector = antenna_rotation @ npr([[0, 1, 0]]).T
        self.base_gain = base_gain
    

class PatchAntenna(Antenna):
    def __init__(self, antenna_rotation, frequency=915e6, base_gain=4.8, hpbw_z=120.0, hpbw_x=62.0, balance_z=0.95, balance_x=0.95) -> None:

        max_power = 4.8
        # NOTE: y is the direction it is radiating in, thus it doesn't affect others, i.e. unity gain
        # base_gains = npr([np.sqrt(max_power), 1, np.sqrt(max_power)])
        hpbw = hpbw_z/180.0*PI
        antenna_func_z = find_figO(hpbw, balance=balance_z)

        hpbw = hpbw_x/180.0*PI
        antenna_func_x = find_figO(hpbw, balance=balance_x)
        antenna_func_x = cosinify(antenna_func_x)

        antenna_func_y = vized(lambda _: 1.0)
        power_funcs = npr([antenna_func_x, antenna_func_y, antenna_func_z])
        
        super().__init__(frequency, power_funcs, antenna_rotation, base_gain)

    def eval(self, xyz, logify=False, include_amplitude=True):
        x,y,z = self.antenna_rotation @ xyz
        
        angles = npr([np.arctan2(z, y), np.arctan2(x, z), np.arctan2(y, x)])
        # print(angles)
        dists = np.linalg.norm(xyz, axis=0)
        
        power_gains = self.eval_power_funcs(angles)  # (3, n)
        power_gain = np.prod(power_gains, axis=0)*self.base_gain  # (n, )
        # print(power_gain.shape)
        phases = (dists % self.wave_length)*TOOPI
        amplitude_vector = (np.sin(phases) * self.polarization_vector)
        amplitudes = (self.wave_length/(4*np.pi*dists))**2
        if logify:
            return 10*np.log10(power_gain) + 10*np.log10((amplitudes if include_amplitude else 1)), amplitude_vector
        else:
            return power_gain*(amplitudes if include_amplitude else np.ones(amplitudes.shape)), amplitude_vector


class RodAntenna(Antenna):  # TODO: finish the power funcs so that it takes ony the angle between z and the xy plane and has a radially symmetrical radiation characteristic
    def __init__(self, antenna_rotation, frequency=915e6, base_gain=1.5, hpbw=80.0) -> None:

        # NOTE: y is the direction it is radiating in, thus it doesn't affect others, i.e. unity gain
        # base_gains = npr([np.sqrt(max_power), 1, np.sqrt(max_power)])
        hpbw = hpbw/180.0*PI
        antenna_func = find_figO(hpbw, balance=1)
        # antenna_func = cosinify(antenna_func)

        antenna_func_z = vized(lambda _: 1.0)
        antenna_func_y = vized(lambda _: 1.0)
        power_funcs = npr([antenna_func, antenna_func_y, antenna_func_z])
        
        super().__init__(frequency, power_funcs, antenna_rotation, base_gain)

    def eval(self, xyz, logify=False, include_amplitude=True):
        rot_xyz = self.antenna_rotation @ xyz
        
        angle = np.arctan2(np.linalg.norm(rot_xyz[:2], axis=0), rot_xyz[2])
        dists = np.linalg.norm(xyz, axis=0)
        
        power_gains = self.power_funcs[0](angle)
        power_gain = power_gains*self.base_gain  # (n, )
        phases = (dists % self.wave_length)*TOOPI
        amplitude_vector = (np.sin(phases) * self.polarization_vector)
        amplitudes = (self.wave_length/(4*np.pi*dists))**2
        if logify:
            return 10*np.log10(power_gain) + 10*np.log10((amplitudes if include_amplitude else 1)), amplitude_vector
        else:
            return power_gain*(amplitudes if include_amplitude else np.ones(amplitudes.shape)), amplitude_vector



class WaveUnit(Orientation):
    def __init__(self, pva, R, t: Iterable = ..., ):
        super().__init__(R, t)
        assert len(pva) == 3, "Phase Value Angle vector (pva) len != 3! "+str(len(pva))
        self.pva = pva
    
    def step():
        ...


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


