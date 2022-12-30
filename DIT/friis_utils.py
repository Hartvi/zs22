
from scipy.spatial.transform import Rotation
from typing import *
import numpy as np
import math as mt

from class_w_units import Orientation

npr = np.array
c = 299792458.0
TOOPI = np.pi*2


class Antenna:
    def __init__(self, frequency, power_funcs, antenna_rotation, base_gains) -> None:
        self.frequency = frequency
        self.wave_length = c/frequency
        # should be a list of the power funcs for each axis xyz
        self.power_funcs = lambda x: npr([power_funcs[0](x[0]), power_funcs[1](x[1]), power_funcs[2](x[2])])
        self.antenna_rotation = antenna_rotation
        # vector points up to correspond to default antenna rotation
        self.polarization_vector = antenna_rotation @ npr([[0, 0, 1]]).T
        self.base_gains = base_gains
    
    def eval(self, xyz, logify=False, include_amplitude=True):
        x,y,z = xyz
        # ro = np.sqrt(x**2+y**2)
        
        angles = npr([np.arctan2(z, y), np.arctan2(x, z), np.arctan2(y, x)])
        # xyz_sqr = xyz * xyz
        # sqr_sum = np.sum(xyz_sqr, axis=0)
        # dists = np.sqrt(sqr_sum)
        # factors = xyz_sqr / sqr_sum
        dists = np.linalg.norm(xyz, axis=0)
        
        # print("angles:", angles)
        # print('factors:',factors)
        power_gains = self.power_funcs(angles)
        # print(self.base_gains)
        # print((power_gains.T))
        # print((power_gains.T*self.base_gains).T)
        # OR DOES THIS HAVE TO BE WEIGHTED BY THE SQUARE??
        power_gain = np.prod((power_gains.T*self.base_gains).T, axis=0)
        # print('power gain:', power_gain)
        # dist = np.sqrt(ro**2+z)
        phases = (dists % self.wave_length)*TOOPI
        # print('polarization', self.polarization_vector)
        # print('phases', np.sin(phases))
        value_vector = (np.sin(phases) * self.polarization_vector)
        # print('polarization vectors', value_vector)
        amplitudes = (self.wave_length/(4*np.pi*dists))**2
        if logify:
            return 10*np.log10(power_gain) + 10*np.log10((amplitudes if include_amplitude else 1)), value_vector
        else:
            return power_gain*(amplitudes if include_amplitude else 1), value_vector


class WaveUnit(Orientation):
    def __init__(self, pva, R, t: Iterable = ..., ):
        super().__init__(R, t)
        assert len(pva) == 3, "Phase Value Angle vector (pva) len != 3! "+str(len(pva))
        self.pva = pva
    
    def step():
        ...
    

