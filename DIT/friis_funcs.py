import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from class_w_units import ClassWUnits as CLU

npr = np.array
c = 299792458


def watt_to_dbm(watts):
    # https://www.everythingrf.com/rf-calculators/watt-to-dbm
    return 10*np.log10(watts) + 30


def f_to_lambda(f):
    return c/f


def friis_Pr(Pt, Gt, Gr, f, d):
    lbd = f_to_lambda(f)
    return Pt * Gt * Gr * (lbd / (4 * math.pi * d))**2


def friis_Pr_log(Pt, Gt, Gr, f, d):
    lbd = f_to_lambda(f)
    return Pt + Gt + Gr + 20 * np.log10((lbd / (4 * math.pi * d))**2)


def VSWR_to_gamma(VSWR):
    pass


def gamma_to_VSWR(gamma):
    abs_gamma = math.abs(gamma)
    return (1+abs_gamma) / (1-abs_gamma)


class Values:
    base_frequency = CLU(915e6, 'Hz')
    rad = CLU(1.0, 'rad')
    watt = CLU(1.0, 'W')
    dB = CLU(1.0, 'dB')


class Antenna:
    """
    rotation: x: right, y: down, z: forward
    """
    # https://en.wikipedia.org/wiki/Directivity#Definition
    def __init__(self, orientation, hpbw, polarization=npr((0, 1, 0)), frequency=9.15e6, peak_gain=None, return_loss=None, vswr=None, eirp=None):
        self.orientation = orientation
        self.hpbw = hpbw
        self.polarization = polarization
        self.frequency = frequency
        self.peak_gain = peak_gain
        self.return_loss = return_loss
        self.vswr = vswr
        self.eirp = eirp


class Beam:
    def __init__(self, R, t):
        self.internal_phase = npr((0, 1, 0))
        self.R = R
        self.t = t
        self.dir = R @ npr([[0], [0], [1]])  # Z is forwards, X is right, Y is up


if __name__ == "__main__":
    r = npr([1,0,0])
    r = Rotation.from_rotvec(r)  # type: Rotation
    # Rotation.as_rotvec(r)
    print(r.as_mrp())
    pass

