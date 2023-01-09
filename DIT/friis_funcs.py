import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os

join = os.path.join
npr = np.array
c = 299792458
vized = lambda x: np.vectorize(x, otypes=[float])


def power_to_db(x):
    return 10*np.log10(x)


def db_to_power(x):
    return 10**(0.1*x)


def get_dbm_func(base, is_power=False):
    inv_base = 1/base
    factor = 10+(0 if is_power else 10)
    return vized(lambda x: factor*np.log10(x*inv_base))


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


def eps_to_gamma(eps1, eps2):
    e1 = np.sqrt(eps1)
    e2 = np.sqrt(eps2)
    return (e2 - e1)/(e1 + e2)


def VSWR_to_gamma(VSWR):
    pass


def gamma_to_VSWR(gamma):
    abs_gamma = math.abs(gamma)
    return (1+abs_gamma) / (1-abs_gamma)


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


def plot_Pr_over_distance(pair, parameters, measured_data, rssi_max_voltage=3300):
    Pt = 20
    Gt = parameters[pair[0]]
    Gr = parameters[pair[1]]
    ds = [0.3, 3]
    measurement_interval = 0.1
    # [0.3, 0.5] => 3 points & rounding up:
    number_of_sections = int((ds[1] - ds[0]+measurement_interval)/measurement_interval+0.5)
    d = np.linspace(*ds, number_of_sections)
    # print(len(d), len(measured_data))
    f = 915e6
    ax = plt.subplot()
    Prlog = friis_Pr_log(Pt, Gt, Gr, f, d)
    ax.plot(d, Prlog, color=(0, 0.9, 0), label='Free space')
    
    # destructive
    Pr = db_to_power(Prlog)
    relative_permittivity = 10
    reflection_coefficient = eps_to_gamma(1, relative_permittivity)
    wall_distance = 0.8

    Pr = Pr - friis_Pr(db_to_power(Pt)*reflection_coefficient*1e-3, db_to_power(Gt), db_to_power(Gr), f, d+wall_distance)
    Pr = power_to_db(Pr)
    ax.plot(d, Pr, color=(0.0, 0.0, 0.0), label='Destructive')
    
    # constructive
    Pr = db_to_power(Prlog)

    Pr = Pr + friis_Pr(db_to_power(Pt)*reflection_coefficient*1e-3, db_to_power(Gt), db_to_power(Gr), f, d+wall_distance)
    Pr = power_to_db(Pr)
    ax.plot(d, Pr, color=(0, 0, 0.9), label='Constructive')
    ax.plot(d[:len(measured_data)], measured_data, color=(0.9, 0, 0), label='Measured data')

    # titles & axes
    title = 'Power transmitted from '+pair[0]+' to '+pair[1]
    ax.set_title(title)
    print(title, ": first value:", measured_data[0], " last value:", measured_data[-1])
    ax.set_ylabel('Pr [dBm] wrt RSSI {:0.1f} V'.format(rssi_max_voltage*1e-3))
    ax.set_xlabel('d [m]')
    ax.legend()
    plt.savefig((pair[0]+'_'+pair[1]+'_d'+'.pdf').replace(' ', '_'))
    plt.savefig((pair[0]+'_'+pair[1]+'_d'+'.png').replace(' ', '_'))
    plt.show()


def plot_polarization_Pr(pair, parameters, measured_data):
    Pt = 20
    Gt = parameters[pair[0]]
    Gr = parameters[pair[1]]
    d = measured_data['d']
    f = 915e6
    angles = np.linspace(0, np.pi/2, 5)
    alignment_factor = np.cos(angles)**2
    Pr = friis_Pr(Pt, Gt, Gr, f, d)
    theoretical_y = Pr*alignment_factor
    theoretical_y /= np.max(theoretical_y)
    mV = npr(measured_data['mV'], dtype=float)
    mV /= np.max(mV)
    f = 915e6
    ax = plt.subplot()
    ax.plot(angles, mV, color=(0.9, 0, 0), label='Measured data')
    ax.plot(angles, theoretical_y, color=(0, 0, 0.9), label='Calculated data')
    title = 'Polarization effect: '+pair[0]+' to '+pair[1]
    ax.set_title(title)
    ax.set_ylabel('Relative Pr [1]')
    ax.set_xlabel('Angle about '+measured_data['axis']+' axis [rad]')
    ax.legend()
    plt.savefig((pair[0]+'_'+pair[1]+'_'+measured_data['axis']+'.pdf').replace(' ', '_'))
    plt.savefig((pair[0]+'_'+pair[1]+'_'+measured_data['axis']+'.png').replace(' ', '_'))
    plt.show()


if __name__ == "__main__":
    antennas = ['868MHz antenna', 'Patch antenna', 'Dipole antenna']
    
    antenna_gains_dBm = power_to_db(npr([2, 4.8, 1.5]))
    antenna_gains = power_to_db(npr([2, 4.8, 1.5]))
    antenna_pairs = [[antennas[1], antennas[2]], [antennas[1], antennas[0]], [antennas[2], antennas[0]]]
    antenna_gains_dBm = dict(zip(antennas, antenna_gains_dBm))
    antenna_gains = dict(zip(antennas, antenna_gains))
    base_path = 'C:/Users/jhart/PythonProjects/zs22/DIT/'
    pair_files = ['patch_to_dipole.txt', 'patch_to_868.txt', 'dipole_to_868.txt']
    board_voltages = [5500, 5500, 3300]
    # board_voltage = 3300  # [mV] maximum rated continuous voltage
    # dbm_func = get_dbm_func(board_voltage, is_power=False)
    # print(dbm_func(3300), dbm_func(330))
    plot_dists = True
    # plot measured & model data
    i=0
    for pair_file, pair in zip(pair_files, antenna_pairs):
        if not plot_dists:
            break
        measured_data = np.loadtxt(join(base_path, pair_file))
        board_voltage = board_voltages[i]
        dbm_func = get_dbm_func(board_voltage, is_power=False)
        measured_data = dbm_func(measured_data)
        # print(measured_data)
        plot_Pr_over_distance(pair, antenna_gains_dBm, measured_data, rssi_max_voltage=board_voltage)
        i+=1

    polarization_pairs = [[antennas[0], antennas[0]], [antennas[0], antennas[0]], [antennas[1], antennas[0]]]
    rotation_measurement1 = {'d': 0.3, 'mV': [457, 372, 254, 118,  8], 'axis': 'y'}
    rotation_measurement2 = {'d': 0.4, 'mV': [308, 255, 116, 28,   6], 'axis': 'x'}
    rotation_measurement3 = {'d': 0.4, 'mV': [555, 500, 335, 180, 85], 'axis': 'z'}
    rotation_measurements = [rotation_measurement1, rotation_measurement2, rotation_measurement3]
    
    plot_polarization = True
    # plot polarization power dependence
    for measured_data, pair in zip(rotation_measurements, polarization_pairs):
        if not plot_polarization:
            break
        plot_polarization_Pr(pair, antenna_gains, measured_data)
    
    # plot all measured data 
    ax = plt.subplot()
    title = 'Measured data comparison'
    ax.set_title(title)
    ax.set_ylabel('Pr [dBm] wrt RSSI {:0.1f} V'.format(board_voltage*1e-3))
    measured_datas = []
    ax.set_xlabel('d [m]')
    i=0
    for pair_file, pair in zip(pair_files, antenna_pairs):
        measured_data = np.loadtxt(join(base_path, pair_file))
        dbm_func = get_dbm_func(board_voltages[i])
        measured_data = dbm_func(measured_data)
        ds = [0.3, 3]
        measurement_interval = 0.1
        # e.g. [0.3, 0.5] => 3 points & rounding up:
        number_of_sections = int((ds[1] - ds[0]+measurement_interval)/measurement_interval+0.5)
        d = np.linspace(*ds, number_of_sections)
        ax.plot(d, measured_data, label=pair[0]+' to '+pair[1])
        measured_datas.append(measured_data)
        i+=1
    # ax.plot(d, np.mean(measured_datas, axis=0), linestyle=':', label='Mean characteristic')
    ax.legend()
    plt.savefig(title.replace(' ', '_')+'.pdf')
    plt.savefig(title.replace(' ', '_')+'.png')
    plt.show()


