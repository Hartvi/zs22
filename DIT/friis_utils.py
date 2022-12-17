
from scipy.spatial.transform import Rotation
from typing import *
import numpy as np

from class_w_units import Orientation

npr = np.array
c = 299792458.0


class WaveUnit(Orientation):
    def __init__(self, pva, R, t: Iterable = ..., ):
        super().__init__(R, t)
        assert len(pva) == 3, "Phase Value Angle vector (pva) len != 3! "+str(len(pva))
        self.pva = pva
    
    def step():
        ...
    

