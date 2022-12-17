from scipy.spatial.transform import Rotation
from typing import *
import numpy as np

npr = np.array


class ClassWUnits:
    max_number_of_units = 32
    units: List[str] = list()
    unit2id = dict()

    def __init__(self, val: float, units: Union[str, np.ndarray] = None):
        self.val = val
        if type(units) == str:
            if units not in ClassWUnits.unit2id:
                ClassWUnits.unit2id[units] = len(ClassWUnits.units)
                ClassWUnits.units.append(units)
                assert len(ClassWUnits.units) <= ClassWUnits.max_number_of_units, "Max number of units exceeded! "+str(len(ClassWUnits.units))+"<"+str(ClassWUnits.max_number_of_units)
            self.unit_list = np.zeros(ClassWUnits.max_number_of_units, dtype=int)
            self.unit_list[ClassWUnits.unit2id[units]] = 1
        else:  # units: np.ndarray
            self.unit_list = units

    def __mul__(self, other):
        if type(other) == ClassWUnits:
            return ClassWUnits(other.val * self.val, self.add_units(other))
        else:  # float/int/etc
            return ClassWUnits(other * self.val, self.unit_list)

    def __rmul__(self, other):
        if type(other) == ClassWUnits:
            return ClassWUnits(other.val * self.val, self.add_units(other))
        else:  # float/int/etc
            if other == 0 or other == 0.0:
                return 0.0
            return ClassWUnits(other * self.val, self.unit_list)

    def __add__(self, other):
        # assert type(other) == ClassWUnits, "ClassWUnits can only be added to other instances of ClassWUnits"
        assert (other.unit_list == self.unit_list).all(), "Cannot add units: '"+str(self)+"' vs '"+str(other)+"'"
        return ClassWUnits(other.val + self.val, self.unit_list)

    def __radd__(self, other):
        # assert type(other) == ClassWUnits, "ClassWUnits can only be added to other instances of ClassWUnits"
        assert other.unit_list == self.unit_list, "Cannot add units: '"+str(self)+"' vs '"+str(other)+"'"
        return ClassWUnits(other.val + self.val, self.unit_list)

    def __repr__(self):
        repr_len = np.max(np.where(0.0 != self.unit_list)[0])+1
        # print(self.unit_list[:repr_len])
        repr_str = str(self.val) + " "
        for i in range(repr_len):
            unit_order = self.unit_list[i]
            # print(unit_order)
            if unit_order > 0:
                repr_str += ClassWUnits.units[i]
                if unit_order > 1:
                    repr_str += "^" + str(self.unit_list[i]) + " "
        return repr_str

    def add_units(self, other):
        ret = other.unit_list + self.unit_list
        return ret


class Orientation:
    def __init__(self, R, t: Iterable=(), deg=False):
        if R.shape == (4,4):
            self.R = R[:3, :3]
            self.t = R[:3, -1]
            self.transform = R
        elif isinstance(R, Rotation):
            self.R = R.as_matrix()
        elif isinstance(R, np.ndarray):
            if R.size == 3:
                self.R = Rotation.from_euler('xyz', R, degrees=deg)
            else:
                assert R.size == 9, "R must be a matrix or euler angles"
                self.R = R
        if not hasattr(self, 't'):
            assert len(t) == 3, "translation must be 3d: t="+str(t)
            self.t = npr(t)
            self.transform = Orientation.stackify_Rt(self.R, self.t)
        # print(self.transform)

    @staticmethod
    def stackify_Rt(R, t):
        return np.vstack([np.hstack((R,t.reshape(-1, 1))), npr([[0,0,0,1]])])
    
    def __matmul__(self, other):
        # print(self.transform)
        # print(other)
        if isinstance(other, Orientation):
            T = self.transform @ other.transform
        else: 
            T = self.transform @ other
        return Orientation(T)

    def get_rotation(self):
        return Rotation.from_matrix(self.R)
    
    def __repr__(self) -> str:
        return 'R: ' + str(self.R[0, :]) + ' t: ' + str(self.t[0]) + '\n   ' + str(self.R[1, :]) + '    ' + str(self.t[1]) + '\n   ' + str(self.R[2, :]) + '    ' + str(self.t[2]) 


if __name__ == "__main__":
    a = ClassWUnits(2.0, "m")
    b = ClassWUnits(3.0, "s")
    print(a*b*a*b)
    print((a*b*a*b))
    print(a)
    print("5*a: ", 5*a)
    print(b)
    try:
        print(a + b)
    except:
        print("see? adding", a, 'and', b, 'doesnt work')

    # ORIENTAITION
    t1 = Orientation(np.eye(3), npr([1, 0, 0]))
    t2 = Orientation(np.eye(3), npr([1, 1, 1]))
    print(t1@t2)



