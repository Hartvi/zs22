import numpy as np
from typing import List, Union, Tuple


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


if __name__ == "__main__":
    a = ClassWUnits(2.0, "m")
    b = ClassWUnits(3.0, "s")
    print(a*b*a*b)
    print((a*b*a*b))
    print(a)
    print("5*a: ", 5*a)
    print(b)
    print(a + b)



