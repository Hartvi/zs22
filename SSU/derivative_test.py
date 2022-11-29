import sympy as sp
from sympy.vector import BaseVector

xyz = sp.symbols('x y z')
x,y,z = xyz
xyz_arr = sp.Array([_ for _ in xyz])
xxx_arr = sp.Array([x,x,x])
arr = sp.exp(xxx_arr)
print(arr.diff(x))