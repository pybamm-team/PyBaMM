import pybamm
import numpy as np

y = pybamm.StateVector(slice(0, 4))
u = pybamm.StateVector(slice(0, 2))
v = pybamm.StateVector(slice(2, 4))

y0 = np.array([1, 2, 3, 4])

f = -1 / u
f.jac(y)
import ipdb

ipdb.set_trace()
