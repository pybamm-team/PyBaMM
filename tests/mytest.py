import pybamm
import numpy as np

y = pybamm.StateVector(slice(0, 4))
a = pybamm.StateVector(slice(0, 2))
b = pybamm.StateVector(slice(2, 4))
print(np.size(y))
equation = 2 * a + 6 * b
equation.render()

jac = equation.diff(y).render()
