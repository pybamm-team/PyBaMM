import pybamm
import numpy as np

y = pybamm.StateVector(slice(0, 4))

#u = pybamm.StateVector(slice(0, 2))
#v = pybamm.StateVector(slice(2, 4))
#
#func = 3 * u + 7 * v
#y0 = np.ones(4)
#diff = func.diff(y).simplify().evaluate(y=y0).toarray()
#
#func = 3 * u + 7 * v ** 2
#y0 = 2 * np.ones(4)
#diff = func.diff(y).simplify().evaluate(y=y0)

a = pybamm.StateVector(slice(0, 2))
print(a.id)
print(y.id)
y0 = np.array([1,2,3,4])
func = a ** 2
func.diff(y).simplify().evaluate(y=y0)
#A = pybamm.Matrix(np.hstack((2*np.eye(2), 0*np.eye(2))))
#(a * A).evaluate(y=y0)
