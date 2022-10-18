import pybamm
import numpy as np
import casadi


def f(x, y, z):
    return 2 * x**3 + 3 * y**2 - z


x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg, zg = np.meshgrid(x, y, z, indexing="ij", sparse=True)
data = f(xg, yg, zg)

var1 = pybamm.StateVector(slice(0, 1))
var2 = pybamm.StateVector(slice(1, 2))
var3 = pybamm.StateVector(slice(2, 3))

x_in = (x, y, z)
interp = pybamm.Interpolant(x_in, data, (var1, var2, var3), interpolator="linear")

casadi_y = casadi.MX.sym("y", 3)
interp_casadi = interp.to_casadi(y=casadi_y)

casadi_f = casadi.Function("f", [casadi_y], [interp_casadi])
y_test = np.array([1, 4, 7])

casadi_sol = casadi_f(y_test)


print("hi")
# casadi_f = casadi.Function("f", [casadi_y], [interp_casadi])
