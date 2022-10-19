import pybamm
import numpy as np
import casadi


def f(x, y, z):
    return 2 * x**3 + 3 * y**2 - z


x = np.arange(1, 4.1, 0.1)
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
pybamm_sol = interp.evaluate(y=y_test)
real_sol = f(*y_test)
print(casadi_sol, pybamm_sol, real_sol)

y_test = np.array([2, 4, 7])
casadi_sol = casadi_f(y_test)
pybamm_sol = interp.evaluate(y=y_test)
real_sol = f(*y_test)
print(casadi_sol, pybamm_sol, real_sol)

y_test = np.array([1, 5, 7])
casadi_sol = casadi_f(y_test)
pybamm_sol = interp.evaluate(y=y_test)
real_sol = f(*y_test)
print(casadi_sol, pybamm_sol, real_sol)

y_test = np.array([1, 4, 8])
casadi_sol = casadi_f(y_test)
pybamm_sol = interp.evaluate(y=y_test)
real_sol = f(*y_test)
print(casadi_sol, pybamm_sol, real_sol)

xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")
y_eval = np.stack([xg.flatten(), yg.flatten(), zg.flatten()], axis=-1)

pybamm_sol = interp.evaluate(y=y_eval)


x_ = [np.linspace(0, 1), np.linspace(0, 1)]

X = list(np.meshgrid(*x_))

x = np.column_stack([el.reshape(-1, 1) for el in X])
y = (pybamm.StateVector(slice(0, 2)), pybamm.StateVector(slice(0, 2)))
casadi_y = casadi.MX.sym("y", 2)
# linear
y_test = np.array([0.4, 0.6])
Y = (2 * x).sum(axis=1).reshape(*[len(el) for el in x_])
for interpolator in ["linear"]:
    interp = pybamm.Interpolant(x_, Y, y, interpolator=interpolator)
    interp_casadi = interp.to_casadi(y=casadi_y)
    f = casadi.Function("f", [casadi_y], [interp_casadi])
# square
y = (pybamm.StateVector(slice(0, 1)), pybamm.StateVector(slice(0, 1)))
Y = (x**2).sum(axis=1).reshape(*[len(el) for el in x_])
interp = pybamm.Interpolant(x_, Y, y, interpolator="linear")
interp_casadi = interp.to_casadi(y=casadi_y)
f = casadi.Function("f", [casadi_y], [interp_casadi])

pybamm_sol = interp.evaluate(y=y_test)
casadi_sol = f(y_test)

print("hi")
# casadi_f = casadi.Function("f", [casadi_y], [interp_casadi])
