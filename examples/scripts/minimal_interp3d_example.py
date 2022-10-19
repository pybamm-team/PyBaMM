import numpy as np

import pybamm
import matplotlib.pyplot as plt


def f(x, y, z):
    return 2 * x**3 + 3 * y**2 - z


x = np.linspace(1, 4, 100)
y = np.linspace(4, 7, 105)
z = np.linspace(7, 9, 110)
xg, yg, zg = np.meshgrid(x, y, z, indexing="ij", sparse=True)
data = f(xg, yg, zg)

x_in = (x, y, z)

model = pybamm.BaseModel()

a = pybamm.Variable("a")
b = pybamm.Variable("b")
c = pybamm.Variable("c")
d = pybamm.Variable("d")

interp = pybamm.Interpolant(x_in, data, (a, b, c), interpolator="linear")

model.rhs = {a: 3, b: 3, c: 2, d: interp}  # add to model
model.initial_conditions = {
    a: pybamm.Scalar(1),
    b: pybamm.Scalar(4),
    c: pybamm.Scalar(7),
    d: pybamm.Scalar(0),
}

model.variables = {
    "Something": interp,
    "a": a,
    "b": b,
    "c": c,
    "d": d,
}

# solver = pybamm.CasadiSolver()
sim = pybamm.Simulation(model)

t_eval = np.linspace(0, 1, 100)
sim.solve(t_eval)

a_eval = sim.solution["a"](t_eval)
b_eval = sim.solution["b"](t_eval)
c_eval = sim.solution["c"](t_eval)
d_eval = sim.solution["d"](t_eval)
something = sim.solution["Something"](t_eval)

difference = something - f(a_eval, b_eval, c_eval)

fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

ax[0].plot(t_eval, f(a_eval, b_eval, c_eval), label="Original")
ax[0].plot(t_eval, something, label="Interpolated")
ax[0].set_ylabel("Value")
ax[0].legend()

ax[1].plot(t_eval, np.abs(f(a_eval, b_eval, c_eval) - something), label="Original")
ax[1].set_ylabel("Difference")

ax[-1].set_xlabel("Time [s]")
for a in ax:
    a.grid()

plt.show()
