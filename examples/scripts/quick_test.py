import numpy as np

import pybamm


def f(x, y, z):
    return 2 * x**3 + 3 * y**2 - z


x = np.linspace(1, 4, 11)
y = np.linspace(4, 7, 22)
z = np.linspace(7, 9, 33)
xg, yg, zg = np.meshgrid(x, y, z, indexing="ij", sparse=True)
data = f(xg, yg, zg)

x_in = (x, y, z)

model = pybamm.BaseModel()

a = pybamm.Variable("a")
b = pybamm.Variable("b")
c = pybamm.Variable("c")
d = pybamm.Variable("d")

interp = pybamm.Interpolant(x_in, data, (a, b, c), interpolator="linear")

model.rhs = {a: 0, b: 0, c: 0, d: interp}  # add to model
model.initial_conditions = {
    a: pybamm.Scalar(1),
    b: pybamm.Scalar(4),
    c: pybamm.Scalar(7),
    d: pybamm.Scalar(0),
}

model.variables = {
    "Something": interp,
}

sim = pybamm.Simulation(model)

t_eval = np.linspace(0, 1, 100)
sim.solve(t_eval)

something = sim.solution["Something"]


print("hi")
