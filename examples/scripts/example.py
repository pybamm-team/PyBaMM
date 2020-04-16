import pybamm
import numpy as np
import matplotlib.pyplot as plt


options = {
    "thermal": "x-lumped",
    "current collector": "potential pair",
    "dimensionality": 1,
}
model = pybamm.lithium_ion.DFN(options=options)

var = pybamm.standard_spatial_vars
# var_pts = {
#     var.x_n: 5,
#     var.x_s: 5,
#     var.x_p: 5,
#     var.r_n: 5,
#     var.r_p: 5,
#     var.y: 5,
#     var.z: 5,
# }

var_pts = None

solver = pybamm.CasadiSolver(mode="fast")
sim = pybamm.Simulation(model, var_pts=var_pts, solver=solver, C_rate=1)
sim.solve()

t = sim.solution.t
l_y = sim.parameter_values.evaluate(pybamm.geometric_parameters.l_y)
x = np.linspace(0, 1, 19)
y = np.linspace(0, l_y, 20)
z = np.linspace(0, 1, 21)

cell_temp = sim.solution["X-averaged cell temperature [K]"](t=t, y=y, z=z)
# cell_temp = sim.solution["Negative current collector potential [V]"](t=t, y=y, z=z)
# max_temp = np.max(np.max(cell_temp, axis=0), axis=0)
# min_temp = np.min(np.min(cell_temp, axis=0), axis=0)
max_temp = np.max(cell_temp, axis=0)
min_temp = np.min(cell_temp, axis=0)
delta_t = max_temp - min_temp


plt.plot(t, delta_t)
plt.show()

plt.plot(t, max_temp)
plt.plot(t, min_temp)

plt.show()
