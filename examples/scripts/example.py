import pybamm
import numpy as np
import matplotlib.pyplot as plt


pybamm.set_logging_level("INFO")

C_rate = 5

options = {
    "thermal": "x-lumped",
    "current collector": "potential pair",
    "dimensionality": 2,
}
model = pybamm.lithium_ion.DFN(options=options)

var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: 7,
    var.x_s: 7,
    var.x_p: 7,
    var.r_n: 7,
    var.r_p: 7,
    var.y: 7,
    var.z: 7,
}

# var_pts = None

chemistry = pybamm.parameter_sets.NCA_Kim2011
parameter_values = pybamm.ParameterValues(chemistry=chemistry)

parameter_values.update(
    {
        "Negative current collector surface heat transfer coefficient [W.m-2.K-1]": 0,
        "Positive current collector surface heat transfer coefficient [W.m-2.K-1]": 0,
        "Negative tab heat transfer coefficient [W.m-2.K-1]": 0,
        "Positive tab heat transfer coefficient [W.m-2.K-1]": 0,
        "Edge heat transfer coefficient [W.m-2.K-1]": 500,
    }
)

solver = pybamm.CasadiSolver(mode="fast")
sim = pybamm.Simulation(
    model,
    var_pts=var_pts,
    solver=solver,
    parameter_values=parameter_values,
    C_rate=C_rate,
)
t_eval = np.linspace(0, 3500 / 6, 100)
sim.solve(t_eval=t_eval)
# sim.plot(["X-averaged cell temperature [K]"])

t = sim.solution.t
l_y = sim.parameter_values.evaluate(pybamm.geometric_parameters.l_y)
x = np.linspace(0, 1, 19)
y = np.linspace(0, l_y, 20)
z = np.linspace(0, 1, 21)

T = sim.solution["X-averaged cell temperature [K]"](t=t[-1], y=y, z=z)
I = sim.solution["Current collector current density [A.m-2]"](t=t[-1], y=y, z=z)
fig, ax = plt.subplots(1, 3)
im = ax[0].pcolormesh(
    y,
    z,
    np.transpose(T),
    # vmin=-0.003,
    # vmax=0,
    shading="gouraud",
    cmap="plasma",
)
plt.colorbar(
    im,
    ax=ax[0],
    # format=ticker.FuncFormatter(fmt),
    # orientation="horizontal",
    # pad=0.2,
    # format=sfmt,
)


cell_temp = sim.solution["X-averaged cell temperature [K]"].entries
# cell_temp = sim.solution["X-averaged cell temperature [K]"](t=t, y=cell_temp_var.y, z=z)
# cell_temp = sim.solution["Negative current collector potential [V]"](t=t, y=y, z=z)
max_temp = np.max(np.max(cell_temp, axis=0), axis=0)
min_temp = np.min(np.min(cell_temp, axis=0), axis=0)
# max_temp = np.max(cell_temp, axis=0)
# min_temp = np.min(cell_temp, axis=0)
delta_t = max_temp - min_temp


ax[1].plot(t, delta_t)

ax[2].plot(t, max_temp)
ax[2].plot(t, min_temp)

plt.show()
