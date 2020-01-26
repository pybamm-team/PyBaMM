#
# Example showing how to load and solve the DFN
#

import pybamm
import numpy as np

pybamm.set_logging_level("INFO")


def CCCV(variables):
    I = variables["Current [A]"]
    V = variables["Terminal voltage [V]"]
    s_I = pybamm.InputParameter("Current switch")
    s_V = pybamm.InputParameter("Voltage switch")
    # s_P = pybamm.InputParameter("Power switch")
    return s_I * (I - pybamm.InputParameter("Current input [A]")) + s_V * (
        V - pybamm.InputParameter("Voltage input [V]")
    )


# load model
model = pybamm.lithium_ion.DFN({"operating mode": CCCV})

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 30, var.x_s: 30, var.x_p: 30, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 0.2, 100)
solver = model.default_solver
solver.rtol = 1e-3
solver.atol = 1e-6
solution = None

solution = solver.step(
    solution,
    model,
    0.01,
    npts=100,
    inputs={
        "Current switch": 1,
        "Voltage switch": 0,
        "Current input [A]": -1,
        "Voltage input [V]": 0,
    },
)
solution = solver.step(
    solution,
    model,
    0.01,
    npts=100,
    inputs={
        "Current switch": 1,
        "Voltage switch": 0,
        "Current input [A]": -0.5,
        "Voltage input [V]": 0,
    },
)
solution = solver.step(
    solution,
    model,
    0.01,
    npts=100,
    inputs={
        "Current switch": 0,
        "Voltage switch": 1,
        "Current input [A]": 0,
        "Voltage input [V]": 4.1,
    },
)

# plot
plot = pybamm.QuickPlot(solution)
plot.dynamic_plot()
