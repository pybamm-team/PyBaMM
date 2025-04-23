#
# Example showing how to solve the DFN with a varying ambient temperature
#

import pybamm

pybamm.set_logging_level("DEBUG")


# load model
options = {"thermal": "lumped"}
model = pybamm.lithium_ion.DFN(options)

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry


def ambient_temperature(y, z, t):
    return 300 + t * 100 / 3600


param = model.default_parameter_values
param.update(
    {"Ambient temperature [K]": ambient_temperature}, check_already_exists=False
)
param.process_model(model)
param.process_geometry(geometry)

# set mesh
var_pts = {"x_n": 30, "x_s": 30, "x_p": 30, "r_n": 10, "r_p": 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = [0, 3600 / 2]
solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-3)
solution = solver.solve(model, t_eval)

# plot
plot = pybamm.QuickPlot(
    solution, ["X-averaged cell temperature [K]", "Ambient temperature [K]"]
)
plot.dynamic_plot()
