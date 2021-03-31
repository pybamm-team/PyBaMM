# #
# # Example showing how to load and solve the SPMe
# #

# import pybamm
# import numpy as np

# pybamm.set_logging_level("INFO")

# # load model
# model = pybamm.lithium_ion.SPMe()
# model.convert_to_format = "python"

# # create geometry
# geometry = model.default_geometry

# # load parameter values and process model and geometry
# param = model.default_parameter_values
# param.process_model(model)
# param.process_geometry(geometry)

# # set mesh
# mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# # discretise model
# disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
# disc.process_model(model)

# # solve model for 1 hour
# t_eval = np.linspace(0, 3600, 100)
# solution = model.default_solver.solve(model, t_eval)

# # plot
# plot = pybamm.QuickPlot(
#     solution,
#     [
#         "Negative particle concentration [mol.m-3]",
#         "Electrolyte concentration [mol.m-3]",
#         "Positive particle concentration [mol.m-3]",
#         "Current [A]",
#         "Negative electrode potential [V]",
#         "Electrolyte potential [V]",
#         "Positive electrode potential [V]",
#         "Terminal voltage [V]",
#     ],
#     time_unit="seconds",
#     spatial_unit="um",
#     variable_limits="tight",
# )
# plot.dynamic_plot()
import pybamm

param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)
model_1D_lumped = pybamm.lithium_ion.SPMe(
    {"dimensionality": 1, "current collector": "potential pair", "thermal": "x-lumped"}
)
sim_1D_lumped = pybamm.Simulation(model_1D_lumped, parameter_values=param)
sim_1D_lumped.solve([0, 36])
temp = sim_1D_lumped.solution["Cell temperature [K]"].entries
t_min = temp.min()
t_max = temp.max()
variable_limits = {
    "X-averaged cell temperature [K]": (t_min, t_max),
    "Cell temperature [K]": (t_min, t_max),
    "Volume-averaged cell temperature [K]": (t_min, t_max),
}
print(t_min, t_max)
sim_1D_lumped.plot(
    [
        "X-averaged cell temperature [K]",
        "Cell temperature [K]",
        "Volume-averaged cell temperature [K]",
    ],
    variable_limits="tight",
)
