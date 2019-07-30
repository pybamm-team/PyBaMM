import pybamm
import numpy as np
import matplotlib.pyplot as plt

# set logging level
pybamm.set_logging_level("INFO")

# load current collector and SPMe models
cell_model = pybamm.lithium_ion.SPMe()
cc_model = pybamm.current_collector.EffectiveResistance2D()
models = [cell_model, cc_model]

# set parameters based on the cell model
param = cell_model.default_parameter_values

# make current collectors not so conductive, just for illustrative purposes
param["Negative current collector conductivity [S.m-1]"] = 5.96e5
param["Positive current collector conductivity [S.m-1]"] = 3.55e5

# process model and geometry, and discretise
for model in models:
    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)


# solve current collector model
cc_solution = cc_model.default_solver.solve(cc_model)

# solve SPMe -- simulate one hour discharge
tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)
t_end = 3600 / tau.evaluate(0)
t_eval = np.linspace(0, t_end, 120)
solution = cell_model.default_solver.solve(cell_model, t_eval)

# plot terminal voltage
t, y = solution.t, solution.y
time = pybamm.ProcessedVariable(cell_model.variables["Time [h]"], t, y)(t)
voltage = pybamm.ProcessedVariable(cell_model.variables["Terminal voltage [V]"], t, y)
current = pybamm.ProcessedVariable(cell_model.variables["Current [A]"], t, y)(t)
delta = param.process_symbol(pybamm.standard_parameters_lithium_ion.delta).evaluate()
R_cc = param.process_symbol(
    cc_model.variables["Effective current collector resistance [Ohm]"]
).evaluate(t=cc_solution.t, y=cc_solution.y)[0][0]
cc_ohmic_losses = -delta * current * R_cc

plt.plot(time, voltage(t), lw=2, label="SPMe")
plt.plot(time, voltage(t) + cc_ohmic_losses, lw=2, label="SPMeCC")
plt.xlabel("Time [h]", fontsize=15)
plt.ylabel("Terminal voltage [V]", fontsize=15)
plt.legend(fontsize=15)
plt.show()
