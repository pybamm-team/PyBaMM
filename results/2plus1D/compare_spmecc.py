import pybamm
import numpy as np
import matplotlib.pyplot as plt
import sys

# set logging level and increase recursion limit
pybamm.set_logging_level("INFO")
sys.setrecursionlimit(10000)

# load current collector and SPMe models
options = {"bc_options": {"dimensionality": 2}}
spme = pybamm.lithium_ion.SPMe(options, name="SPMe (2+1D)")
spme_av = pybamm.lithium_ion.SPMe(name="SPMeCC")
cc_model = pybamm.current_collector.EffectiveResistance2D()
models = [spme, spme_av, cc_model]

# set parameters based on the spme
param = spme.default_parameter_values

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: 10,
    var.x_s: 10,
    var.x_p: 10,
    var.r_n: 10,
    var.r_p: 10,
    var.y: 10,
    var.z: 10,
}

# process model and geometry, and discretise
for model in models:
    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)


# solve current collector model
cc_solution = cc_model.default_solver.solve(cc_model)

# solve SPMe -- simulate one hour discharge
tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)
t_end = 3600 / tau.evaluate(0)
t_eval = np.linspace(0, t_end, 120)
solutions = [None] * len(models[:-1])
for i, model in enumerate(models[:-1]):
    solutions[i] = model.default_solver.solve(model, t_eval)


#phi_neg_tab = pybamm.ProcessedVariable(
#    spme.variables["Negative tab potential [V]"],
#    solutions[0].t, solutions[0].y
#)
#phi_pos_tab = pybamm.ProcessedVariable(
#    spme.variables["Positive tab potential [V]"],
#    solutions[0].t, solutions[0].y
#)
#
#current_1D = pybamm.ProcessedVariable(
#    spme_av.variables["Current collector current density"],
#    solutions[0].t, solutions[0].y
#)
#current_2D = pybamm.ProcessedVariable(
#    spme.variables["Current collector current density"],
#    solutions[0].t, solutions[0].y
#)

# plot terminal voltage
for i, model in enumerate(models[:-1]):
    t, y = solutions[i].t, solutions[i].y
    time = pybamm.ProcessedVariable(model.variables["Time [h]"], t, y)(t)
    voltage = pybamm.ProcessedVariable(model.variables["Terminal voltage [V]"], t, y)(t)

    # add current collector Ohmic losses to SPMeCC
    if model.name == "SPMeCC":
        current = pybamm.ProcessedVariable(model.variables["Current [A]"], t, y)(t)
        delta = param.process_symbol(
            pybamm.standard_parameters_lithium_ion.delta
        ).evaluate()
        R_cc = param.process_symbol(
            cc_model.variables["Effective current collector resistance [Ohm]"]
        ).evaluate(t=cc_solution.t, y=cc_solution.y)[0][0]
        cc_ohmic_losses = -delta * current * R_cc
        voltage = voltage + cc_ohmic_losses

    # plot
    plt.plot(time, voltage, lw=2, label=model.name)

# TODO: comparison of curr coll potentials

plt.xlabel("Time [h]", fontsize=15)
plt.ylabel("Terminal voltage [V]", fontsize=15)
plt.legend(fontsize=15)
plt.show()
