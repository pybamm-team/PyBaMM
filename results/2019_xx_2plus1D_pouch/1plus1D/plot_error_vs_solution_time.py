#
# Check convergence of pybamm model to "true" comsol solution (i.e. extremely fine mesh)
#

import pybamm
import numpy as np
import os
import sys
import pickle
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import shared

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

# increase recursion limit for large expression trees
sys.setrecursionlimit(100000)

# choose npts for comparison
npts = [4, 8, 16, 32, 64]  # number of points per domain

"-----------------------------------------------------------------------------"
"Load comsol data"

savefiles = [
    "input/comsol_results/comsol_thermal_1plus1D_1C_extra_coarse.pickle",
    "input/comsol_results/comsol_thermal_1plus1D_1C_coarse.pickle",
    "input/comsol_results/comsol_thermal_1plus1D_1C_normal.pickle",
    "input/comsol_results/comsol_thermal_1plus1D_1C_fine.pickle",
]

# "exact" solution in the comsol solution on the finest mesh
exact_solution = pickle.load(
    open("input/comsol_results/comsol_thermal_1plus1D_1C_fine.pickle", "rb")
)

comsol_t = exact_solution["time"]


"-----------------------------------------------------------------------------"
"Create and solve pybamm models for different number of points per domain"

pybamm.set_logging_level("INFO")

# load models, parameters and process geometry
options = {
    "current collector": "potential pair",
    "dimensionality": 1,
    "thermal": "x-lumped",
}
models = [None] * len(npts)
for i in range(len(npts)):
    models[i] = pybamm.lithium_ion.DFN(options)
param = models[0].default_parameter_values
param.update({"C-rate": 1})
geometry = models[0].default_geometry
param.process_geometry(geometry)

# set spatial methods
spatial_methods = models[0].default_spatial_methods
var = pybamm.standard_spatial_vars

# discretise and solve models. Then compute "error"
errors = {"Terminal voltage [V]": [None] * len(npts)}

sol_times = [None] * len(npts)
for i, model in enumerate(models):
    # process
    param.process_model(model)
    var_pts = {
        var.x_n: 15,
        var.x_s: 10,
        var.x_p: 15,
        var.r_n: 15,
        var.r_p: 15,
        var.z: npts[i],
    }
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, spatial_methods)
    disc.process_model(model, check_model=False)

    # solve
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
    time = comsol_t / tau
    solver = pybamm.CasadiSolver(
        atol=1e-6, rtol=1e-6, root_tol=1e-3, root_method="hybr", mode="fast"
    )
    solution = solver.solve(model, time)
    sol_times[i] = solution.solve_time

    # create comsol vars interpolated onto pybamm mesh to compare errors
    # comsol_model = shared.make_comsol_model(exact_solution, mesh, param, thermal=True)
    comsol_model = pybamm.BaseModel()
    comsol_voltage = interp.interp1d(comsol_t, exact_solution["voltage"], kind="cubic")
    comsol_model.variables = {
        "Terminal voltage [V]": pybamm.Function(
            comsol_voltage, pybamm.t * tau, name="voltage_comsol"
        )
    }
    # compute "error" using times up to voltage cut off
    t = solution.t

    # Note: casadi doesnt support events so we find this time after the solve
    if isinstance(solver, pybamm.CasadiSolver):
        V_cutoff = param.evaluate(
            pybamm.standard_parameters_lithium_ion.voltage_low_cut_dimensional
        )
        voltage = pybamm.ProcessedVariable(
            model.variables["Terminal voltage [V]"], solution.t, solution.y, mesh=mesh
        )(time)
        # only use times up to the voltage cutoff
        voltage_OK = voltage[voltage > V_cutoff]
        t = t[0 : len(voltage_OK)]

    def compute_error(variable_name):
        domain = comsol_model.variables[variable_name].domain

        if domain == []:
            comsol_var = pybamm.ProcessedVariable(
                comsol_model.variables[variable_name], solution.t, solution.y, mesh=mesh
            )(t=t)
            pybamm_var = pybamm.ProcessedVariable(
                model.variables[variable_name], solution.t, solution.y, mesh=mesh
            )(t=t)
        elif domain == ["current collector"]:
            z = mesh.combine_submeshes(*domain)[0].nodes
            comsol_var = pybamm.ProcessedVariable(
                comsol_model.variables[variable_name], solution.t, solution.y, mesh=mesh
            )(z=z, t=t)
            pybamm_var = pybamm.ProcessedVariable(
                model.variables[variable_name], solution.t, solution.y, mesh=mesh
            )(z=z, t=t)

        # compute RMS error
        error = pybamm.rmse(pybamm_var, comsol_var)
        return error

    for variable in errors.keys():
        errors[variable][i] = compute_error(variable)


"-----------------------------------------------------------------------------"
"Compute errors for comsol models"

comsol_errors = {"Terminal voltage [V]": [None] * len(savefiles)}
exact_voltage = exact_solution["voltage"]
comsol_sol_times = [None] * len(savefiles)

for i, file in enumerate(savefiles):
    comsol_variables = pickle.load(open(file, "rb"))
    comsol_voltage = comsol_variables["voltage"]

    # compute RMS error
    comsol_errors["Terminal voltage [V]"][i] = pybamm.rmse(
        comsol_voltage, exact_voltage
    )
    comsol_sol_times[i] = comsol_variables["solution_time"]


"-----------------------------------------------------------------------------"
"Plot error"
fig, ax = plt.subplots(1, 2, figsize=(6.0, 3.75))
ax[0].text(-0.1, 1.1, "(a)", transform=ax[0].transAxes)
ax[0].set_xlabel("Solution time [s]")
ax[0].set_ylabel("RMS Voltage difference [V]")
ax[1].text(-0.1, 1.1, "(b)", transform=ax[1].transAxes)
ax[1].set_xlabel("Solution time [s]")
ax[1].set_ylabel("RMS Voltage difference [V]")
ax[0].set_ylabel("Terminal voltage [V]")

# ax[0].set_xscale("log")
# ax[0].set_yscale("log")
ax[0].plot(sol_times, errors["Terminal voltage [V]"], "ko-", label="PyBaMM")
# ax[1].set_xscale("log")
# ax[1].set_yscale("log")
ax[1].plot(
    comsol_sol_times, comsol_errors["Terminal voltage [V]"], "ko-", label="COMSOL"
)

ax[0].legend()
ax[1].legend()

plt.tight_layout()

plt.savefig("1plus1D_error_vs_time.eps", format="eps", dpi=1000)
plt.show()
