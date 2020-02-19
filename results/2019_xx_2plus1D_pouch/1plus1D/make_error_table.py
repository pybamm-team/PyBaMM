#
# Check convergence of pybamm model to "true" comsol solution (i.e. extremely fine mesh)
#

import pybamm
import os
import sys
import pickle
from pprint import pprint
import shared
import numpy as np

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

# increase recursion limit for large expression trees
sys.setrecursionlimit(100000)

# choose npts for comparison
npts = [4, 8, 16, 32]  # , 64, 128]  # number of points per domain

"-----------------------------------------------------------------------------"
"Load comsol data"

try:
    comsol_variables = pickle.load(
        open("input/comsol_results/comsol_thermal_1plus1D_1C.pickle", "rb")
    )
except FileNotFoundError:
    raise FileNotFoundError(
        "COMSOL data not found. Try running load_thermal_comsol_data.py"
    )

comsol_t = comsol_variables["time"]

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
errors = {
    "Negative current collector potential [V]": [None] * len(npts),
    "Positive current collector potential [V]": [None] * len(npts),
    "X-averaged negative particle surface concentration [mol.m-3]": [None] * len(npts),
    "X-averaged positive particle surface concentration [mol.m-3]": [None] * len(npts),
    "Current collector current density [A.m-2]": [None] * len(npts),
    "X-averaged cell temperature [K]": [None] * len(npts),
    "Terminal voltage [V]": [None] * len(npts),
}
sol_times = [None] * len(npts)
for i, model in enumerate(models):
    # process
    param.process_model(model)
    var_pts = {
        var.x_n: npts[i],
        var.x_s: npts[i],
        var.x_p: npts[i],
        var.r_n: npts[i],
        var.r_p: npts[i],
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
    comsol_model = shared.make_comsol_model(comsol_variables, mesh, param, thermal=True)

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
        else:
            z = mesh["current collector"][0].nodes
            comsol_var = pybamm.ProcessedVariable(
                comsol_model.variables[variable_name], solution.t, solution.y, mesh=mesh
            )(z=z, t=t)
            pybamm_var = pybamm.ProcessedVariable(
                model.variables[variable_name], solution.t, solution.y, mesh=mesh
            )(z=z, t=t)

        # Compute error in positive potential with respect to the voltage
        if variable_name == "Positive current collector potential [V]":
            comsol_var = comsol_var - pybamm.ProcessedVariable(
                comsol_model.variables["Terminal voltage [V]"],
                solution.t,
                solution.y,
                mesh=mesh,
            )(t=t)
            pybamm_var = pybamm_var - pybamm.ProcessedVariable(
                model.variables["Terminal voltage [V]"],
                solution.t,
                solution.y,
                mesh=mesh,
            )(t=t)

        # compute RMS difference divided by RMS of comsol_var
        error = np.sqrt(np.nanmean((pybamm_var - comsol_var) ** 2)) / np.sqrt(
            np.nanmean((comsol_var) ** 2)
        )
        return error

    for variable in errors.keys():
        try:
            errors[variable][i] = compute_error(variable)
        except KeyError:
            pass


"-----------------------------------------------------------------------------"
"Print error"
pprint("Number of points per domain")
pprint(npts)
pprint("Solve times:")
pprint(sol_times)
pprint("Errors in:")
for var, error in errors.items():
    print(var)
    pprint(error)
