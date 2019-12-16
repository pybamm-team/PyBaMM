#
# Check convergence of pybamm model to "true" comsol solution (i.e. extremely fine mesh)
#

import pybamm
import numpy as np
import os
import pickle
import scipy.interpolate as interp
import matplotlib.pyplot as plt

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

# choose npts for comparison
npts = [4, 8, 16, 32, 64, 128]  # number of points per domain

"-----------------------------------------------------------------------------"
"Load comsol data"

savefiles = [
    "input/comsol_results/comsol_thermal_1C_extremely_coarse.pickle",
    "input/comsol_results/comsol_thermal_1C_extra_coarse.pickle",
    "input/comsol_results/comsol_thermal_1C_coarser.pickle",
    "input/comsol_results/comsol_thermal_1C_coarse.pickle",
    "input/comsol_results/comsol_thermal_1C_normal.pickle",
    "input/comsol_results/comsol_thermal_1C_fine.pickle",
    "input/comsol_results/comsol_thermal_1C_finer.pickle",
    "input/comsol_results/comsol_thermal_1C_extra_fine.pickle",
    "input/comsol_results/comsol_thermal_1C_extremely_fine.pickle",
]

# "exact" solution in the comsol solution on the finest mesh
exact_solution = pickle.load(
    open("input/comsol_results/comsol_thermal_1C_extremely_fine.pickle", "rb")
)

comsol_t = exact_solution["time"]


"-----------------------------------------------------------------------------"
"Create and solve pybamm models for different number of points per domain"

pybamm.set_logging_level("INFO")

# load models, parameters and process geometry
options = {"thermal": "x-full"}
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
    "Negative electrode potential [V]": [None] * len(npts),
    "Positive electrode potential [V]": [None] * len(npts),
    "Electrolyte potential [V]": [None] * len(npts),
    "Negative particle surface concentration [mol.m-3]": [None] * len(npts),
    "Positive particle surface concentration [mol.m-3]": [None] * len(npts),
    "Electrolyte concentration [mol.m-3]": [None] * len(npts),
    "Terminal voltage [V]": [None] * len(npts),
    "Volume-averaged cell temperature [K]": [None] * len(npts),
}
scales = {
    "Negative electrode potential [V]": param.evaluate(
        pybamm.standard_parameters_lithium_ion.thermal_voltage
    ),
    "Positive electrode potential [V]": param.evaluate(
        pybamm.standard_parameters_lithium_ion.thermal_voltage
    ),
    "Electrolyte potential [V]": param.evaluate(
        pybamm.standard_parameters_lithium_ion.thermal_voltage
    ),
    "Negative particle surface concentration [mol.m-3]": param.evaluate(
        pybamm.standard_parameters_lithium_ion.c_n_max
    ),
    "Positive particle surface concentration [mol.m-3]": param.evaluate(
        pybamm.standard_parameters_lithium_ion.c_p_max
    ),
    "Electrolyte concentration [mol.m-3]": param.evaluate(
        pybamm.standard_parameters_lithium_ion.c_e_typ
    ),
    "Terminal voltage [V]": param.evaluate(
        pybamm.standard_parameters_lithium_ion.thermal_voltage
    ),
    "Volume-averaged cell temperature [K]": param.evaluate(
        pybamm.standard_parameters_lithium_ion.Delta_T
    ),
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
    }
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, spatial_methods)
    disc.process_model(model, check_model=False)

    # solve
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
    time = comsol_t / tau
    solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, root_tol=1e-8, mode="fast")
    solution = solver.solve(model, time)
    sol_times[i] = solution.solve_time

    # create comsol vars interpolated onto pybamm mesh to compare errors
    whole_cell = ["negative electrode", "separator", "positive electrode"]
    L_x = param.evaluate(pybamm.standard_parameters_lithium_ion.L_x)
    interp_kind = "cubic"

    def get_interp_fun(variable_name, domain):
        """
        Create a :class:`pybamm.Function` object using the variable, to allow
        plotting with :class:`'pybamm.QuickPlot'` (interpolate in space to match
        edges, and then create function to interpolate in time)
        """
        variable = exact_solution[variable_name]
        if domain == ["negative electrode"]:
            comsol_x = exact_solution["x_n"]
        elif domain == ["separator"]:
            comsol_x = exact_solution["x_s"]
        elif domain == ["positive electrode"]:
            comsol_x = exact_solution["x_p"]
        elif domain == whole_cell:
            comsol_x = exact_solution["x"]
        # Make sure to use dimensional space
        pybamm_x = mesh.combine_submeshes(*domain)[0].nodes * L_x
        variable = interp.interp1d(comsol_x, variable, axis=0, kind="linear")(pybamm_x)

        def myinterp(t):
            return interp.interp1d(comsol_t, variable, kind=interp_kind)(t)[
                :, np.newaxis
            ]

        # Make sure to use dimensional time
        fun = pybamm.Function(myinterp, pybamm.t * tau, name=variable_name + "_comsol")
        fun.domain = domain
        return fun

    comsol_phi_n = get_interp_fun("phi_n", ["negative electrode"])
    comsol_phi_p = get_interp_fun("phi_p", ["positive electrode"])
    comsol_phi_e = get_interp_fun("phi_e", whole_cell)

    comsol_c_n_surf = get_interp_fun("c_n_surf", ["negative electrode"])
    comsol_c_p_surf = get_interp_fun("c_p_surf", ["positive electrode"])
    comsol_c_e = get_interp_fun("c_e", whole_cell)
    comsol_voltage = interp.interp1d(
        comsol_t, exact_solution["voltage"], kind=interp_kind
    )
    comsol_temperature_av = interp.interp1d(
        comsol_t, exact_solution["average temperature"], kind=interp_kind
    )
    comsol_model = pybamm.BaseModel()
    comsol_model.variables = {
        "Negative electrode potential [V]": comsol_phi_n,
        "Positive electrode potential [V]": comsol_phi_p,
        "Electrolyte potential [V]": comsol_phi_e,
        "Negative particle surface concentration [mol.m-3]": comsol_c_n_surf,
        "Positive particle surface concentration [mol.m-3]": comsol_c_p_surf,
        "Electrolyte concentration [mol.m-3]": comsol_c_e,
        "Terminal voltage [V]": pybamm.Function(comsol_voltage, pybamm.t * tau),
        "Volume-averaged cell temperature [K]": pybamm.Function(
            comsol_temperature_av, pybamm.t * tau
        ),
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
        else:
            x = mesh.combine_submeshes(*domain)[0].nodes
            comsol_var = pybamm.ProcessedVariable(
                comsol_model.variables[variable_name], solution.t, solution.y, mesh=mesh
            )(x=x, t=t)
            pybamm_var = pybamm.ProcessedVariable(
                model.variables[variable_name], solution.t, solution.y, mesh=mesh
            )(x=x, t=t)

        # compute RMS error
        # scale = scales[variable_name]
        scale = 1
        error = pybamm.rmse(pybamm_var / scale, comsol_var / scale)
        return error

    for variable in errors.keys():
        errors[variable][i] = compute_error(variable)


"-----------------------------------------------------------------------------"
"Compute errors for comsol models"

comsol_errors = {
    "Negative electrode potential [V]": [None] * len(savefiles),
    "Positive electrode potential [V]": [None] * len(savefiles),
    "Electrolyte potential [V]": [None] * len(savefiles),
    "Negative particle surface concentration [mol.m-3]": [None] * len(savefiles),
    "Positive particle surface concentration [mol.m-3]": [None] * len(savefiles),
    "Electrolyte concentration [mol.m-3]": [None] * len(savefiles),
    "Terminal voltage [V]": [None] * len(savefiles),
    "Volume-averaged cell temperature [K]": [None] * len(savefiles),
}
exact_voltage = exact_solution["voltage"]
comsol_sol_times = [None] * len(savefiles)
# plt.figure()
for i, file in enumerate(savefiles):
    comsol_variables = pickle.load(open(file, "rb"))
    # comsol_time = comsol_variables["time"]
    comsol_voltage = comsol_variables["voltage"]
    # plt.plot(comsol_time, comsol_voltage, label=file)
    # compute RMS error
    # scale = scales["Terminal voltage [V]"]
    scale = 1
    comsol_errors["Terminal voltage [V]"][i] = pybamm.rmse(
        comsol_voltage / scale, exact_voltage / scale
    )
    comsol_sol_times[i] = comsol_variables["solution_time"]
# plt.legend()
# plt.show()

"-----------------------------------------------------------------------------"
"Plot error"
fig, ax = plt.subplots(1, 2, figsize=(6.0, 3.75))
ax[0].text(-0.1, 1.05, "(a)", transform=ax[0].transAxes)
ax[0].set_xlabel("Solution time [s]")
ax[0].set_ylabel("RMS Voltage difference [V]")
ax[1].text(-0.1, 1.05, "(b)", transform=ax[1].transAxes)
ax[1].set_xlabel("Solution time [s]")
ax[1].set_ylabel("RMS Voltage difference [V]")
ax[0].set_ylabel("Terminal voltage [V]")

ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].plot(sol_times, errors["Terminal voltage [V]"], "ko-", label="PyBaMM")
ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].plot(
    comsol_sol_times, comsol_errors["Terminal voltage [V]"], "ko-", label="COMSOL"
)

ax[0].legend()
ax[1].legend()

plt.tight_layout()

plt.savefig("1D_error_vs_time.eps", format="eps", dpi=1000)
plt.show()
