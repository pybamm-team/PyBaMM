#
# Check convergence of pybamm model to "true" comsol solution
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
npts = [10, 20, 40, 80, 120]  # number of points per domain

"-----------------------------------------------------------------------------"
"Load comsol data"

try:
    comsol_variables = pickle.load(
        open("input/comsol_results/comsol_thermal_1C.pickle", "rb")
    )
except FileNotFoundError:
    raise FileNotFoundError("COMSOL data not found. Try running load_comsol_data.py")

comsol_t = comsol_variables["time"]

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
    "Electrolyte concentration [mol.m-3]": [None] * len(npts),
    "Electrolyte potential [V]": [None] * len(npts),
    "Negative electrode potential [V]": [None] * len(npts),
    "Positive electrode potential [V]": [None] * len(npts),
}
grid_sizes = {
    "Electrolyte concentration [mol.m-3]": [None] * len(npts),
    "Electrolyte potential [V]": [None] * len(npts),
    "Negative electrode potential [V]": [None] * len(npts),
    "Positive electrode potential [V]": [None] * len(npts),
}
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
    disc.process_model(model)

    # solve
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
    time = comsol_t / tau
    solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, root_tol=1e-6, mode="fast")
    solution = solver.solve(model, time)

    # create comsol vars to compare errors
    whole_cell = ["negative electrode", "separator", "positive electrode"]
    comsol_t = comsol_variables["time"]
    L_x = param.evaluate(pybamm.standard_parameters_lithium_ion.L_x)

    def get_interp_fun(variable_name, domain):
        """
        Create a :class:`pybamm.Function` object using the variable, to allow
        plotting with :class:`'pybamm.QuickPlot'` (interpolate in space to match
        edges, and then create function to interpolate in time)
        """
        variable = comsol_variables[variable_name]
        if domain == ["negative electrode"]:
            comsol_x = comsol_variables["x_n"]
        elif domain == ["separator"]:
            comsol_x = comsol_variables["x_s"]
        elif domain == ["positive electrode"]:
            comsol_x = comsol_variables["x_p"]
        elif domain == whole_cell:
            comsol_x = comsol_variables["x"]
        # Make sure to use dimensional space
        pybamm_x = mesh.combine_submeshes(*domain)[0].nodes * L_x
        variable = interp.interp1d(comsol_x, variable, axis=0)(pybamm_x)

        def myinterp(t):
            return interp.interp1d(comsol_t, variable)(t)[:, np.newaxis]

        # Make sure to use dimensional time
        fun = pybamm.Function(myinterp, pybamm.t * tau, name=variable_name + "_comsol")
        fun.domain = domain
        return fun

    comsol_c_e = get_interp_fun("c_e", whole_cell)
    comsol_phi_e = get_interp_fun("phi_e", whole_cell)
    comsol_phi_n = get_interp_fun("phi_n", ["negative electrode"])
    comsol_phi_p = get_interp_fun("phi_p", ["positive electrode"])
    comsol_model = pybamm.BaseModel()
    comsol_model.variables = {
        "Electrolyte concentration [mol.m-3]": comsol_c_e,
        "Electrolyte potential [V]": comsol_phi_e,
        "Negative electrode potential [V]": comsol_phi_n,
        "Positive electrode potential [V]": comsol_phi_p,
    }

    # get grid points for comparison from coarsest mesh
    if i == 0:
        x_n = mesh.combine_submeshes(*["negative electrode"])[0].nodes
        x_n_point = x_n[int(len(x_n) / 2)]
        x_s = mesh.combine_submeshes(*["separator"])[0].nodes
        x_s_point = x_s[int(len(x_s) / 2)]
        x_p = mesh.combine_submeshes(*["positive electrode"])[0].nodes
        x_p_point = x_p[int(len(x_p) / 2)]

    # compute "error" at a point
    def compute_error_and_grid_size(variable_name, t_point):
        domain = comsol_model.variables[variable_name].domain[0]
        if domain == "negative electrode":
            x_point = x_n_point
            grid_size = np.mean(
                mesh.combine_submeshes(*["negative electrode"])[0].d_edges
            )
        elif domain in [["separator"], whole_cell]:
            x_point = x_s_point
            grid_size = np.mean(mesh.combine_submeshes(*["separator"])[0].d_edges)
        elif domain == "positive electrode":
            x_point = x_p_point
            grid_size = np.mean(
                mesh.combine_submeshes(*["positive electrode"])[0].d_edges
            )

        comsol_var = pybamm.ProcessedVariable(
            comsol_model.variables[variable_name], solution.t, solution.y, mesh=mesh
        )(x=x_point, t=t_point)
        pybamm_var = pybamm.ProcessedVariable(
            model.variables[variable_name], solution.t, solution.y, mesh=mesh
        )(x=x_point, t=t_point)
        error = np.linalg.norm(pybamm_var - comsol_var)
        return error, grid_size

    t_point = comsol_t[int(len(comsol_t) / 2)] / tau
    for variable in errors.keys():
        errors[variable][i], grid_sizes[variable][i] = compute_error_and_grid_size(
            variable, t_point
        )


# plot errors
h = np.linspace(1e-3, 1e-1, 20)
plt.loglog(h, np.array(h), ":", label="h")  # plot O(h) convergence
plt.loglog(h, np.array(h) ** 2, "--", label="h**2")  # plot expected O(h^2) convergence
for var in errors.keys():
    plt.loglog(grid_sizes[var], errors[var], label=var)
plt.xlabel("h")
plt.legend()
plt.show()
