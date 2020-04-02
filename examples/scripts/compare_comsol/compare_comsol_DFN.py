import pybamm
import numpy as np
import os
import pickle
import scipy.interpolate as interp

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

"-----------------------------------------------------------------------------"
"Pick C_rate and load comsol data"

# C_rate
# NOTE: the results in pybamm stop when a voltage cutoff is reached, so
# for higher C-rate the pybamm solution may stop before the comsol solution
C_rates = {"01": 0.1, "05": 0.5, "1": 1, "2": 2, "3": 3}
C_rate = "1"  # choose the key from the above dictionary of available results

# load the comsol results
comsol_results_path = pybamm.get_parameters_filepath(
    "input/comsol_results/comsol_{}C.pickle".format(C_rate)
)
comsol_variables = pickle.load(open(comsol_results_path, "rb"))

"-----------------------------------------------------------------------------"
"Create and solve pybamm model"

# load model and geometry
pybamm.set_logging_level("INFO")
pybamm_model = pybamm.lithium_ion.DFN()
geometry = pybamm_model.default_geometry

# load parameters and process model and geometry
param = pybamm_model.default_parameter_values
param["Electrode width [m]"] = 1
param["Electrode height [m]"] = 1
param["Current function [A]"] = 24 * C_rates[C_rate]
param.process_model(pybamm_model)
param.process_geometry(geometry)

# create mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 31, var.x_s: 11, var.x_p: 31, var.r_n: 11, var.r_p: 11}
mesh = pybamm.Mesh(geometry, pybamm_model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, pybamm_model.default_spatial_methods)
disc.process_model(pybamm_model)

# solve model at comsol times
time = comsol_variables["time"]
pybamm_solution = pybamm.CasadiSolver(mode="fast").solve(pybamm_model, time)


# Make Comsol 'model' for comparison
whole_cell = ["negative electrode", "separator", "positive electrode"]
comsol_t = comsol_variables["time"]
L_x = param.evaluate(pybamm.standard_parameters_lithium_ion.L_x)


def get_interp_fun(variable_name, domain):
    """
    Create a :class:`pybamm.Function` object using the variable, to allow plotting with
    :class:`pybamm.QuickPlot` (interpolate in space to match edges, and then create
    function to interpolate in time)
    """
    variable = comsol_variables[variable_name]
    if domain == ["negative electrode"]:
        comsol_x = comsol_variables["x_n"]
    elif domain == ["positive electrode"]:
        comsol_x = comsol_variables["x_p"]
    elif domain == whole_cell:
        comsol_x = comsol_variables["x"]
    # Make sure to use dimensional space
    pybamm_x = mesh.combine_submeshes(*domain)[0].nodes * L_x
    variable = interp.interp1d(comsol_x, variable, axis=0)(pybamm_x)

    def myinterp(t):
        try:
            return interp.interp1d(
                comsol_t, variable, fill_value="extrapolate", bounds_error=False,
            )(t)[:, np.newaxis]
        except ValueError as err:
            raise ValueError(
                (
                    "Failed to interpolate '{}' with time range [{}, {}] at time {}."
                    + "Original error: {}"
                ).format(variable_name, comsol_t[0], comsol_t[-1], t, err)
            )

    # Make sure to use dimensional time
    fun = pybamm.Function(
        myinterp,
        pybamm.t * pybamm_model.timescale.evaluate(),
        name=variable_name + "_comsol",
    )
    fun.domain = domain
    fun.mesh = mesh.combine_submeshes(*domain)
    fun.secondary_mesh = None
    return fun


comsol_c_n_surf = get_interp_fun("c_n_surf", ["negative electrode"])
comsol_c_e = get_interp_fun("c_e", whole_cell)
comsol_c_p_surf = get_interp_fun("c_p_surf", ["positive electrode"])
comsol_phi_n = get_interp_fun("phi_n", ["negative electrode"])
comsol_phi_e = get_interp_fun("phi_e", whole_cell)
comsol_phi_p = get_interp_fun("phi_p", ["positive electrode"])
comsol_voltage = pybamm.Function(
    interp.interp1d(
        comsol_t,
        comsol_variables["voltage"],
        fill_value="extrapolate",
        bounds_error=False,
    ),
    pybamm.t * pybamm_model.timescale.evaluate(),
)
comsol_voltage.mesh = None
comsol_voltage.secondary_mesh = None

# Create comsol model with dictionary of Matrix variables
comsol_model = pybamm.BaseModel()
comsol_model.variables = {
    "Negative particle surface concentration [mol.m-3]": comsol_c_n_surf,
    "Electrolyte concentration [mol.m-3]": comsol_c_e,
    "Positive particle surface concentration [mol.m-3]": comsol_c_p_surf,
    "Current [A]": pybamm_model.variables["Current [A]"],
    "Negative electrode potential [V]": comsol_phi_n,
    "Electrolyte potential [V]": comsol_phi_e,
    "Positive electrode potential [V]": comsol_phi_p,
    "Terminal voltage [V]": comsol_voltage,
}
# Make new solution with same t and y
comsol_solution = pybamm.Solution(pybamm_solution.t, pybamm_solution.y)
comsol_solution.model = comsol_model
# plot
plot = pybamm.QuickPlot(
    [pybamm_solution, comsol_solution],
    output_variables=comsol_model.variables.keys(),
    labels=["PyBaMM", "Comsol"],
)
plot.dynamic_plot()
