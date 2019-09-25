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
C_rate = 1

# load the comsol results
comsol_variables = pickle.load(
    open(
        "results/2019_09_sulzer_thesis/compare_comsol/comsol_data_{}C.pickle".format(
            C_rate
        ),
        "rb",
    )
)

"-----------------------------------------------------------------------------"
"Create and solve pybamm model"

# load model and geometry
pybamm.set_logging_level("INFO")
pybamm_model = pybamm.lead_acid.Full()
geometry = pybamm_model.default_geometry

# load parameters and process model and geometry
param = pybamm_model.default_parameter_values
param["Typical current [A]"] = 17 * C_rate
# Change the t_plus function to agree with Comsol
param["Darken thermodynamic factor"] = np.ones_like
param["MacInnes t_plus function"] = lambda x: 1 - 2 * x
param.process_model(pybamm_model)
param.process_geometry(geometry)

# create mesh
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 50, var.x_s: 50, var.x_p: 50}
mesh = pybamm.Mesh(geometry, pybamm_model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, pybamm_model.default_spatial_methods)
disc.process_model(pybamm_model)

# discharge timescale
tau = param.process_symbol(pybamm.standard_parameters_lead_acid.tau_discharge)

# solve model at comsol times
comsol_t = comsol_variables["time"] * 3600
time = comsol_t / tau.evaluate(0)
solution = pybamm_model.default_solver.solve(pybamm_model, time)


# Make Comsol 'model' for comparison
whole_cell = ["negative electrode", "separator", "positive electrode"]
L_x = param.process_symbol(pybamm.standard_parameters_lithium_ion.L_x).evaluate()


def get_interp_fun(var_name, domain):
    """
    Create a :class:`pybamm.Function` object using the variable, to allow plotting with
    :class:`'pybamm.QuickPlot'` (interpolate in space to match edges, and then create
    function to interpolate in time)
    """
    if domain == ["negative electrode"]:
        comsol_x = comsol_variables["x_n"]
    elif domain == ["positive electrode"]:
        comsol_x = comsol_variables["x_p"]
    elif domain == whole_cell:
        comsol_x = comsol_variables["x"]
    # Make sure to use dimensional space
    pybamm_x = mesh.combine_submeshes(*domain)[0].nodes * L_x
    variable = interp.interp1d(comsol_x, comsol_variables[var_name], axis=0)(pybamm_x)

    def myinterp(t):
        return interp.interp1d(comsol_t, variable)(t)[:, np.newaxis]

    # Make sure to use dimensional time
    fun = pybamm.Function(myinterp, pybamm.t * tau, name=var_name)
    fun.domain = domain
    return fun


comsol_c_e = get_interp_fun("c_e", whole_cell)
comsol_eps_n = get_interp_fun("eps_n", ["negative electrode"])
comsol_phi_e = get_interp_fun("phi_e", whole_cell)
comsol_eps_p = get_interp_fun("eps_p", ["positive electrode"])
comsol_voltage = interp.interp1d(comsol_t, comsol_variables["voltage"])
# Create comsol model with dictionary of Matrix variables
comsol_model = pybamm.BaseModel()
comsol_model.variables = {
    "Electrolyte concentration [mol.m-3]": comsol_c_e,
    "Current [A]": pybamm_model.variables["Current [A]"],
    "Negative electrode porosity": comsol_eps_n,
    "Electrolyte potential [V]": comsol_phi_e,
    "Positive electrode porosity": comsol_eps_p,
    "Terminal voltage [V]": pybamm.Function(comsol_voltage, pybamm.t * tau),
}

# plot
plot = pybamm.QuickPlot(
    [pybamm_model, comsol_model],
    mesh,
    [solution, solution],
    output_variables=comsol_model.variables.keys(),
    labels=["PyBaMM", "Comsol"],
)
plot.dynamic_plot()
