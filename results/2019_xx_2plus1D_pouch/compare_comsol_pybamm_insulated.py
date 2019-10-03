import pybamm
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt
import shared

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

# increase recursion limit for large expression trees
sys.setrecursionlimit(10000)

"-----------------------------------------------------------------------------"
"Pick C_rate and load comsol data"

# C_rate
# NOTE: the results in pybamm stop when a voltage cutoff is reached, so
# for higher C-rate the pybamm solution may stop before the comsol solution
C_rates = {"01": 0.1, "05": 0.5, "1": 1, "2": 2, "3": 3}
C_rate = "1"  # choose the key from the above dictionary of available results

# load the comsol results
try:
    comsol_variables = pickle.load(open("comsol_nocool_{}C.pickle".format(C_rate), "rb"))
except FileNotFoundError:
    raise FileNotFoundError("COMSOL data not found. Try running load_comsol_data.py")

"-----------------------------------------------------------------------------"
"Create and solve pybamm model"

# load model and geometry
pybamm.set_logging_level("INFO")
options = {
    "current collector": "potential pair",
    "dimensionality": 2,
    "thermal": "x-lumped",
}
pybamm_model = pybamm.lithium_ion.SPM(options)
pybamm_model.use_simplify = False
geometry = pybamm_model.default_geometry

# load parameters and process model and geometry
param = pybamm_model.default_parameter_values
# adjust current to correspond to a typical current density of 24 [A.m-2]
param["Typical current [A]"] = (
    C_rates[C_rate]
    * 24
    * param.process_symbol(pybamm.geometric_parameters.A_cc).evaluate()
)
#param["Heat transfer coefficient [W.m-2.K-1]"] = 1e-6
param.process_model(pybamm_model)
param.process_geometry(geometry)

# create mesh
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
mesh = pybamm.Mesh(geometry, pybamm_model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, pybamm_model.default_spatial_methods)
disc.process_model(pybamm_model)

# discharge timescale
tau = param.process_symbol(
    pybamm.standard_parameters_lithium_ion.tau_discharge
).evaluate()

# solve model -- simulate one hour discharge
t_end = 3600 / tau
t_eval = np.linspace(0, t_end, 120)
solution = pybamm_model.default_solver.solve(pybamm_model, t_eval)


"-----------------------------------------------------------------------------"
"Make Comsol 'model' for comparison"

comsol_model = shared.make_comsol_model(comsol_variables, mesh, param)

# Process pybamm variables for which we have corresponding comsol variables
output_variables = {}
for var in comsol_model.variables.keys():
    output_variables[var] = pybamm.ProcessedVariable(
        pybamm_model.variables[var], solution.t, solution.y, mesh=mesh
    )

"-----------------------------------------------------------------------------"
"Make plots"

t_plot = comsol_variables["time"]  # dimensional in seconds
shared.plot_t_var("Terminal voltage [V]", t_plot, comsol_model, output_variables, param)
shared.plot_t_var(
    "Volume-averaged cell temperature [K]",
    t_plot,
    comsol_model,
    output_variables,
    param,
)
t_plot = 1800  # dimensional in seconds
shared.plot_2D_var(
    "Negative current collector potential [V]",
    t_plot,
    comsol_model,
    output_variables,
    param,
    cmap="cividis",
)
U_ref = param.process_symbol(
    pybamm.standard_parameters_lithium_ion.U_p_ref
    - pybamm.standard_parameters_lithium_ion.U_n_ref
).evaluate()
shared.plot_2D_var(
    "Positive current collector potential [V]",
    t_plot,
    comsol_model,
    output_variables,
    param,
    cmap="viridis",
    ref=U_ref,
)
T0 = param.process_symbol(pybamm.standard_parameters_lithium_ion.T_ref).evaluate()
shared.plot_2D_var(
    "X-averaged cell temperature [K]",
    t_plot,
    comsol_model,
    output_variables,
    param,
    cmap="inferno",
    ref=T0,
)
shared.plot_2D_var(
    "Current collector current density [A.m-2]",
    t_plot,
    comsol_model,
    output_variables,
    param,
    cmap="plasma",
    ref=T0,
)
plt.show()
