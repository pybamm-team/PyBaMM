import pybamm
import numpy as np
import os
import sys
import pickle
import scipy.interpolate as interp
import matplotlib.pyplot as plt

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
comsol_variables = pickle.load(open("comsol_{}C.pickle".format(C_rate), "rb"))

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
pybamm_model.use_simplify_jacobian = False
geometry = pybamm_model.default_geometry

# load parameters and process model and geometry
param = pybamm_model.default_parameter_values
# adjust current to correspond to a typical current density of 24 [A.m-2]
param["Typical current [A]"] = (
    C_rates[C_rate]
    * 24
    * param.process_symbol(pybamm.geometric_parameters.A_cc).evaluate()
)
param.process_model(pybamm_model)
param.process_geometry(geometry)

# create mesh
var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: 5,
    var.x_s: 5,
    var.x_p: 5,
    var.r_n: 5,
    var.r_p: 5,
    var.y: 5,
    var.z: 5,
}
mesh = pybamm.Mesh(geometry, pybamm_model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, pybamm_model.default_spatial_methods)
disc.process_model(pybamm_model)

# discharge timescale
tau = param.process_symbol(
    pybamm.standard_parameters_lithium_ion.tau_discharge
).evaluate()

# solve model to final comsol time
t_end = comsol_variables["time"][-1] / tau
time = np.linspace(0, t_end, 50)
solution = pybamm_model.default_solver.solve(pybamm_model, time)


"-----------------------------------------------------------------------------"
"Make Comsol 'model' for comparison"

comsol_t = comsol_variables["time"]

L_y = param.process_symbol(pybamm.standard_parameters_lithium_ion.L_y).evaluate()
L_z = param.process_symbol(pybamm.standard_parameters_lithium_ion.L_z).evaluate()
pybamm_y = mesh["current collector"][0].edges["y"]
pybamm_z = mesh["current collector"][0].edges["z"]

y_plot = pybamm_y * L_z  # np.linspace(0, L_y, 20)
z_plot = pybamm_z * L_z  # np.linspace(0, L_z, 20)
grid_y, grid_z = np.meshgrid(y_plot, z_plot)


def get_interp_fun(variable, domain):
    """
    Interpolate in space to plotting nodes, and then create function to interpolate
    in time that can be called for plotting at any t.
    """
    if domain == ["negative current collector"]:
        comsol_y = comsol_variables["y_neg_cc"]
        comsol_z = comsol_variables["z_neg_cc"]
    elif domain == ["positive current collector"]:
        comsol_y = comsol_variables["y_pos_cc"]
        comsol_z = comsol_variables["z_pos_cc"]
    elif domain == ["separator"]:
        comsol_y = comsol_variables["y_sep"]
        comsol_z = comsol_variables["z_sep"]

    # Note order of rows and cols!
    interp_var = np.zeros((len(z_plot), len(y_plot), variable.shape[1]))
    for i in range(0, variable.shape[1]):
        interp_var[:, :, i] = interp.griddata(
            np.column_stack((comsol_y, comsol_z)),
            variable[:, i],
            (grid_y, grid_z),
            method="cubic",
        )

    def myinterp(t):
        return interp.interp1d(comsol_t, interp_var, axis=2)(t)

    return myinterp


# Create interpolating functions to put in comsol_model.variables dict
def comsol_voltage(t):
    return interp.interp1d(comsol_t, comsol_variables["voltage"])(t)


comsol_phi_s_cn = get_interp_fun(
    comsol_variables["phi_s_cn"], ["negative current collector"]
)
comsol_phi_s_cp = get_interp_fun(
    comsol_variables["phi_s_cp"], ["positive current collector"]
)
comsol_temperature = get_interp_fun(comsol_variables["temperature"], ["separator"])

# Create comsol model with dictionary of Matrix variables
comsol_model = pybamm.BaseModel()
comsol_model.variables = {
    "Terminal voltage [V]": comsol_voltage,
    "Negative current collector potential [V]": comsol_phi_s_cn,
    "Positive current collector potential [V]": comsol_phi_s_cp,
    "X-averaged cell temperature [K]": comsol_temperature,
}

# Process variables
output_variables = {}
for var in comsol_model.variables.keys():
    output_variables[var] = pybamm.ProcessedVariable(
        pybamm_model.variables[var], solution.t, solution.y, mesh=mesh
    )

# Plotting function
def plot(var, t, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(15, 8))

    # find t index (note: t is dimensional)
    ind = (np.abs(solution.t - t / tau)).argmin()

    # plot pybamm solution
    plt.subplot(131)
    pybamm_plot = plt.pcolormesh(
        y_plot,
        z_plot,
        np.transpose(output_variables[var](y=y_plot/L_y, z=z_plot/L_z, t=solution.t[ind])),
        shading="gouraud",
    )
    plt.axis([0, y_plot[-1], 0, z_plot[-1]])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"PyBaMM: " + var)
    plt.set_cmap(cmap)
    plt.colorbar(pybamm_plot)

    # plot comsol solution
    plt.subplot(132)
    comsol_plot = plt.pcolormesh(
        y_plot,
        z_plot,
        comsol_model.variables[var](t=solution.t[ind]),
        shading="gouraud",
    )
    plt.axis([0, y_plot[-1], 0, z_plot[-1]])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"COMSOL: " + var)
    plt.set_cmap(cmap)
    plt.colorbar(comsol_plot)

    # plot "error"
    plt.subplot(133)
    diff_plot = plt.pcolormesh(
        y_plot,
        z_plot,
        np.abs(
            np.transpose(output_variables[var](y=y_plot/L_y, z=z_plot/L_z, t=solution.t[ind]))
            - comsol_model.variables[var](t=solution.t[ind])
        ),
        shading="gouraud",
    )
    plt.axis([0, y_plot[-1], 0, z_plot[-1]])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"Error: " + var)
    plt.set_cmap(cmap)
    plt.colorbar(diff_plot)


plot("Negative current collector potential [V]", comsol_t[-1] / 2, cmap="cividis")
#plot("Positive current collector potential [V]", comsol_t[-1] / 2, cmap="cividis")
#plot("X-averaged cell temperature [K]", comsol_t[-1] / 2, cmap="cividis")
plt.show()
