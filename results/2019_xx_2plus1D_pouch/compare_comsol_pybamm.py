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
    comsol_variables = pickle.load(
        open("comsol_{}C.pickle".format(C_rate), "rb")
        # open("comsol_normal_mesh_{}C.pickle".format(C_rate), "rb")
    )
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
# pybamm_model.use_simplify = False
geometry = pybamm_model.default_geometry

# load parameters and process model and geometry
param = pybamm_model.default_parameter_values
param.update({"C-rate": C_rates[C_rate]})
param.process_model(pybamm_model)
param.process_geometry(geometry)

# create custom mesh
var = pybamm.standard_spatial_vars
submesh_types = pybamm_model.default_submesh_types

# cube root sequence in particles
r_n_edges = np.linspace(0, 1, 11) ** (1 / 3)
submesh_types["negative particle"] = pybamm.MeshGenerator(
    pybamm.UserSupplied1DSubMesh, submesh_params={"edges": r_n_edges}
)
r_p_edges = np.linspace(0, 1, 11) ** (1 / 3)
submesh_types["positive particle"] = pybamm.MeshGenerator(
    pybamm.UserSupplied1DSubMesh, submesh_params={"edges": r_p_edges}
)

# custom mesh in y to ensure edges align with tab edges
l_y = param.evaluate(pybamm.geometric_parameters.l_y)
l_tab_n = param.evaluate(pybamm.geometric_parameters.l_tab_n)
l_tab_p = param.evaluate(pybamm.geometric_parameters.l_tab_p)
centre_tab_n = param.evaluate(pybamm.geometric_parameters.centre_y_tab_n)
centre_tab_p = param.evaluate(pybamm.geometric_parameters.centre_y_tab_p)
y0 = np.linspace(0, centre_tab_n - l_tab_n / 2, 2)  # mesh up to start of neg tab
y1 = np.linspace(
    centre_tab_n - l_tab_n / 2, centre_tab_n + l_tab_n / 2, 2
)  # mesh neg tab
y2 = np.linspace(
    centre_tab_n + l_tab_n / 2, centre_tab_p - l_tab_p / 2, 2
)  # mesh gap between tabs
y3 = np.linspace(
    centre_tab_p - l_tab_p / 2, centre_tab_p + l_tab_p / 2, 2
)  # mesh pos tab
y4 = np.linspace(centre_tab_p + l_tab_p / 2, l_y, 2)  # mesh from pos tab to cell edge
y_edges = np.concatenate((y0, y1[1:], y2[1:], y3[1:], y4[1:]))

# cube root sequence in z direction
z_edges = np.linspace(0, 1, 5) ** (1 / 3)
submesh_types["current collector"] = pybamm.MeshGenerator(
    pybamm.UserSupplied2DSubMesh,
    submesh_params={"y_edges": y_edges, "z_edges": z_edges},
)

var_pts = {
    var.x_n: 5,
    var.x_s: 5,
    var.x_p: 5,
    var.r_n: len(r_n_edges) - 1,  # Finite Volume nodes one less than edges
    var.r_p: len(r_p_edges) - 1,  # Finite Volume nodes one less than edges
    var.y: len(y_edges),
    var.z: len(z_edges),
}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

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
solver = pybamm.KLU(atol=1e-8, rtol=1e-8, root_tol=1e-8)
solution = solver.solve(pybamm_model, t_eval)


"-----------------------------------------------------------------------------"
"Make Comsol 'model' for comparison"

# interpolate using *dimensional* space. Note that both y and z are scaled with L_z
L_z = param.process_symbol(pybamm.standard_parameters_lithium_ion.L_z).evaluate()
pybamm_y = mesh["current collector"][0].edges["y"]
pybamm_z = mesh["current collector"][0].edges["z"]
y_interp = pybamm_y * L_z
z_interp = pybamm_z * L_z
# y_interp = np.linspace(pybamm_y[0], pybamm_y[-1], 100) * L_z
# z_interp = np.linspace(pybamm_z[0], pybamm_z[-1], 100) * L_z

comsol_model = shared.make_comsol_model(
    comsol_variables, mesh, param, y_interp=y_interp, z_interp=z_interp
)

# Process pybamm variables for which we have corresponding comsol variables
output_variables = {}
for var in comsol_model.variables.keys():
    try:
        output_variables[var] = pybamm.ProcessedVariable(
            pybamm_model.variables[var], solution.t, solution.y, mesh=mesh
        )
    except KeyError:
        pass


"-----------------------------------------------------------------------------"
"Make plots"

t_plot = comsol_variables["time"]  # dimensional in seconds
shared.plot_t_var("Terminal voltage [V]", t_plot, comsol_model, output_variables, param)
# plt.savefig("voltage.eps", format="eps", dpi=1000)
shared.plot_t_var(
    "Volume-averaged cell temperature [K]",
    t_plot,
    comsol_model,
    output_variables,
    param,
)
# plt.savefig("temperature_av.eps", format="eps", dpi=1000)
try:
    shared.plot_t_var(
        "Volume-averaged negative electrode irreversible heating [W.m-3]",
        t_plot,
        comsol_model,
        output_variables,
        param,
    )
    shared.plot_t_var(
        "Volume-averaged separator irreversible heating [W.m-3]",
        t_plot,
        comsol_model,
        output_variables,
        param,
    )
    shared.plot_t_var(
        "Volume-averaged positive electrode irreversible heating [W.m-3]",
        t_plot,
        comsol_model,
        output_variables,
        param,
    )
    shared.plot_t_var(
        "Volume-averaged negative electrode reversible heating [W.m-3]",
        t_plot,
        comsol_model,
        output_variables,
        param,
    )
    shared.plot_t_var(
        "Volume-averaged positive electrode reversible heating [W.m-3]",
        t_plot,
        comsol_model,
        output_variables,
        param,
    )
except KeyError:
    pass

t_plot = 1800  # dimensional in seconds
shared.plot_2D_var(
    "Negative current collector potential [V]",
    t_plot,
    comsol_model,
    output_variables,
    param,
    cmap="cividis",
    error="abs",
)
# plt.savefig("phi_s_cn.eps", format="eps", dpi=1000)
shared.plot_2D_var(
    "Positive current collector potential [V]",
    t_plot,
    comsol_model,
    output_variables,
    param,
    cmap="viridis",
    error="abs",
)
# plt.savefig("phi_s_cp.eps", format="eps", dpi=1000)
shared.plot_2D_var(
    "X-averaged cell temperature [K]",
    t_plot,
    comsol_model,
    output_variables,
    param,
    cmap="inferno",
    error="abs",
)
# plt.savefig("temperature.eps", format="eps", dpi=1000)
shared.plot_2D_var(
    "Current collector current density [A.m-2]",
    t_plot,
    comsol_model,
    output_variables,
    param,
    cmap="plasma",
    error="abs",
)
# plt.savefig("current.eps", format="eps", dpi=1000)
plt.show()
