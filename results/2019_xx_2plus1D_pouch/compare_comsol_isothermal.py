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

pybamm.set_logging_level("INFO")


"-----------------------------------------------------------------------------"
"Load comsol data"

try:
    comsol_variables = pickle.load(
        open("input/comsol_results/comsol_isothermal_2plus1D_1C.pickle", "rb")
    )
except FileNotFoundError:
    raise FileNotFoundError("COMSOL data not found. Try running load_comsol_data.py")


"-----------------------------------------------------------------------------"
"Load or set up pybamm simulation"

compute = True
filename = "results/2019_xx_2plus1D_pouch/pybamm_isothermal_2plus1D_1C.pickle.pickle"

if compute is False:
    try:
        simulation = pybamm.load_sim(filename)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Run script with compute=True first to generate results"
        )
else:
    # model
    options = {"current collector": "potential pair", "dimensionality": 2}
    pybamm_model = pybamm.lithium_ion.DFN(options)

    # parameters
    param = pybamm_model.default_parameter_values
    param.update({"C-rate": 1})

    # create custom mesh
    var = pybamm.standard_spatial_vars
    submesh_types = pybamm_model.default_submesh_types

    # cube root sequence in particles
    r_n_edges = np.linspace(0, 1, 20) ** (1 / 3)
    submesh_types["negative particle"] = pybamm.MeshGenerator(
        pybamm.UserSupplied1DSubMesh, submesh_params={"edges": r_n_edges}
    )
    r_p_edges = np.linspace(0, 1, 20) ** (1 / 3)
    submesh_types["positive particle"] = pybamm.MeshGenerator(
        pybamm.UserSupplied1DSubMesh, submesh_params={"edges": r_p_edges}
    )

    # custom mesh in y to ensure edges align with tab edges
    l_y = param.evaluate(pybamm.geometric_parameters.l_y)
    l_tab_n = param.evaluate(pybamm.geometric_parameters.l_tab_n)
    l_tab_p = param.evaluate(pybamm.geometric_parameters.l_tab_p)
    centre_tab_n = param.evaluate(pybamm.geometric_parameters.centre_y_tab_n)
    centre_tab_p = param.evaluate(pybamm.geometric_parameters.centre_y_tab_p)
    y0 = np.linspace(0, centre_tab_n - l_tab_n / 2, 5)  # mesh up to start of neg tab
    y1 = np.linspace(
        centre_tab_n - l_tab_n / 2, centre_tab_n + l_tab_n / 2, 5
    )  # mesh neg tab
    y2 = np.linspace(
        centre_tab_n + l_tab_n / 2, centre_tab_p - l_tab_p / 2, 5
    )  # mesh gap between tabs
    y3 = np.linspace(
        centre_tab_p - l_tab_p / 2, centre_tab_p + l_tab_p / 2, 5
    )  # mesh pos tab
    y4 = np.linspace(
        centre_tab_p + l_tab_p / 2, l_y, 5
    )  # mesh from pos tab to cell edge
    y_edges = np.concatenate((y0, y1[1:], y2[1:], y3[1:], y4[1:]))

    # square root sequence in z direction
    z_edges = np.linspace(0, 1, 20) ** (1 / 2)
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

    # solver
    solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, root_tol=1e-3, root_method="krylov", mode="fast")
    # solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6, root_tol=1e-6)

    # simulation object
    simulation = pybamm.Simulation(
        pybamm_model,
        parameter_values=param,
        submesh_types=submesh_types,
        var_pts=var_pts,
        solver=solver,
    )

    # build and save simulation
    simulation.build(check_model=False)
    simulation.save(filename)

"-----------------------------------------------------------------------------"
"Solve model if not already solved"

force_solve = False  # if True, then model is resolved

# discharge timescale
tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)

# solve model at comsol times
t_eval = comsol_variables["time"] / tau

if force_solve is True:
    simulation.solve(t_eval=t_eval)
elif simulation._solution is None:
    simulation.solve(t_eval=t_eval)


"-----------------------------------------------------------------------------"
"Make Comsol 'model' for comparison"

# interpolate using *dimensional* space. Note that both y and z are scaled with L_z
mesh = simulation._mesh
L_z = param.evaluate(pybamm.standard_parameters_lithium_ion.L_z)
pybamm_y = mesh["current collector"][0].edges["y"]
pybamm_z = mesh["current collector"][0].edges["z"]
y_interp = pybamm_y * L_z
z_interp = pybamm_z * L_z
#y_interp = np.linspace(pybamm_y[0], pybamm_y[-1], 100) * L_z
#z_interp = np.linspace(pybamm_z[0], pybamm_z[-1], 100) * L_z

comsol_model = shared.make_comsol_model(
    comsol_variables, mesh, param, y_interp=y_interp, z_interp=z_interp, thermal=False
)

# Process pybamm variables for which we have corresponding comsol variables
output_variables = simulation.post_process_variables(
    list(comsol_model.variables.keys())
)

"-----------------------------------------------------------------------------"
"Make plots"

t_plot = comsol_variables["time"]  # dimensional in seconds
shared.plot_t_var("Terminal voltage [V]", t_plot, comsol_model, output_variables, param)
plt.savefig("voltage.eps", format="eps", dpi=1000)
t_plot = 1800  # dimensional in seconds
shared.plot_2D_var(
    "Negative current collector potential [V]",
    t_plot,
    comsol_model,
    output_variables,
    param,
    cmap="cividis",
    error="both",
)
plt.savefig("phi_s_cn.eps", format="eps", dpi=1000)
shared.plot_2D_var(
    "Positive current collector potential [V]",
    t_plot,
    comsol_model,
    output_variables,
    param,
    cmap="viridis",
    error="both",
)
plt.savefig("phi_s_cp.eps", format="eps", dpi=1000)
shared.plot_2D_var(
    "Current collector current density [A.m-2]",
    t_plot,
    comsol_model,
    output_variables,
    param,
    cmap="plasma",
    error="both",
)
plt.savefig("current.eps", format="eps", dpi=1000)
plt.show()
