import pybamm
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

# increase recursion limit for large expression trees
sys.setrecursionlimit(10000)

# load model and geometry
pybamm.set_logging_level("INFO")
options = {
    "current collector": "potential pair",
    "dimensionality": 2,
    "thermal": "x-lumped",
}
model = pybamm.lithium_ion.SPMe(options)
geometry = model.default_geometry

# load parameters and process model and geometry
param = model.default_parameter_values
param.update({"C-rate": 2, "Heat transfer coefficient [W.m-2.K-1]": 0.1})
param.process_model(model)
param.process_geometry(geometry)

# create custom mesh
var = pybamm.standard_spatial_vars
submesh_types = model.default_submesh_types

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
y0 = np.linspace(0, centre_tab_n - l_tab_n / 2, 3)  # mesh up to start of neg tab
y1 = np.linspace(
    centre_tab_n - l_tab_n / 2, centre_tab_n + l_tab_n / 2, 3
)  # mesh neg tab
y2 = np.linspace(
    centre_tab_n + l_tab_n / 2, centre_tab_p - l_tab_p / 2, 3
)  # mesh gap between tabs
y3 = np.linspace(
    centre_tab_p - l_tab_p / 2, centre_tab_p + l_tab_p / 2, 3
)  # mesh pos tab
y4 = np.linspace(centre_tab_p + l_tab_p / 2, l_y, 3)  # mesh from pos tab to cell edge
y_edges = np.concatenate((y0, y1[1:], y2[1:], y3[1:], y4[1:]))

# cube root sequence in z direction
z_edges = np.linspace(0, 1, 10) ** (1 / 3)
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
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# discharge timescale
tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)

# solve model
t_end = 900 / tau
t_eval = np.linspace(0, t_end, 120)
model.convert_to_format = "casadi"  # use casadi for jacobian
solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6, root_tol=1e-6)
# solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-6, root_tol=1e-6)
solution = solver.solve(model, t_eval)

# TO DO: 2+1D automated plotting
phi_s_cn = pybamm.ProcessedVariable(
    model.variables["Negative current collector potential [V]"],
    solution.t,
    solution.y,
    mesh=mesh,
)
phi_s_cp = pybamm.ProcessedVariable(
    model.variables["Positive current collector potential [V]"],
    solution.t,
    solution.y,
    mesh=mesh,
)
I = pybamm.ProcessedVariable(
    model.variables["Current collector current density [A.m-2]"],
    solution.t,
    solution.y,
    mesh=mesh,
)
T = pybamm.ProcessedVariable(
    model.variables["X-averaged cell temperature [K]"],
    solution.t,
    solution.y,
    mesh=mesh,
)
l_y = phi_s_cp.y_sol[-1]
l_z = phi_s_cp.z_sol[-1]
y_plot = np.linspace(0, l_y, 21)
z_plot = np.linspace(0, l_z, 21)


def plot(t):
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.tight_layout()
    plt.subplots_adjust(left=-0.1)

    # find t index
    ind = (np.abs(solution.t - t)).argmin()

    # negative current collector potential
    plt.subplot(221)
    phi_s_cn_plot = plt.pcolormesh(
        y_plot,
        z_plot,
        np.transpose(phi_s_cn(y=y_plot, z=z_plot, t=solution.t[ind])),
        shading="gouraud",
    )
    plt.axis([0, l_y, 0, l_z])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"$\phi_{s,cn}$ [V]")
    plt.set_cmap("cividis")
    plt.colorbar(phi_s_cn_plot)

    # positive current collector potential
    plt.subplot(222)
    phi_s_cp_plot = plt.pcolormesh(
        y_plot,
        z_plot,
        np.transpose(phi_s_cp(y=y_plot, z=z_plot, t=solution.t[ind])),
        shading="gouraud",
    )

    plt.axis([0, l_y, 0, l_z])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"$\phi_{s,cp}$ [V]")
    plt.set_cmap("viridis")
    plt.colorbar(phi_s_cp_plot)

    # current
    plt.subplot(223)
    I_plot = plt.pcolormesh(
        y_plot,
        z_plot,
        np.transpose(I(y=y_plot, z=z_plot, t=solution.t[ind])),
        shading="gouraud",
    )

    plt.axis([0, l_y, 0, l_z])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"$I$ [A.m-2]")
    plt.set_cmap("plasma")
    plt.colorbar(I_plot)

    plt.subplots_adjust(
        top=0.92, bottom=0.15, left=0.10, right=0.9, hspace=0.5, wspace=0.5
    )

    # temperature
    plt.subplot(224)
    T_plot = plt.pcolormesh(
        y_plot,
        z_plot,
        np.transpose(T(y=y_plot, z=z_plot, t=solution.t[ind])),
        shading="gouraud",
    )

    plt.axis([0, l_y, 0, l_z])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"$T$ [K]")
    plt.set_cmap("inferno")
    plt.colorbar(T_plot)

    plt.subplots_adjust(
        top=0.92, bottom=0.15, left=0.10, right=0.9, hspace=0.5, wspace=0.5
    )


plot(800 / tau)
plt.show()
