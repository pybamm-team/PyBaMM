import pybamm
import numpy as np
import matplotlib.pyplot as plt
import sys

# set logging level
pybamm.set_logging_level("INFO")

# load (2+1D) SPM model
options = {"current collector": "potential pair", "dimensionality": 2}
model = pybamm.lithium_ion.SPM(options)

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
# adjust current to correspond to a typical current density of 24 [A.m-2]
C_rate = 1
param["Typical current [A]"] = (
    C_rate * 24 * param.process_symbol(pybamm.geometric_parameters.A_cc).evaluate()
)
param.process_model(model)
param.process_geometry(geometry)

# set mesh
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
# depnding on number of points in y-z plane may need to increase recursion depth...
sys.setrecursionlimit(10000)
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model -- simulate one hour discharge
tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)
t_end = 3600 / tau.evaluate(0)
t_eval = np.linspace(0, t_end, 120)
solution = model.default_solver.solve(model, t_eval)

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
    plt.subplot(121)
    phi_s_cn_plot = plt.pcolormesh(
        y_plot,
        z_plot,
        np.transpose(phi_s_cn(y=y_plot, z=z_plot, t=solution.t[ind])),
        shading="gouraud",
    )
    plt.axis([0, l_y, 0, l_z])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"$\phi_{s,cn}$")
    plt.set_cmap("cividis")
    plt.colorbar(phi_s_cn_plot)

    # positive current collector potential
    plt.subplot(122)
    phi_s_cp_plot = plt.pcolormesh(
        y_plot,
        z_plot,
        np.transpose(phi_s_cp(y=y_plot, z=z_plot, t=solution.t[ind])),
        shading="gouraud",
    )

    plt.axis([0, l_y, 0, l_z])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"$\phi_{s,cp}$")
    plt.set_cmap("viridis")
    plt.colorbar(phi_s_cp_plot)

    plt.subplots_adjust(
        top=0.92, bottom=0.15, left=0.10, right=0.9, hspace=0.5, wspace=0.5
    )
    plt.show()


plot(solution.t[-1] / 2)
