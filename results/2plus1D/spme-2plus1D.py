import pybamm
import numpy as np
import matplotlib.pyplot as plt

# set logging level
pybamm.set_logging_level("DEBUG")

# load (2+1D) SPM model
options = {"bc_options": {"dimensionality": 2}}
model = pybamm.lithium_ion.SPMe(options)

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values
param.process_model(model)
param.process_geometry(geometry)

# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 2, 100)
solution = model.default_solver.solve(model, t_eval)

# TO DO: 2+1D automated plotting
phi_s_cn = pybamm.ProcessedVariable(
    model.variables["Negative current collector potential"],
    solution.t,
    solution.y,
    mesh=mesh,
)
phi_s_cp = pybamm.ProcessedVariable(
    model.variables["Positive current collector potential"],
    solution.t,
    solution.y,
    mesh=mesh,
)
l_y = phi_s_cp.x_sol[-1]
l_z = phi_s_cp.z_sol[-1]
y_plot = np.linspace(0, l_y, 51)
z_plot = np.linspace(0, l_z, 51)


def plot(t):
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.tight_layout()
    plt.subplots_adjust(left=-0.1)

    # find t index
    ind = (np.abs(solution.t - t)).argmin()

    # negative current collector potential
    plt.subplot(121)
    phi_s_cn_plot = plt.contourf(
        y_plot, z_plot, np.transpose(phi_s_cn(x=y_plot, r=z_plot, t=solution.t[ind]))
    )
    plt.axis([0, l_y, 0, l_z])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"$\phi_{s,cn}$")
    plt.set_cmap("cividis")
    plt.colorbar(phi_s_cn_plot)

    # positive current collector potential
    plt.subplot(122)
    phi_s_cp_plot = plt.contourf(
        y_plot, z_plot, np.transpose(phi_s_cp(x=y_plot, r=z_plot, t=solution.t[ind]))
    )
    plt.axis([0, l_y, 0, l_z])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"$\phi_{s,cp}$")
    plt.set_cmap("viridis")
    plt.colorbar(phi_s_cp_plot)

    plt.subplots_adjust(top=0.92, bottom=0.15, left=0.10, right=0.9, hspace=0.5, wspace=0.5)
    plt.show()


plot(solution.t[-1] / 2)
