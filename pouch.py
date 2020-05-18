import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level("DEBUG")

# load model
options = {
    "current collector": "potential pair",
    "dimensionality": 2,
    "thermal": "x-lumped",
    # The below option replaces the PDEs in the particles with ODEs under the
    # assumption of fast diffusion (so that the concentration is uniform within
    # each particle). This will speed up the simulation and is an OK assumption
    # for a lot of cases. Uncomment it to switch it on/off.
    # "particle": "fast diffusion",
}
model = pybamm.lithium_ion.SPM(options)

# parameters can be updated here
param = model.default_parameter_values

# set mesh points
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

# solver
# casadi fast mode is pretty quick, but doesn't support events out of the box
# solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-3, root_tol=1e-3, mode="fast")
# KLU is sometimes better for bigger problems and supports events
solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-3, root_tol=1e-6, root_method="hybr")
# KLU performs better if you convert the final model to a python function
if isinstance(solver, pybamm.IDAKLUSolver):
    model.convert_to_format = "python"

# simulation object
simulation = pybamm.Simulation(
    model, parameter_values=param, var_pts=var_pts, solver=solver
)

# build simulation
# by default pybamm performs some checks on the discretised model. this can
# be a little slow for bigger problems, so you can turn if off. if you start to
# get obscure errors when you change things, set this to True and the error should
# get caught sooner and give a more informative message
simulation.build(check_model=False)

# solve simulation
t_eval = np.linspace(0, 3600, 100)  # time in seconds
simulation.solve(t_eval=t_eval)
solution = simulation.solution

# plotting
# TO DO: 2+1D automated plotting

# post-process variables
phi_s_cn = solution["Negative current collector potential [V]"]
phi_s_cp = solution["Positive current collector potential [V]"]
I = solution["Current collector current density [A.m-2]"]
T = solution["X-averaged cell temperature [K]"]

# get y and z points for plotting (these are non-dimensional)
l_y = phi_s_cp.y_sol[-1]
l_z = phi_s_cp.z_sol[-1]
y_plot = np.linspace(0, l_y, 21)
z_plot = np.linspace(0, l_z, 21)

# Can multiply by L_z to get dimensional y and z. Note that both y and z are
# scaled with L_z
L_z = param.evaluate(pybamm.standard_parameters_lithium_ion.L_z)
y_plot_dim = np.linspace(0, l_y, 21) * L_z
z_plot_dim = np.linspace(0, l_z, 21) * L_z


# define plotting function
def plot(time_in_seconds):
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.tight_layout()
    plt.subplots_adjust(left=-0.1)

    # get non-dim time
    tau = param.evaluate(pybamm.standard_parameters_lithium_ion.tau_discharge)
    t_non_dim = time_in_seconds / tau

    # negative current collector potential
    plt.subplot(221)
    phi_s_cn_plot = plt.pcolormesh(
        y_plot_dim,
        z_plot_dim,
        np.transpose(
            phi_s_cn(y=y_plot, z=z_plot, t=t_non_dim)
        ),  # accepts non-dim values
        shading="gouraud",
    )
    plt.axis([0, l_y * L_z, 0, l_z * L_z])
    plt.xlabel(r"$y$ [m]")
    plt.ylabel(r"$z$ [m]")
    plt.title(r"$\phi_{s,cn}$ [V]")
    plt.set_cmap("cividis")
    plt.colorbar(phi_s_cn_plot)

    # positive current collector potential
    plt.subplot(222)
    phi_s_cp_plot = plt.pcolormesh(
        y_plot_dim,
        z_plot_dim,
        np.transpose(
            phi_s_cp(y=y_plot, z=z_plot, t=t_non_dim)
        ),  # accepts non-dim values
        shading="gouraud",
    )
    plt.axis([0, l_y * L_z, 0, l_z * L_z])
    plt.xlabel(r"$y$ [m]")
    plt.ylabel(r"$z$ [m]")
    plt.title(r"$\phi_{s,cp}$ [V]")
    plt.set_cmap("viridis")
    plt.colorbar(phi_s_cp_plot)

    # through-cell current
    plt.subplot(223)
    I_plot = plt.pcolormesh(
        y_plot_dim,
        z_plot_dim,
        np.transpose(I(y=y_plot, z=z_plot, t=t_non_dim)),  # accepts non-dim values
        shading="gouraud",
    )
    plt.axis([0, l_y * L_z, 0, l_z * L_z])
    plt.xlabel(r"$y$ [m]")
    plt.ylabel(r"$z$ [m]")
    plt.title(r"$I$ [A.m-2]")
    plt.set_cmap("plasma")
    plt.colorbar(I_plot)
    # temperature
    plt.subplot(224)
    T_plot = plt.pcolormesh(
        y_plot_dim,
        z_plot_dim,
        np.transpose(T(y=y_plot, z=z_plot, t=t_non_dim)),  # accepts non-dim values
        shading="gouraud",
    )
    plt.axis([0, l_y * L_z, 0, l_z * L_z])
    plt.xlabel(r"$y$ [m]")
    plt.ylabel(r"$z$ [m]")
    plt.title(r"$T$ [K]")
    plt.set_cmap("inferno")
    plt.colorbar(T_plot)

    plt.subplots_adjust(
        top=0.92, bottom=0.15, left=0.10, right=0.9, hspace=0.5, wspace=0.5
    )
    plt.show()


# call plot
plot(1000)
