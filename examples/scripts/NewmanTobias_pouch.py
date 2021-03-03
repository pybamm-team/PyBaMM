import pybamm
import numpy as np
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")

# load model
options = {
    "current collector": "potential pair",
    "dimensionality": 2,  # 2D current collectors
    "thermal": "x-lumped",  # thermal model (ignores through-cell variation)
}
model = pybamm.lithium_ion.NewmanTobias(options)

# parameters can be updated here
param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)
# e.g. reduce current collector conductivity by a factor of 100
sigma_ccn = param["Negative current collector conductivity [S.m-1]"]
sigma_ccp = param["Positive current collector conductivity [S.m-1]"]
param.update(
    {
        "Negative current collector conductivity [S.m-1]": sigma_ccn / 100,
        "Positive current collector conductivity [S.m-1]": sigma_ccp / 100,
    }
)

# set mesh points
var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: 5,  # negative electrode
    var.x_s: 5,  # separator
    var.x_p: 5,  # positive electrode
    var.r_n: 5,  # negative particle
    var.r_p: 5,  # positive particle
    var.y: 10,  # current collector y-direction
    var.z: 10,  # current collector z-direction
}

# solver
# casadi fast mode is quick, but doesn't support events (e.g. voltage cut-off)
# out of the box. turn mode to "safe" for events
solver = pybamm.CasadiSolver(atol=1e-6, rtol=1e-3, root_tol=1e-3, mode="fast")

# simulation object
simulation = pybamm.Simulation(
    model, parameter_values=param, var_pts=var_pts, solver=solver
)

# solve simulation
t_eval = np.linspace(0, 600, 100)  # time in seconds
solution = simulation.solve(t_eval=t_eval)

# plotting ---------------------------------------------------------------------

# create a quick slider plot
# simulation.plot(
#    ["X-averaged cell temperature [K]", "Volume-averaged cell temperature [K]"],
#    variable_limits="tight",
# )

# post-process variables
phi_s_cn = solution["Negative current collector potential [V]"]
phi_s_cp = solution["Positive current collector potential [V]"]
I = solution["Current collector current density [A.m-2]"]
T = solution["X-averaged cell temperature [K]"]

# create y and z mesh for plotting
L_y = param.evaluate(model.param.L_y)
L_z = param.evaluate(model.param.L_z)
y_plot = np.linspace(0, L_y, 21)
z_plot = np.linspace(0, L_z, 21)


# define plotting function
def plot(t):
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.tight_layout()
    plt.subplots_adjust(left=-0.1)

    # negative current collector potential
    plt.subplot(221)
    phi_s_cn_plot = plt.pcolormesh(
        y_plot, z_plot, phi_s_cn(y=y_plot, z=z_plot, t=t), shading="gouraud"
    )
    plt.axis([0, L_y, 0, L_z])
    plt.xlabel(r"$y$ [m]")
    plt.ylabel(r"$z$ [m]")
    plt.title(r"$\phi_{s,cn}$ [V]")
    plt.set_cmap("cividis")
    plt.colorbar(phi_s_cn_plot)

    # positive current collector potential
    plt.subplot(222)
    phi_s_cp_plot = plt.pcolormesh(
        y_plot, z_plot, phi_s_cp(y=y_plot, z=z_plot, t=t), shading="gouraud"
    )
    plt.axis([0, L_y, 0, L_z])
    plt.xlabel(r"$y$ [m]")
    plt.ylabel(r"$z$ [m]")
    plt.title(r"$\phi_{s,cp}$ [V]")
    plt.set_cmap("viridis")
    plt.colorbar(phi_s_cp_plot)

    # through-cell current
    plt.subplot(223)
    I_plot = plt.pcolormesh(
        y_plot, z_plot, I(y=y_plot, z=z_plot, t=t), shading="gouraud"
    )
    plt.axis([0, L_y, 0, L_z])
    plt.xlabel(r"$y$ [m]")
    plt.ylabel(r"$z$ [m]")
    plt.title(r"$I$ [A.m-2]")
    plt.set_cmap("plasma")
    plt.colorbar(I_plot)

    # temperature
    plt.subplot(224)
    T_plot = plt.pcolormesh(
        y_plot, z_plot, T(y=y_plot, z=z_plot, t=t), shading="gouraud"
    )
    plt.axis([0, L_y, 0, L_z])
    plt.xlabel(r"$y$ [m]")
    plt.ylabel(r"$z$ [m]")
    plt.title(r"$T$ [K]")
    plt.set_cmap("inferno")
    plt.colorbar(T_plot)

    plt.subplots_adjust(
        top=0.92, bottom=0.15, left=0.10, right=0.9, hspace=0.5, wspace=0.5
    )
    plt.show()


# call plot with time in seconds
plot(t_eval[-1] / 2)
