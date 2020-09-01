import pybamm
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(pybamm.__path__[0] + "/..")
# model = pybamm.lithium_ion.DFN(build=False,options = {"particle": "Fickian diffusion", "thermal": "lumped"})
model = pybamm.lithium_ion.DFN(
    build=False, options={"particle": "Fickian diffusion", "thermal": "lumped"}
)
model.submodels["particle cracking"] = pybamm.particle_cracking.CrackPropagation(
    model.param, "Negative"
)
model.build_model()
# param = model.default_parameter_values
chemistry = pybamm.parameter_sets.Ai2020
param = pybamm.ParameterValues(chemistry=chemistry)

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param.process_model(model)
param.process_geometry(geometry)

# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
t_eval = np.linspace(0, 3600, 100)
solution = model.default_solver.solve(model, t_eval)

# extract voltage
stress_t_n_surf = solution["Negative particle surface tangential stress"]
c_s_n = solution["Negative particle concentration"]
c_s_surf_t = solution["Negative particle surface concentration"]
disp_t = solution["Negative particle surface displacement [m]"]
l_cr_n_t = solution["Negative particle crack length"]
dl_cr = solution["Negative particle cracking rate"]
T_n = solution["Negative electrode temperature"]
t_all = solution["Time [s]"].entries
x = solution["x [m]"].entries[0:19, 0]
c_s_n = solution["Negative particle concentration"]
r_n = solution["r_n [m]"].entries[:, 0, 0]

# plot
def plot_concentrations(t):
    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 4))
    ax1.plot(x, stress_t_n_surf(t=t, x=x))
    ax1.set_xlabel(r"$x$ [m]")
    ax1.set_ylabel("$\sigma_t/E_n$")
    # ax1.set_ylim(0, 0.0015)

    (plot_c_n,) = ax2.plot(
        r_n, c_s_n(r=r_n, t=t, x=x[0])
    )  # can evaluate at arbitrary x (single representative particle)
    ax2.set_ylabel("Negative particle concentration")
    ax2.set_xlabel(r"$r_n$ [m]")
    ax2.set_ylim(0, 1)
    ax2.set_title("Close to current collector")
    ax2.grid()

    (plot_c_n,) = ax3.plot(
        r_n, c_s_n(r=r_n, t=t, x=x[10])
    )  # can evaluate at arbitrary x (single representative particle)
    ax3.set_ylabel("Negative particle concentration")
    ax3.set_xlabel(r"$r_n$ [m]")
    ax3.set_ylim(0, 1)
    ax3.set_title("In the middle")
    ax3.grid()

    (plot_c_n,) = ax4.plot(
        r_n, c_s_n(r=r_n, t=t, x=x[-1])
    )  # can evaluate at arbitrary x (single representative particle)
    ax4.set_ylabel("Negative particle concentration")
    ax4.set_xlabel(r"$r_n$ [m]")
    ax4.set_ylim(0, 1)
    ax4.set_title("Close to separator")
    ax4.grid()
    plt.show()


import ipywidgets as widgets

widgets.interact(
    plot_concentrations, t=widgets.FloatSlider(min=0, max=3600, step=10, value=0)
)
