import pybamm
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir(pybamm.__path__[0] + "/..")
model = pybamm.lithium_ion.SPM(build=False)
model.submodels["negative particle"] = pybamm.particle.FickianSingleParticle(
    model.param, "Negative"
)
model.submodels["positive particle"] = pybamm.particle.FickianSingleParticle(
    model.param, "Positive"
)
model.submodels[
    "negative particle cracking"
] = pybamm.particle_cracking.CrackPropagation(model.param, "Negative")
model.build_model()
param = model.default_parameter_values

import pandas as pd

mechanics = (
    pd.read_csv(
        "pybamm/input/parameters/lithium-ion/mechanicals/lico2_graphite_Ai2020/parameters.csv",
        index_col=0,
        comment="#",
        skip_blank_lines=True,
        header=None,
    )[1][1:]
    .dropna()
    .astype(float)
    .to_dict()
)
param.update(mechanics, check_already_exists=False)
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
t_all = solution["Time [s]"].entries
x = solution["x [m]"].entries[:, 0]

# plot
c_s_n = solution["Negative particle concentration"]
r_n = solution["r_n [m]"].entries[:, 0, 0]


def plot_concentrations(t):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(t_all, stress_t_n_surf(t=t_all, x=x[0]))
    ax1.set_xlabel(r"$t$ [s]")
    ax1.set_ylabel("$\sigma_t/E_n$")
    ax1.set_ylim(0, 0.0015)

    (plot_c_n,) = ax2.plot(
        r_n, c_s_n(r=r_n, t=t, x=x[0])
    )  # can evaluate at arbitrary x (single representative particle)
    ax2.set_ylabel("Negative particle concentration")
    ax2.set_xlabel(r"$r_n$ [m]")
    ax2.set_ylim(0, 1)
    plt.show()


import ipywidgets as widgets

widgets.interact(
    plot_concentrations, t=widgets.FloatSlider(min=0, max=3600, step=10, value=0)
)
