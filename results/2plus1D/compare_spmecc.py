import pybamm
import numpy as np
import matplotlib.pyplot as plt
import sys

# set logging level and increase recursion limit
pybamm.set_logging_level("INFO")
sys.setrecursionlimit(10000)

# load current collector and SPMe models
cc_model = pybamm.current_collector.EffectiveResistance2D()
spme_av = pybamm.lithium_ion.SPMe(name="Average SPMe")
spme = pybamm.lithium_ion.SPMe(
    {"current collector": "potential pair", "dimensionality": 2}, name="2+1D SPMe"
)
models = {"Current collector": cc_model, "Average SPMe": spme_av, "2+1D SPMe": spme}

# set parameters based on the spme
param = spme.default_parameter_values
# adjust current to correspond to a typical current density of 24 [A.m-2]
C_rate = 1
param["Typical current [A]"] = (
    C_rate * 24 * param.process_symbol(pybamm.geometric_parameters.A_cc).evaluate()
)

# set mesh
var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: 5,
    var.x_s: 5,
    var.x_p: 5,
    var.r_n: 5,
    var.r_p: 5,
    var.y: 10,
    var.z: 10,
}

# process model and geometry, and discretise
meshes = {}
for name, model in models.items():
    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)
    meshes[name] = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(meshes[name], model.default_spatial_methods)
    disc.process_model(model)

# solve models -- simulate one hour discharge
tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)
t_end = 3600 / tau.evaluate(0)
t_eval = np.linspace(0, t_end, 120)
solutions = {}
for name, model in models.items():
    if name == "Current collector":
        solutions[name] = model.default_solver.solve(model)
    else:
        solutions[name] = model.default_solver.solve(model, t_eval)

# plot terminal voltage
for name in ["Average SPMe", "2+1D SPMe"]:
    t, y = solutions[name].t, solutions[name].y
    model = models[name]
    time = pybamm.ProcessedVariable(model.variables["Time [h]"], t, y)(t)
    voltage = pybamm.ProcessedVariable(
        model.variables["Terminal voltage [V]"], t, y, mesh=meshes[name]
    )(t)

    # add current collector Ohmic losses to avergae SPMEe to get SPMeCC voltage
    if model.name == "Average SPMe":
        current = pybamm.ProcessedVariable(model.variables["Current [A]"], t, y)(t)
        delta = param.process_symbol(
            pybamm.standard_parameters_lithium_ion.delta
        ).evaluate()
        R_cc = param.process_symbol(
            cc_model.variables["Effective current collector resistance [Ohm]"]
        ).evaluate(
            t=solutions["Current collector"].t, y=solutions["Current collector"].y
        )[
            0
        ][
            0
        ]
        cc_ohmic_losses = -delta * current * R_cc
        voltage = voltage + cc_ohmic_losses

    # plot
    plt.plot(time, voltage, label=model.name)
plt.xlabel("Time [h]")
plt.ylabel("Terminal voltage [V]")
plt.legend()


# plot potentials in current collector

# get processed potentials from SPMeCC
V_av = pybamm.ProcessedVariable(
    spme_av.variables["Terminal voltage"],
    solutions["Average SPMe"].t,
    solutions["Average SPMe"].y,
    mesh=meshes["Average SPMe"],
)
I_av = pybamm.ProcessedVariable(
    spme_av.variables["Total current density"],
    solutions["Average SPMe"].t,
    solutions["Average SPMe"].y,
    mesh=meshes["Average SPMe"],
)
potentials = cc_model.get_processed_potentials(
    solutions["Current collector"], meshes["Current collector"], param, V_av, I_av
)
phi_s_cn_spmecc = potentials["Negative current collector potential [V]"]
phi_s_cp_spmecc = potentials["Positive current collector potential [V]"]

# get processed potentials from 2+1D SPMe
phi_s_cn = pybamm.ProcessedVariable(
    model.variables["Negative current collector potential [V]"],
    solutions["2+1D SPMe"].t,
    solutions["2+1D SPMe"].y,
    mesh=meshes["2+1D SPMe"],
)
phi_s_cp = pybamm.ProcessedVariable(
    model.variables["Positive current collector potential [V]"],
    solutions["2+1D SPMe"].t,
    solutions["2+1D SPMe"].y,
    mesh=meshes["2+1D SPMe"],
)

# make plot
l_y = phi_s_cp.y_sol[-1]
l_z = phi_s_cp.z_sol[-1]
y_plot = np.linspace(0, l_y, 21)
z_plot = np.linspace(0, l_z, 21)


def plot(t):
    plt.subplots(figsize=(15, 8))
    plt.tight_layout()
    plt.subplots_adjust(left=-0.1)

    # negative current collector potential
    plt.subplot(221)
    phi_s_cn_plot = plt.pcolormesh(
        y_plot,
        z_plot,
        np.transpose(phi_s_cn(y=y_plot, z=z_plot, t=t)),
        shading="gouraud",
    )
    plt.axis([0, l_y, 0, l_z])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"$\phi_{s,cn}$")
    plt.set_cmap("cividis")
    plt.colorbar(phi_s_cn_plot)
    plt.subplot(222)
    phi_s_cn_spmecc_plot = plt.pcolormesh(
        y_plot,
        z_plot,
        np.transpose(phi_s_cn_spmecc(y=y_plot, z=z_plot, t=t)),
        shading="gouraud",
    )
    plt.axis([0, l_y, 0, l_z])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"$\phi_{s,cn}$ SPMeCC")
    plt.set_cmap("cividis")
    plt.colorbar(phi_s_cn_spmecc_plot)

    # positive current collector potential
    plt.subplot(223)
    phi_s_cp_plot = plt.pcolormesh(
        y_plot,
        z_plot,
        np.transpose(phi_s_cp(y=y_plot, z=z_plot, t=t)),
        shading="gouraud",
    )

    plt.axis([0, l_y, 0, l_z])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"$\phi_{s,cp}$")
    plt.set_cmap("viridis")
    plt.colorbar(phi_s_cp_plot)
    plt.subplot(224)
    phi_s_cp_spmecc_plot = plt.pcolormesh(
        y_plot,
        z_plot,
        np.transpose(phi_s_cp_spmecc(y=y_plot, z=z_plot, t=t)),
        shading="gouraud",
    )
    plt.axis([0, l_y, 0, l_z])
    plt.xlabel(r"$y$")
    plt.ylabel(r"$z$")
    plt.title(r"$\phi_{s,cp}$ SPMeCC")
    plt.set_cmap("viridis")
    plt.colorbar(phi_s_cp_spmecc_plot)

    plt.subplots_adjust(
        top=0.92, bottom=0.15, left=0.10, right=0.9, hspace=0.5, wspace=0.5
    )


plot(solutions["2+1D SPMe"].t[-1] / 2)
plt.show()
