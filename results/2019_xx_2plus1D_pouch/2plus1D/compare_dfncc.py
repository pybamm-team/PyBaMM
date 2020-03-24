import pybamm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys

# set logging level and increase recursion limit
pybamm.set_logging_level("INFO")
sys.setrecursionlimit(1000000)

# set style
matplotlib.rc_file(
    "results/2019_xx_2plus1D_pouch/_matplotlibrc", use_default_template=True
)

# load current collector and DFN models
cc_model = pybamm.current_collector.EffectiveResistance2D()
dfn_av = pybamm.lithium_ion.DFN(name="Average DFN")
dfn = pybamm.lithium_ion.DFN(
    {"current collector": "potential pair", "dimensionality": 2}, name="2+1D DFN"
)
models = {"Current collector": cc_model, "Average DFN": dfn_av, "2+1D DFN": dfn}

# parameters
param = dfn.default_parameter_values
param.update({"C-rate": 1})

sigma_cn = param.process_symbol(
    pybamm.standard_parameters_lithium_ion.sigma_cn
).evaluate()
sigma_cp = param.process_symbol(
    pybamm.standard_parameters_lithium_ion.sigma_cp
).evaluate()
h_cool = param.process_symbol(pybamm.standard_parameters_lithium_ion.h).evaluate()
delta = param.process_symbol(pybamm.standard_parameters_lithium_ion.delta).evaluate()

# process model and geometry, and discretise
meshes = {}
for name, model in models.items():
    param.process_model(model)
    geometry = model.default_geometry
    param.process_geometry(geometry)

    # set mesh
    var = pybamm.standard_spatial_vars
    submesh_types = model.default_submesh_types
    var_pts = {
        var.x_n: 5,
        var.x_s: 5,
        var.x_p: 5,
        var.r_n: 10,
        var.r_p: 10,
        var.y: 10,
        var.z: 10,
    }
    meshes[name] = pybamm.Mesh(geometry, submesh_types, var_pts)
    disc = pybamm.Discretisation(meshes[name], model.default_spatial_methods)
    disc.process_model(model, check_model=False)

# solve models -- simulate one hour discharge
tau = param.process_symbol(pybamm.standard_parameters_lithium_ion.tau_discharge)
t_end = 3600 / tau.evaluate(0)
t_eval = np.linspace(0, t_end, 120)
solutions = {}
for name, model in models.items():
    if name == "Current collector":
        solutions[name] = model.default_solver.solve(model)
    else:
        solver = pybamm.CasadiSolver(
            atol=1e-6, rtol=1e-6, root_tol=1e-3, root_method="krylov", mode="fast"
        )
        solutions[name] = solver.solve(model, t_eval)

# plot terminal voltage
for name in ["Average DFN", "2+1D DFN"]:
    t, y = solutions[name].t, solutions[name].y
    model = models[name]
    time = pybamm.ProcessedVariable(model.variables["Time [h]"], t, y)(t)
    voltage = pybamm.ProcessedVariable(
        model.variables["Terminal voltage [V]"], t, y, mesh=meshes[name]
    )(t)

    # add current collector Ohmic losses to average DFN to get DFNCC voltage
    if model.name == "Average DFN":
        current = pybamm.ProcessedVariable(model.variables["Current [A]"], t, y)(t)
        delta = param.evaluate(pybamm.standard_parameters_lithium_ion.delta)
        R_cc = param.process_symbol(
            cc_model.variables["Effective current collector resistance [Ohm]"]
        ).evaluate(
            t=solutions["Current collector"].t, y=solutions["Current collector"].y
        )[
            0
        ][
            0
        ]
        cc_ohmic_losses = -current * R_cc
        voltage = voltage + cc_ohmic_losses

    # plot
    plt.plot(time, voltage, label=model.name)
plt.xlabel("Time [h]")
plt.ylabel("Terminal voltage [V]")
plt.legend()


R_cn = param.process_symbol(
    cc_model.variables["Effective negative current collector resistance"]
).evaluate(t=solutions["Current collector"].t, y=solutions["Current collector"].y)[0][0]
R_cp = param.process_symbol(
    cc_model.variables["Effective positive current collector resistance"]
).evaluate(t=solutions["Current collector"].t, y=solutions["Current collector"].y)[0][0]
R_cc = param.process_symbol(
    cc_model.variables["Effective current collector resistance"]
).evaluate(t=solutions["Current collector"].t, y=solutions["Current collector"].y)[0][0]
# plot potentials in current collector

# get processed potentials from DFNCC
V_av_1D = pybamm.ProcessedVariable(
    dfn_av.variables["Terminal voltage"],
    solutions["Average DFN"].t,
    solutions["Average DFN"].y,
    mesh=meshes["Average DFN"],
)
I_av = pybamm.ProcessedVariable(
    dfn_av.variables["Total current density"],
    solutions["Average DFN"].t,
    solutions["Average DFN"].y,
    mesh=meshes["Average DFN"],
)


def V_av(t):
    return V_av_1D(t) - I_av(t) * R_cc


potentials = cc_model.get_processed_potentials(
    solutions["Current collector"], meshes["Current collector"], param, V_av, I_av
)
phi_s_cn_dfncc = potentials["Negative current collector potential [V]"]
phi_s_cp_dfncc = potentials["Positive current collector potential [V]"]
R_cn = pybamm.ProcessedVariable(
    cc_model.variables["Negative current collector resistance"],
    solutions["Current collector"].t,
    solutions["Current collector"].y,
    mesh=meshes["Current collector"],
)
R_cp = pybamm.ProcessedVariable(
    cc_model.variables["Positive current collector resistance"],
    solutions["Current collector"].t,
    solutions["Current collector"].y,
    mesh=meshes["Current collector"],
)

# get processed potentials from 2+1D DFN
phi_s_cn = pybamm.ProcessedVariable(
    model.variables["Negative current collector potential [V]"],
    solutions["2+1D DFN"].t,
    solutions["2+1D DFN"].y,
    mesh=meshes["2+1D DFN"],
)
phi_s_cp = pybamm.ProcessedVariable(
    model.variables["Positive current collector potential [V]"],
    solutions["2+1D DFN"].t,
    solutions["2+1D DFN"].y,
    mesh=meshes["2+1D DFN"],
)

# make plot
l_y = phi_s_cp.y_sol[-1]
l_z = phi_s_cp.z_sol[-1]
y_plot = np.linspace(0, l_y, 21)
z_plot = np.linspace(0, l_z, 21)


def plot(t):
    # compute potentials
    pot_scale = param.evaluate(pybamm.standard_parameters_lithium_ion.thermal_voltage)
    U_ref = param.evaluate(
        pybamm.standard_parameters_lithium_ion.U_p_ref
        - pybamm.standard_parameters_lithium_ion.U_n_ref
    )

    def V_av_dim(t):
        return U_ref + V_av(t) * pot_scale

    phi_s_cn_plot = np.transpose(phi_s_cn(y=y_plot, z=z_plot, t=t))
    dfncc_phi_s_cn_plot = np.transpose(phi_s_cn_dfncc(y=y_plot, z=z_plot, t=t))
    phi_s_cp_plot = np.transpose(phi_s_cp(y=y_plot, z=z_plot, t=t)) - V_av_dim(t)
    dfncc_phi_s_cp_plot = np.transpose(
        phi_s_cp_dfncc(y=y_plot, z=z_plot, t=t)
    ) - V_av_dim(t)
    diff_phi_s_cn_plot = np.abs(dfncc_phi_s_cn_plot - phi_s_cn_plot)
    diff_phi_s_cp_plot = np.abs(dfncc_phi_s_cp_plot - phi_s_cp_plot)

    # make plot
    fig, ax = plt.subplots(2, 2, figsize=(6.4, 6))
    fig.subplots_adjust(
        left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.3, hspace=0.5
    )

    cmap_n = plt.get_cmap("cividis")
    cmap_p = plt.get_cmap("viridis")

    plot_phi_s_cn = ax[0, 0].pcolormesh(
        y_plot * 1e3, z_plot * 1e3, dfncc_phi_s_cn_plot, shading="gouraud", cmap=cmap_n
    )
    plt.colorbar(plot_phi_s_cn, ax=ax[0, 0], format="%.0e")
    plot_phi_s_cp = ax[0, 1].pcolormesh(
        y_plot * 1e3, z_plot * 1e3, dfncc_phi_s_cp_plot, shading="gouraud", cmap=cmap_p
    )
    plt.colorbar(plot_phi_s_cp, ax=ax[0, 1], format="%.0e")
    plot_diff_s_cn = ax[1, 0].pcolormesh(
        y_plot * 1e3, z_plot * 1e3, diff_phi_s_cn_plot, shading="gouraud", cmap=cmap_n
    )
    plt.colorbar(plot_diff_s_cn, ax=ax[1, 0], format="%.0e")
    plot_diff_s_cp = ax[1, 1].pcolormesh(
        y_plot * 1e3, z_plot * 1e3, diff_phi_s_cp_plot, shading="gouraud", cmap=cmap_p
    )
    plt.colorbar(plot_diff_s_cp, ax=ax[1, 1], format="%.0e")

    # set ticks
    ax[0, 0].tick_params(which="both")
    ax[0, 1].tick_params(which="both")
    ax[1, 0].tick_params(which="both")
    ax[1, 1].tick_params(which="both")

    # set labels
    ax[0, 0].set_xlabel(r"$y^*$ [mm]")
    ax[0, 0].set_ylabel(r"$z^*$ [mm]")
    ax[0, 0].set_title(r"$\phi^*_{\mathrm{s,cn}}$ [V]")
    ax[0, 1].set_xlabel(r"$y^*$ [mm]")
    ax[0, 1].set_ylabel(r"$z^*$ [mm]")
    ax[0, 1].set_title(r"$\phi^*_{\mathrm{s,cp}} - V^*$ [V]")
    ax[1, 0].set_xlabel(r"$y^*$ [mm]")
    ax[1, 0].set_ylabel(r"$z^*$ [mm]")
    ax[1, 0].set_title(r"$\phi^*_{\mathrm{s,cn}}$ (difference) [V]")
    ax[1, 1].set_xlabel(r"$y^*$ [mm]")
    ax[1, 1].set_ylabel(r"$z^*$ [mm]")
    ax[1, 1].set_title(r"$\phi^*_{\mathrm{s,cp}}$ (difference) [V]")

    ax[0, 0].text(-0.1, 1.1, "(a)", transform=ax[0, 0].transAxes)
    ax[0, 1].text(-0.1, 1.1, "(b)", transform=ax[0, 1].transAxes)
    ax[1, 0].text(-0.1, 1.1, "(c)", transform=ax[1, 0].transAxes)
    ax[1, 1].text(-0.1, 1.1, "(d)", transform=ax[1, 1].transAxes)


plot(solutions["2+1D DFN"].t[-1] / 2)
plt.savefig("dfncc_pots.pdf", format="pdf", dpi=1000)

plt.show()
