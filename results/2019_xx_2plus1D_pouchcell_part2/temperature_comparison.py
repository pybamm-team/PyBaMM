import pybamm
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import models

import pickle

path = "/home/scott/Projects/PyBaMM/results/2019_xx_2plus1D_pouchcell_part2/"
pybamm.set_logging_level("INFO")


load = True
thermal = True
c_rate = 1
t_eval = np.linspace(0, 0.17, 100)


final_time = 0.16
time_pts = [0.5 * final_time]

var_pts = {
    pybamm.standard_spatial_vars.x_n: 5,
    pybamm.standard_spatial_vars.x_s: 5,
    pybamm.standard_spatial_vars.x_p: 5,
    pybamm.standard_spatial_vars.r_n: 5,
    pybamm.standard_spatial_vars.r_p: 5,
    pybamm.standard_spatial_vars.y: 5,
    pybamm.standard_spatial_vars.z: 5,
}

param = {
    # "Heat transfer coefficient [W.m-2.K-1]": 0.1,
    # "Negative current collector conductivity [S.m-1]": 5.96e6,
    # "Positive current collector conductivity [S.m-1]": 3.55e6,
    "Negative current collector conductivity [S.m-1]": 5.96e6,
    "Positive current collector conductivity [S.m-1]": 3.55e6,
}

if load is False:
    models = {
        "2+1D DFN": models.DFN_2p1D(thermal, param),
        "2+1D SPM": models.SPM_2p1D(thermal, param),
        "2+1D SPMe": models.SPMe_2p1D(thermal, param),
    }


linestyles = {
    "2+1D DFN": "-",
    "2+1D SPM": ":",
    "2+1D SPMe": "--",
}

if load is False:
    temperature_solution = {}

    for model_name, model in models.items():

        model.solve(var_pts, c_rate, t_eval)
        variables = [
            "Discharge capacity [A.h]",
            "Time [h]",
            "X-averaged cell temperature [K]",
            "X-averaged total heating [W.m-3]",
        ]

        y = np.linspace(0, 1.5, 100)
        z = np.linspace(0, 1, 100)
        L_z = model.param.process_symbol(pybamm.geometric_parameters.L_z).evaluate()
        y_dim = L_z * y
        z_dim = L_z * z

        processed_variables = model.processed_variables(variables)

        for t in time_pts:

            dc = processed_variables["Discharge capacity [A.h]"](t)
            t_hours = processed_variables["Time [h]"](t)

            # negative particle
            T = processed_variables["X-averaged cell temperature [K]"](t=t, y=y, z=z)

            # positive particle
            Q_tot = processed_variables["X-averaged total heating [W.m-3]"](
                t=t, y=y, z=z
            )

            temperature_solution[model_name] = (
                t_hours,
                dc,
                y_dim,
                z_dim,
                np.transpose(T),
                np.transpose(Q_tot),
            )

    pickle.dump(temperature_solution, open(path + "x_av_temperature.p", "wb"))


else:
    temperature_solution = pickle.load(open(path + "x_av_temperature.p", "rb"))


fig, axes = plt.subplots(1, len(temperature_solution))

# for errors
truth = temperature_solution["2+1D DFN"]
tim, dc, _, _, T_truth, Q_tot_truth = truth

for count, (model_name, solution) in enumerate(temperature_solution.items()):

    t_hours, dc, y_dim, z_dim, T, Q_tot = solution

    if model_name == "2+1D DFN":
        im = axes[count].pcolormesh(
            y_dim, z_dim, T, vmin=None, vmax=None, shading="gouraud"
        )

        title = model_name

    else:
        error = np.abs(T - T_truth)
        title = model_name + " vs. 2+1D DFN"
        im = axes[count].pcolormesh(y_dim, z_dim, error, shading="gouraud")

    axes[count].set_xlabel(r"$y$")
    axes[count].set_ylabel(r"$z$")
    axes[count].set_title(title)

    sfmt = ticker.ScalarFormatter(useMathText=False)
    sfmt.set_powerlimits((0, 0))

    plt.colorbar(
        im,
        ax=axes[count],
        # format=ticker.FuncFormatter(fmt),
        orientation="horizontal",
        # pad=0.2,
        format=sfmt,
    )
    # fig.colorbar(im, ax=axes[count])


# # volume average
# error = np.abs(vol_av_neg_surf_concentration - c_s_n_surf_xav_truth)
# title = "Volume Average vs. 2+1D DFN"
# count = len(x_av_surface_concentrations)
# im = axes[count].pcolormesh(y_dim, z_dim, error, shading="gouraud")
# axes[count].set_xlabel(r"$y$")
# axes[count].set_ylabel(r"$z$")
# axes[count].set_title(title)
# plt.colorbar(
#     im,
#     ax=axes[count],
#     # format=ticker.FuncFormatter(fmt),
#     orientation="horizontal",
#     # pad=0.2,
#     format=sfmt,
# )

plt.subplots_adjust(
    left=0.08, bottom=0.02, right=0.96, top=0.9, wspace=0.35, hspace=0.4
)

fig.set_figheight(4)
fig.set_figwidth(8)

plt.show()

fig, axes = plt.subplots(1, len(temperature_solution))

for count, (model_name, solution) in enumerate(temperature_solution.items()):

    t_hours, dc, y_dim, z_dim, T, Q_tot = solution

    if model_name == "2+1D DFN":
        im = axes[count].pcolormesh(
            y_dim, z_dim, Q_tot, vmin=None, vmax=None, shading="gouraud"
        )

        title = model_name

    else:
        error = np.abs(Q_tot - Q_tot_truth) / np.max(Q_tot_truth)
        title = model_name + " vs. 2+1D DFN"
        im = axes[count].pcolormesh(
            y_dim, z_dim, error, vmin=None, vmax=None, shading="gouraud"
        )

    axes[count].set_xlabel(r"$y$")
    axes[count].set_ylabel(r"$z$")
    axes[count].set_title(title)

    sfmt = ticker.ScalarFormatter(useMathText=False)
    sfmt.set_powerlimits((0, 0))

    plt.colorbar(
        im,
        ax=axes[count],
        # format=ticker.FuncFormatter(fmt),
        orientation="horizontal",
        # pad=0.2,
        format=sfmt,
    )
    # fig.colorbar(im, ax=axes[count])


# # volume average
# error = np.abs(vol_av_neg_surf_concentration - c_s_n_surf_xav_truth)
# title = "Volume Average vs. 2+1D DFN"
# count = len(x_av_surface_concentrations)
# im = axes[count].pcolormesh(y_dim, z_dim, error, shading="gouraud")
# axes[count].set_xlabel(r"$y$")
# axes[count].set_ylabel(r"$z$")
# axes[count].set_title(title)
# plt.colorbar(
#     im,
#     ax=axes[count],
#     # format=ticker.FuncFormatter(fmt),
#     orientation="horizontal",
#     # pad=0.2,
#     format=sfmt,
# )

plt.subplots_adjust(
    left=0.08, bottom=0.02, right=0.96, top=0.9, wspace=0.35, hspace=0.4
)

fig.set_figheight(4)
fig.set_figwidth(8)

plt.show()
