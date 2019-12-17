import pybamm
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.ticker as ticker

import models

import pickle

path = "/home/scott/Projects/PyBaMM/results/2019_xx_2plus1D_pouchcell_part2/"
pybamm.set_logging_level("INFO")


load = True
thermal = True
c_rate = 1
t_eval = np.linspace(0, 0.1, 100)


final_time = 0.16
time_pts = [0.5 * final_time]

var_pts = {
    pybamm.standard_spatial_vars.x_n: 5,
    pybamm.standard_spatial_vars.x_s: 5,
    pybamm.standard_spatial_vars.x_p: 5,
    pybamm.standard_spatial_vars.r_n: 5,
    pybamm.standard_spatial_vars.r_p: 5,
    pybamm.standard_spatial_vars.y: 5,  # can't seem to put this to 10...
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
        "DFNCC": models.DFNCC(thermal, param),
        "SPMeCC": models.SPMeCC(thermal, param),
    }

solvers = {
    "2+1D DFN": pybamm.CasadiSolver(mode="fast"),
    "2+1D SPM": pybamm.CasadiSolver(mode="fast"),
    "2+1D SPMe": pybamm.CasadiSolver(mode="fast"),
    "DFNCC": pybamm.CasadiSolver(mode="fast"),
    "SPMeCC": None,
}


linestyles = {
    "2+1D DFN": "-",
    "2+1D SPM": ":",
    "2+1D SPMe": "--",
}

if load is False:
    current_collector_potentials = {}

    for model_name, model in models.items():

        model.solve(var_pts, c_rate, t_eval, solvers[model_name])
        variables = [
            "Discharge capacity [A.h]",
            "Time [h]",
            "Negative current collector potential [V]",
            "Positive current collector potential [V]",
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
            phi_s_cn = processed_variables["Negative current collector potential [V]"](
                t=t, y=y, z=z
            )

            # positive particle
            phi_s_cp = processed_variables["Positive current collector potential [V]"](
                t=t, y=y, z=z
            )

            current_collector_potentials[model_name] = (
                t_hours,
                dc,
                y_dim,
                z_dim,
                np.transpose(phi_s_cn),
                np.transpose(phi_s_cp),
            )

    pickle.dump(current_collector_potentials, open(path + "cc_potentials.p", "wb"))


else:
    current_collector_potentials = pickle.load(open(path + "cc_potentials.p", "rb"))


fig, axes = plt.subplots(1, len(current_collector_potentials))

# for errors
truth = current_collector_potentials["2+1D DFN"]
tim, dc, _, _, phi_s_cn_truth, phi_s_cp_truth = truth

# print("Time [h] = ", tim, " and Discharge capacity [A.h] = ", dc)

for count, (model_name, solution) in enumerate(current_collector_potentials.items()):

    t_hours, dc, y_dim, z_dim, phi_s_cn, phi_s_cp = solution

    if model_name == "2+1D DFN":
        im = axes[count].pcolormesh(
            y_dim, z_dim, phi_s_cn, vmin=None, vmax=None, shading="gouraud"
        )

        title = model_name

    else:
        error = np.abs(phi_s_cn - phi_s_cn_truth)
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

plt.subplots_adjust(
    left=0.05, bottom=0.02, right=0.96, top=0.9, wspace=0.35, hspace=0.4
)

fig.set_figheight(5)
fig.set_figwidth(13)

plt.show()


fig, axes = plt.subplots(1, len(current_collector_potentials))
for count, (model_name, solution) in enumerate(current_collector_potentials.items()):

    t_hours, dc, y_dim, z_dim, phi_s_cn, phi_s_cp = solution

    if model_name == "2+1D DFN":
        im = axes[count].pcolormesh(
            y_dim, z_dim, phi_s_cp, vmin=None, vmax=None, shading="gouraud"
        )

        title = model_name

    else:
        error = np.abs(phi_s_cp - phi_s_cp_truth)
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

plt.subplots_adjust(
    left=0.05, bottom=0.02, right=0.96, top=0.9, wspace=0.35, hspace=0.4
)

fig.set_figheight(5)
fig.set_figwidth(13)

plt.show()
