import pybamm
import numpy as np
import matplotlib.pyplot as plt

import matplotlib
import matplotlib.ticker as ticker

import models

import pickle

path = "/home/scott/Projects/PyBaMM/results/2019_xx_2plus1D_pouchcell_part2/"
pybamm.set_logging_level("INFO")


load = False
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
    "SPMeCC": pybamm.CasadiSolver(mode="fast"),
}


linestyles = {
    "2+1D DFN": "-",
    "2+1D SPM": ":",
    "2+1D SPMe": "--",
}

if load is False:
    current_collector_current = {}

    for model_name, model in models.items():

        model.solve(var_pts, c_rate, t_eval, solvers[model_name])
        variables = [
            "Discharge capacity [A.h]",
            "Time [h]",
            "Current collector current density [A.m-2]",
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
            i_cc = processed_variables["Current collector current density [A.m-2]"](
                t=t, y=y, z=z
            )

            current_collector_current[model_name] = (
                t_hours,
                dc,
                y_dim,
                z_dim,
                np.transpose(i_cc),
            )

    pickle.dump(current_collector_current, open(path + "cc_current.p", "wb"))


else:
    current_collector_current = pickle.load(open(path + "cc_current.p", "rb"))


fig, axes = plt.subplots(1, len(current_collector_current) - 1)

# for errors
truth = current_collector_current["2+1D DFN"]
tim, dc, _, _, i_cc_truth = truth

# print("Time [h] = ", tim, " and Discharge capacity [A.h] = ", dc)

for count, (model_name, solution) in enumerate(current_collector_current.items()):

    t_hours, dc, y_dim, z_dim, i_cc = solution

    if model_name == "2+1D DFN":
        im = axes[count].pcolormesh(
            y_dim, z_dim, i_cc, vmin=None, vmax=None, shading="gouraud", cmap="plasma"
        )

        title = model_name

    elif model_name == "SPMeCC":
        pass

    elif model_name == "DFNCC":
        error = np.abs(i_cc - i_cc_truth)
        title = r"$y$-$z$ averaged vs. 2+1D DFN"
        im = axes[count].pcolormesh(y_dim, z_dim, error, shading="gouraud")

    else:
        error = np.abs(i_cc - i_cc_truth)
        title = model_name + " vs. 2+1D DFN"
        im = axes[count].pcolormesh(y_dim, z_dim, error, shading="gouraud")

    if count < len(current_collector_current) - 1:
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
