import pybamm
import numpy as np
import matplotlib.pyplot as plt

import models

import pickle

pybamm.set_logging_level("INFO")


load = False
thermal = True
c_rate = 1
t_eval = np.linspace(0, 0.17, 100)

final_time = 0.16
time_pts = [0.2 * 0.16, 0.5 * 0.16, 0.8 * 0.16]

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
    x_av_surface_concentrations = {}

    for model_name, model in models.items():

        model.solve(var_pts, c_rate, t_eval)
        variables = [
            "Discharge capacity [A.h]",
            "Time [h]",
            "X-averaged negative particle surface concentration",
            "X-averaged positive particle surface concentration",
        ]

        processed_variables = model.processed_variables(variables)

        x_av_surface_concentrations[model_name]["Negative particle"] = {}
        x_av_surface_concentrations[model_name]["Positive particle"] = {}
        for t in time_pts:
            # title
            dc = processed_variables["Discharge capacity [A.h]"](t)
            t_hours = processed_variables["Time [h]"](t)
            title = str(dc) + " A.h / " + str(t_hours) + " h"

            # negative particle
            c_s_n_surf_xav = processed_variables[
                "X-averaged negative particle surface concentation"
            ](t)
            x_av_surface_concentrations[model_name]["Negative particle"][
                title
            ] = c_s_n_surf_xav

            # positive particle
            c_s_p_surf_xav = processed_variables[
                "X-averaged positive particle surface concentation"
            ](t)
            x_av_surface_concentrations[model_name]["Positive particle"][
                title
            ] = c_s_p_surf_xav

    pickle.dump(
        x_av_surface_concentrations, open("x_av_surface_concentrations.p", "wb")
    )


else:
    x_av_surface_concentrations = pickle.load(
        open("x_av_surface_concentrations.p", "rb")
    )

