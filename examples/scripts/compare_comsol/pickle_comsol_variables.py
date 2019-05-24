#
# script used to pickle the comsol results from csv files
#

import pybamm
import numpy as np
import os
import pandas as pd
import pickle

# change working directory to the root of pybamm
os.chdir(pybamm.__path__[0] + "/..")

# dictionary of available comsol results
C_rates = {"01": 0.1, "05": 0.5, "1": 1, "2": 2, "3": 3}

# loop over C_rates and pickle results
for key, C_rate in C_rates.items():
    # time-voltage
    comsol = pd.read_csv(
        "input/comsol_results_csv/{}C/Voltage.csv".format(key), sep=",", header=None
    )
    comsol_time = comsol[0].values
    comsol_time_npts = len(comsol_time)
    comsol_voltage = comsol[1].values

    # negative electrode potential
    comsol = pd.read_csv(
        "input/comsol_results/{}C/phi_n.csv".format(key), sep=",", header=None
    )
    comsol_x_n_npts = int(len(comsol[0].values) / comsol_time_npts)
    comsol_x_n = comsol[0].values[0:comsol_x_n_npts]
    comsol_phi_n_vals = np.reshape(
        comsol[1].values, (comsol_x_n_npts, comsol_time_npts), order="F"
    )

    # negative particle surface concentration
    comsol = pd.read_csv(
        "input/comsol_results/{}C/c_n_surf.csv".format(key), sep=",", header=None
    )
    comsol_c_n_surf_vals = np.reshape(
        comsol[1].values, (comsol_x_n_npts, comsol_time_npts), order="F"
    )

    # positive electrode potential
    comsol = pd.read_csv(
        "input/comsol_results/{}C/phi_p.csv".format(key), sep=",", header=None
    )
    comsol_x_p_npts = int(len(comsol[0].values) / comsol_time_npts)
    comsol_x_p = comsol[0].values[0:comsol_x_p_npts]
    comsol_phi_p_vals = np.reshape(
        comsol[1].values, (comsol_x_p_npts, comsol_time_npts), order="F"
    )

    # positive particle surface concentration
    comsol = pd.read_csv(
        "input/comsol_results/{}C/c_p_surf.csv".format(key), sep=",", header=None
    )
    comsol_c_p_surf_vals = np.reshape(
        comsol[1].values, (comsol_x_p_npts, comsol_time_npts), order="F"
    )

    # electrolyte concentration
    comsol = pd.read_csv(
        "input/comsol_results/{}C/c_e.csv".format(key), sep=",", header=None
    )
    comsol_x_npts = int(len(comsol[0].values) / comsol_time_npts)
    comsol_x = comsol[0].values[0:comsol_x_npts]
    comsol_c_e_vals = np.reshape(
        comsol[1].values, (comsol_x_npts, comsol_time_npts), order="F"
    )

    # electrolyte potential
    comsol = pd.read_csv(
        "input/comsol_results/{}C/phi_e.csv".format(key), sep=",", header=None
    )
    comsol_phi_e_vals = np.reshape(
        comsol[1].values, (comsol_x_npts, comsol_time_npts), order="F"
    )

    # create dictionary of variables
    comsol_variables = {
        "time": comsol_time,
        "x": comsol_x,
        "x_n": comsol_x_n,
        "x_p": comsol_x_p,
        "c_n_surf": comsol_c_n_surf_vals,
        "c_e": comsol_c_e_vals,
        "c_p_surf": comsol_c_p_surf_vals,
        "phi_n": comsol_phi_n_vals,
        "phi_e": comsol_phi_e_vals,
        "phi_p": comsol_phi_p_vals,
        "voltage": comsol_voltage,
    }

    # pickle the dictionary for later use
    pickle.dump(
        comsol_variables,
        open("input/comsol_results/comsol_{}C.pickle".format(key), "wb"),
        protocol=pickle.HIGHEST_PROTOCOL,
    )
