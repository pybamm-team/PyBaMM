#
# Load data from 1D isothermal comsol model and pickle
#

import pybamm
import os
import pandas as pd
import pickle
import numpy as np

# change working directory the root of pybamm
os.chdir(pybamm.root_dir())

# set filepaths for data and names of file to pickle to
paths = [
    "input/comsol_results_csv/isothermal/05C/",
    "input/comsol_results_csv/isothermal/1C/",
    "input/comsol_results_csv/isothermal/2C/",
    "input/comsol_results_csv/isothermal/3C/",
    # "input/comsol_results_csv/isothermal/extremely_coarse/1C/",
    # "input/comsol_results_csv/isothermal/extra_coarse/1C/",
    # "input/comsol_results_csv/isothermal/coarser/1C/",
    # "input/comsol_results_csv/isothermal/coarse/1C/",
    # "input/comsol_results_csv/isothermal/normal/1C/",
    # "input/comsol_results_csv/isothermal/fine/1C/",
    # "input/comsol_results_csv/isothermal/finer/1C/",
    # "input/comsol_results_csv/isothermal/extra_fine/1C/",
    # "input/comsol_results_csv/isothermal/extremely_fine/1C/",
]
savefiles = [
    "input/comsol_results/comsol_isothermal_05C.pickle",
    "input/comsol_results/comsol_isothermal_1C.pickle",
    "input/comsol_results/comsol_isothermal_2C.pickle",
    "input/comsol_results/comsol_isothermal_3C.pickle",
    # "input/comsol_results/comsol_isothermal_1C_extremely_coarse.pickle",
    # "input/comsol_results/comsol_isothermal_1C_extra_coarse.pickle",
    # "input/comsol_results/comsol_isothermal_1C_coarser.pickle",
    # "input/comsol_results/comsol_isothermal_1C_coarse.pickle",
    # "input/comsol_results/comsol_isothermal_1C_normal.pickle",
    # "input/comsol_results/comsol_isothermal_1C_fine.pickle",
    # "input/comsol_results/comsol_isothermal_1C_finer.pickle",
    # "input/comsol_results/comsol_isothermal_1C_extra_fine.pickle",
    # "input/comsol_results/comsol_isothermal_1C_extremely_fine.pickle",
]

for path, savefile in zip(paths, savefiles):

    # solution time
    try:
        sol_time = pd.read_csv(path + "solution_time.csv", sep=",", header=None).values[
            0
        ][0]
    except FileNotFoundError:
        sol_time = None

    # time-voltage
    comsol = pd.read_csv(path + "voltage.csv", sep=",", header=None)
    comsol_time = comsol[0].values
    comsol_time_npts = len(comsol_time)
    comsol_voltage = comsol[1].values

    # negative electrode potential
    comsol = pd.read_csv(path + "phi_s_n.csv", sep=",", header=None)
    comsol_x_n_npts = int(len(comsol[0].values) / comsol_time_npts)
    comsol_x_n = comsol[0].values[0:comsol_x_n_npts]
    comsol_phi_n_vals = np.reshape(
        comsol[1].values, (comsol_x_n_npts, comsol_time_npts), order="F"
    )

    # negative electrode solid current
    comsol = pd.read_csv(path + "i_s_n.csv", sep=",", header=None)
    comsol_i_s_n_vals = np.reshape(
        comsol[1].values, (comsol_x_n_npts, comsol_time_npts), order="F"
    )

    # negative electrode electrolyte current
    comsol = pd.read_csv(path + "i_e_n.csv", sep=",", header=None)
    comsol_i_e_n_vals = np.reshape(
        comsol[1].values, (comsol_x_n_npts, comsol_time_npts), order="F"
    )

    # negative particle surface concentration
    comsol = pd.read_csv(path + "c_n_surf.csv", sep=",", header=None)
    comsol_c_n_surf_vals = np.reshape(
        comsol[1].values, (comsol_x_n_npts, comsol_time_npts), order="F"
    )

    # positive electrode potential
    comsol = pd.read_csv(path + "phi_s_p.csv", sep=",", header=None)
    comsol_x_p_npts = int(len(comsol[0].values) / comsol_time_npts)
    comsol_x_p = comsol[0].values[0:comsol_x_p_npts]
    comsol_phi_p_vals = np.reshape(
        comsol[1].values, (comsol_x_p_npts, comsol_time_npts), order="F"
    )

    # positive electrode solid current
    comsol = pd.read_csv(path + "i_s_p.csv", sep=",", header=None)
    comsol_i_s_p_vals = np.reshape(
        comsol[1].values, (comsol_x_p_npts, comsol_time_npts), order="F"
    )

    # positive electrode electrolyte current
    comsol = pd.read_csv(path + "i_e_p.csv", sep=",", header=None)
    comsol_i_e_p_vals = np.reshape(
        comsol[1].values, (comsol_x_p_npts, comsol_time_npts), order="F"
    )

    # positive particle surface concentration
    comsol = pd.read_csv(path + "c_p_surf.csv", sep=",", header=None)
    comsol_c_p_surf_vals = np.reshape(
        comsol[1].values, (comsol_x_p_npts, comsol_time_npts), order="F"
    )

    # electrolyte concentration
    comsol = pd.read_csv(path + "c_e.csv", sep=",", header=None)
    comsol_x_npts = int(len(comsol[0].values) / comsol_time_npts)
    comsol_x = comsol[0].values[0:comsol_x_npts]
    comsol_c_e_vals = np.reshape(
        comsol[1].values, (comsol_x_npts, comsol_time_npts), order="F"
    )

    # electrolyte potential
    comsol = pd.read_csv(path + "phi_e.csv", sep=",", header=None)
    comsol_phi_e_vals = np.reshape(
        comsol[1].values, (comsol_x_npts, comsol_time_npts), order="F"
    )

    # add comsol variables to dict and pickle
    comsol_variables = {
        "time": comsol_time,
        "x_n": comsol_x_n,
        "x_p": comsol_x_p,
        "x": comsol_x,
        "voltage": comsol_voltage,
        "phi_n": comsol_phi_n_vals,
        "phi_p": comsol_phi_p_vals,
        "phi_e": comsol_phi_e_vals,
        "i_s_n": comsol_i_s_n_vals,
        "i_s_p": comsol_i_s_p_vals,
        "i_e_n": comsol_i_e_n_vals,
        "i_e_p": comsol_i_e_p_vals,
        "c_n_surf": comsol_c_n_surf_vals,
        "c_p_surf": comsol_c_p_surf_vals,
        "c_e": comsol_c_e_vals,
        "solution_time": sol_time,
    }

    with open(savefile, "wb") as f:
        pickle.dump(comsol_variables, f, pickle.HIGHEST_PROTOCOL)
