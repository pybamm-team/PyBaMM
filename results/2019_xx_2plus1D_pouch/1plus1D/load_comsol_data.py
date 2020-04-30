import pybamm
import os
import pandas as pd
import pickle
import numpy as np

# change working directory the root of pybamm
os.chdir(pybamm.root_dir())

# set filepaths for data and names of file to pickle to
paths = [
    "input/new_comsol_results_csv/1plus1D/1C/",
    "input/new_comsol_results_csv/1plus1D/3C/",
    "input/new_comsol_results_csv/1plus1D/sigma_1e5/",
    "input/new_comsol_results_csv/1plus1D/sigma_1e6/",
    "input/new_comsol_results_csv/1plus1D/sigma_1e7/",
    "input/new_comsol_results_csv/1plus1D/sigma_1e8/",
    "input/new_comsol_results_csv/1plus1D/sigma_1e9/",
]
savefiles = [
    "input/comsol_results/comsol_1plus1D_1C.pickle",
    "input/comsol_results/comsol_1plus1D_3C.pickle",
    "input/comsol_results/comsol_1plus1D_sigma_1e5.pickle",
    "input/comsol_results/comsol_1plus1D_sigma_1e6.pickle",
    "input/comsol_results/comsol_1plus1D_sigma_1e7.pickle",
    "input/comsol_results/comsol_1plus1D_sigma_1e8.pickle",
    "input/comsol_results/comsol_1plus1D_sigma_1e9.pickle",
]

for path, savefile in zip(paths, savefiles):

    # solution time
    try:
        sol_time = pd.read_csv(path + "solution_time.csv", sep=",", header=None).values[
            0
        ][0]
    except FileNotFoundError:
        sol_time = None

    # time-voltage (both just 1D arrays)
    comsol = pd.read_csv(path + "voltage.csv", sep=",", header=None)
    time = comsol[0].values
    time_npts = len(time)
    voltage = comsol[1].values

    # negative current collector potential (stored as a (z_npts, time_npts) size
    # array)
    comsol = pd.read_csv(path + "phi_s_cn.csv", sep=",", header=None)
    phi_s_cn_z = comsol[0].values  # first column z
    phi_s_cn = comsol.values[:, 1:]  # second to end columns var data

    # positive current collector potential (stored as a (z_npts, time_npts) size
    # array)
    comsol = pd.read_csv(path + "phi_s_cp.csv", sep=",", header=None)
    phi_s_cp_z = comsol[0].values  # first column z
    phi_s_cp = comsol.values[:, 1:]  # second to end columns var data

    # current (stored as a (z_npts, time_npts)  size array)
    comsol = pd.read_csv(path + "current.csv", sep=",", header=None)
    I_z = comsol[0].values  # first column z
    I = comsol.values[:, 1:]  # second to end columns var data

    # temperature (stored as a (z_npts, time_npts)  size array)
    comsol = pd.read_csv(path + "temperature.csv", sep=",", header=None)
    T_z = comsol[0].values  # first column z
    T = comsol.values[:, 1:]  # second to end columns var data
    vol_av_T = np.mean(T, axis=0)

    # add comsol variables to dict and pickle
    comsol_variables = {
        "time": time,
        "voltage": voltage,
        "volume-averaged temperature": vol_av_T,
        "phi_s_cn_z": phi_s_cn_z,
        "phi_s_cn": phi_s_cn,
        "phi_s_cp_z": phi_s_cp_z,
        "phi_s_cp": phi_s_cp,
        "current_z": I_z,
        "current": I,
        "temperature_z": T_z,
        "temperature": T,
        "solution_time": sol_time,
    }

    # add concentrations if provided
    try:
        # c_s_n_surf_av (stored as a (z_npts, time_npts)  size array)
        comsol = pd.read_csv(path + "c_s_n.csv", sep=",", header=None)
        c_s_n_z = comsol[0].values  # second column x
        c_s_n = comsol.values[:, 1:]  # second to end columns var data

        # c_s_p_surf_av (stored as a (z_npts, time_npts)  size array)
        comsol = pd.read_csv(path + "c_s_p.csv", sep=",", header=None)
        c_s_p_z = comsol[0].values  # first column z
        c_s_p = comsol.values[:, 1:]  # second to end columns var data

        comsol_variables.update(
            {"c_s_n_z": c_s_n_z, "c_s_n": c_s_n, "c_s_p_z": c_s_p_z, "c_s_p": c_s_p}
        )
    except FileNotFoundError:
        print("No concentration data for {}".format(path))

    with open(savefile, "wb") as f:
        pickle.dump(comsol_variables, f, pickle.HIGHEST_PROTOCOL)
