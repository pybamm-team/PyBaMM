import pybamm
import os
import pandas as pd
import pickle
import numpy as np

# change working directory the root of pybamm
os.chdir(pybamm.root_dir())

# set filepaths for data and names of file to pickle to
paths = [
    "input/comsol_results_csv/1plus1D/thermal/05C/",
    "input/comsol_results_csv/1plus1D/thermal/1C/",
    "input/comsol_results_csv/1plus1D/thermal/2C/",
    "input/comsol_results_csv/1plus1D/thermal/3C/",
    "input/comsol_results_csv/1plus1D/thermal/extra_coarse/1C/",
    "input/comsol_results_csv/1plus1D/thermal/coarse/1C/",
    "input/comsol_results_csv/1plus1D/thermal/normal/1C/",
    "input/comsol_results_csv/1plus1D/thermal/fine/1C/",
]
savefiles = [
    "input/comsol_results/comsol_thermal_1plus1D_05C.pickle",
    "input/comsol_results/comsol_thermal_1plus1D_1C.pickle",
    "input/comsol_results/comsol_thermal_1plus1D_2C.pickle",
    "input/comsol_results/comsol_thermal_1plus1D_3C.pickle",
    "input/comsol_results/comsol_thermal_1plus1D_1C_extra_coarse.pickle",
    "input/comsol_results/comsol_thermal_1plus1D_1C_coarse.pickle",
    "input/comsol_results/comsol_thermal_1plus1D_1C_normal.pickle",
    "input/comsol_results/comsol_thermal_1plus1D_1C_fine.pickle",
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

    # negative current collector potential (stored as a (xz_npts, time_npts) size
    # array)
    comsol = pd.read_csv(path + "phi_s_cn.csv", sep=",", header=None)
    phi_s_cn_x = comsol[0].values  # first column x
    phi_s_cn_z = comsol[1].values  # second column z
    phi_s_cn = comsol.values[:, 2:]  # third to end columns var data

    # positive current collector potential (stored as a (xz_npts, time_npts) size
    # array)
    comsol = pd.read_csv(path + "phi_s_cp.csv", sep=",", header=None)
    phi_s_cp_x = comsol[0].values  # first column x
    phi_s_cp_z = comsol[1].values  # second column z
    phi_s_cp = comsol.values[:, 2:]  # third to end columns var data

    # current (stored as a (xz_npts, time_npts)  size array)
    comsol = pd.read_csv(path + "current.csv", sep=",", header=None)
    I_x = comsol[0].values  # first column x
    I_z = comsol[1].values  # second column z
    I = comsol.values[:, 2:]  # third to end columns var data

    # temperature (stored as a (xz_npts, time_npts)  size array)
    comsol = pd.read_csv(path + "temperature.csv", sep=",", header=None)
    T_x = comsol[0].values  # first column x
    T_z = comsol[1].values  # second column z
    T = comsol.values[:, 2:]  # third to end columns var data
    vol_av_T = np.mean(T, axis=0)

    # add comsol variables to dict and pickle
    comsol_variables = {
        "time": time,
        "voltage": voltage,
        "volume-averaged temperature": vol_av_T,
        "phi_s_cn_x": phi_s_cn_x,
        "phi_s_cn_z": phi_s_cn_z,
        "phi_s_cn": phi_s_cn,
        "phi_s_cp_x": phi_s_cp_x,
        "phi_s_cp_z": phi_s_cp_z,
        "phi_s_cp": phi_s_cp,
        "current_x": I_x,
        "current_z": I_z,
        "current": I,
        "temperature_x": T_x,
        "temperature_z": T_z,
        "temperature": T,
        "solution_time": sol_time,
    }

    # add concentrations if provided
    try:
        # c_s_n_surf_av (stored as a (xz_npts, time_npts)  size array)
        comsol = pd.read_csv(path + "c_s_n.csv", sep=",", header=None)
        c_s_n_x = comsol[0].values  # first column x
        c_s_n_z = comsol[1].values  # second column z
        c_s_n = comsol.values[:, 2:]  # third to end columns var data

        # c_s_p_surf_av (stored as a (xz_npts, time_npts)  size array)
        comsol = pd.read_csv(path + "c_s_p.csv", sep=",", header=None)
        c_s_p_x = comsol[0].values  # first column x
        c_s_p_z = comsol[1].values  # second column z
        c_s_p = comsol.values[:, 2:]  # third to end columns var data

        comsol_variables.update({
            "c_s_n_x": c_s_n_x,
            "c_s_n_z": c_s_n_z,
            "c_s_n": c_s_n,
            "c_s_p_x": c_s_p_x,
            "c_s_p_z": c_s_p_z,
            "c_s_p": c_s_p,
        })
    except FileNotFoundError:
        print("no concentration data for {}".format(path))

    with open(savefile, "wb") as f:
        pickle.dump(comsol_variables, f, pickle.HIGHEST_PROTOCOL)
