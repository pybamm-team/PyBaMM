import pybamm
import os
import pandas as pd
import pickle
import numpy as np

# change working directory the root of pybamm
os.chdir(pybamm.root_dir())

# set filepath for data and name of file to pickle to
path = "input/comsol_results_csv/1plus1D/thermal/2C/"
savefile = "input/comsol_results/comsol_thermal_1plus1D_2C.pickle"

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
    "temperature_x": T_z,
    "temperature_z": T_z,
    "temperature": T,
}

with open(savefile, "wb") as f:
    pickle.dump(comsol_variables, f, pickle.HIGHEST_PROTOCOL)
