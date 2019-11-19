import pybamm
import os
import pandas as pd
import pickle
import numpy as np

# change working directory the root of pybamm
os.chdir(pybamm.root_dir())

# set filepath for data and name of file to pickle to
path = "input/comsol_results_csv/2plus1D/isothermal/1C/"
savefile = "input/comsol_results/comsol_isothermal_2plus1D_1C.pickle"

# time-voltage (both just 1D arrays)
comsol = pd.read_csv(path + "voltage.csv", sep=",", header=None)
time = comsol[0].values
time_npts = len(time)
voltage = comsol[1].values

# negative current collector potential (stored as a (yz_npts, time_npts) size
# array)
comsol = pd.read_csv(path + "phi_s_cn.csv", sep=",", header=None)
phi_s_cn_y = comsol[0].values  # first column y
phi_s_cn_z = comsol[1].values  # second column z
phi_s_cn = comsol.values[:, 3:]  # fourth to end columns var data

# positive current collector potential (stored as a (yz_npts, time_npts) size
# array)
comsol = pd.read_csv(path + "phi_s_cp.csv", sep=",", header=None)
phi_s_cp_y = comsol[0].values  # first column y
phi_s_cp_z = comsol[1].values  # second column z
phi_s_cp = comsol.values[:, 3:]  # fourth to end columns var data

# current (stored as a (yz_npts, time_npts)  size array)
comsol = pd.read_csv(path + "I.csv", sep=",", header=None)
I_y = comsol[0].values  # first column y
I_z = comsol[1].values  # second column z
I = comsol.values[:, 3:]  # fourth to end columns var data

# add comsol variables to dict and pickle
comsol_variables = {
    "time": time,
    "voltage": voltage,
    "phi_s_cn_y": phi_s_cn_y,
    "phi_s_cn_z": phi_s_cn_z,
    "phi_s_cn": phi_s_cn,
    "phi_s_cp_y": phi_s_cp_y,
    "phi_s_cp_z": phi_s_cp_z,
    "phi_s_cp": phi_s_cp,
    "current_y": I_y,
    "current_z": I_z,
    "current": I,
}

with open(savefile, "wb") as f:
    pickle.dump(comsol_variables, f, pickle.HIGHEST_PROTOCOL)
