import pybamm
import os
import pandas as pd
import pickle
import numpy as np

# change working directory the root of pybamm
os.chdir(pybamm.root_dir())

# set filepath for data and name of file to pickle to
path = "input/comsol_results_csv/2plus1D/1C/"
savefile = "input/comsol_results/comsol_2plus1D_1C.pickle"

# time-voltage (both just 1D arrays)
comsol = pd.read_csv(path + "voltage.csv",
    sep=",",
    header=None,
)
time = comsol[0].values
time_npts = len(time)
voltage = comsol[1].values

# negative current collector potential (stored as a (yz_npts, time_npts) size
# array)
comsol = pd.read_csv(
    path + "phi_s_cn.csv",
    sep=",",
    header=None,
)
y = comsol[0].values  # first column y
z = comsol[1].values  # second column z
phi_s_cn = comsol.values[:, 3:]  # fourth to end columns var data

# positive current collector potential (stored as a (yz_npts, time_npts) size
# array)
comsol = pd.read_csv(
    path + "phi_s_cp.csv",
    sep=",",
    header=None,
)
phi_s_cp = comsol.values[:, 3:]  # fourth to end columns var data

# temperature (stored as a (yz_npts, time_npts)  size array)
comsol = pd.read_csv(
    path + "T.csv", sep=",", header=None
)
T = comsol.values[:, 3:]  # fourth to end columns var data
vol_av_T = np.mean(T, axis=0)

# current (stored as a (yz_npts, time_npts)  size array)
comsol = pd.read_csv(
    path + "I.csv", sep=",", header=None
)
I = comsol.values[:, 3:]  # fourth to end columns var data

# add comsol variables to dict and pickle
comsol_variables = {
    "time": time,
    "y": y,
    "z": z,
    "voltage": voltage,
    "volume-averaged temperature": vol_av_T,
    "phi_s_cn": phi_s_cn,
    "phi_s_cp": phi_s_cp,
    "temperature": T,
    "current": I,
}

with open(savefile, "wb") as f:
    pickle.dump(comsol_variables, f, pickle.HIGHEST_PROTOCOL)
