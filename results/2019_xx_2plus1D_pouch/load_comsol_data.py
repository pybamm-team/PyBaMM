import pybamm
import os
import pandas as pd
import pickle
import numpy as np

# change working directory the root of pybamm
os.chdir(pybamm.root_dir())

# pick C_rate and load comsol data
C_rates = {"1": 1}
C_rate = "1"  # choose the key from the above dictionary of available results

# time-voltage (both just 1D arrays)
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/{}C/voltage.csv".format(C_rate),
    sep=",",
    header=None,
)
time = comsol[0].values
time_npts = len(time)
voltage = comsol[1].values

# negative current collector potential (stored as a (yz_npts, time_npts) size
# array)
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/{}C/phi_s_cn.csv".format(C_rate),
    sep=",",
    header=None,
)
yz_npts = int(len(comsol[0].values) / time_npts)  # get length of y-z pts
z_neg_cc = comsol[0].values[::time_npts]  # first column z
y_neg_cc = comsol[1].values[::time_npts]  # second column y
phi_s_cn = np.reshape(
    comsol[3].values, (yz_npts, time_npts), order="C"
)  # fourth column is phi_s_cn vals

## test interp onto regular grid
# y = np.linspace(0, np.max(y_neg_cc), 5)
# z = np.linspace(0, np.max(z_neg_cc), 4)
# grid_y, grid_z = np.meshgrid(y, z)
# interp_var = np.zeros((len(z), len(y), len(time)))
# for i in range(0, phi_s_cn.shape[1]):
#    interp_var[:, :, i] = interp.griddata(
#        np.column_stack((y_neg_cc, z_neg_cc)), phi_s_cn[:, i],(grid_y, grid_z), method="cubic")
#
#
# def myinterp(t):
#    return interp.interp1d(time, interp_var, axis=2)(t)
#
#
# plt.pcolormesh(y,z,myinterp(2),shading="gouraud")
# plt.show()

# positive current collector potential (stored as a (yz_npts, time_npts) size
# array)
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/{}C/phi_s_cp.csv".format(C_rate),
    sep=",",
    header=None,
)
yz_npts = int(len(comsol[0].values) / time_npts)  # get length of y-z pts
z_pos_cc = comsol[0].values[::time_npts]  # first column z
y_pos_cc = comsol[1].values[::time_npts]  # second column y
phi_s_cp = np.reshape(
    comsol[3].values, (yz_npts, time_npts), order="C"
)  # fourth column is phi_s_cp vals

# temperature (evaluated on separator nodes) (stored as a (yz_npts, time_npts)
# size array)
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/{}C/T.csv".format(C_rate), sep=",", header=None
)
yz_npts = int(len(comsol[0].values) / time_npts)  # get length of y-z pts
z_sep = comsol[0].values[::time_npts]  # first column z
y_sep = comsol[1].values[::time_npts]  # second column y
T = np.reshape(
    comsol[3].values, (yz_npts, time_npts), order="C"
)  # fourth column is T vals

# add comsol variables to dict and pickle
comsol_variables = {
    "time": time,
    "voltage": voltage,
    "y_neg_cc": y_neg_cc,
    "z_neg_cc": z_neg_cc,
    "phi_s_cn": phi_s_cn,
    "y_pos_cc": y_pos_cc,
    "z_pos_cc": z_pos_cc,
    "phi_s_cp": phi_s_cp,
    "y_sep": y_sep,
    "z_sep": z_sep,
    "temperature": T,
}

savefile = "comsol_{}C.pickle".format(C_rate)
with open(savefile, "wb") as f:
    pickle.dump(comsol_variables, f, pickle.HIGHEST_PROTOCOL)
