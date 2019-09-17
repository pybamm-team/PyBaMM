import pybamm
import os
import pandas as pd
import pickle

# change working directory the root of pybamm
os.chdir(pybamm.root_dir())

"-----------------------------------------------------------------------------"
"Pick C_rate and load comsol data"

# C_rate
C_rates = {"01": 0.1, "05": 0.5, "1": 1, "2": 2, "3": 3}
C_rate = "1"  # choose the key from the above dictionary of available results

# time-voltage
comsol = pd.read_csv(
    "input/data/comsol/{}C/Voltage.csv".format(C_rate), sep=",", header=None
)
comsol_time = comsol[0].values
comsol_time_npts = len(comsol_time)
comsol_voltage = comsol[1].values

# negative current collector potential

# positive current collector potential

# x-averaged temperature

# add comsol variables to dict and pickle
comsol_variables = {"time": comsol_time, "voltage": comsol_voltage}
pickle.dump(
    comsol_variables,
    open("input/comsol_results/2plus1D/comsol_{}C.pickle".format(C_rate), "wb")
)
## negative electrode potential
#comsol = pd.read_csv(
#    "input/data/comsol/{}C/phi_n.csv".format(C_rate), sep=",", header=None
#)
#comsol_x_n_npts = int(len(comsol[0].values) / comsol_time_npts)
#comsol_x_n = comsol[0].values[0:comsol_x_n_npts]
#comsol_phi_n_vals = np.reshape(
#    comsol[1].values, (comsol_x_n_npts, comsol_time_npts), order="F"
#)
#
## negative particle surface concentration
#comsol = pd.read_csv(
#    "input/data/comsol/{}C/c_n_surf.csv".format(C_rate), sep=",", header=None
#)
#comsol_c_n_surf_vals = np.reshape(
#    comsol[1].values, (comsol_x_n_npts, comsol_time_npts), order="F"
#)
#
## positive electrode potential
#comsol = pd.read_csv(
#    "input/data/comsol/{}C/phi_p.csv".format(C_rate), sep=",", header=None
#)
#comsol_x_p_npts = int(len(comsol[0].values) / comsol_time_npts)
#comsol_x_p = comsol[0].values[0:comsol_x_p_npts]
#comsol_phi_p_vals = np.reshape(
#    comsol[1].values, (comsol_x_p_npts, comsol_time_npts), order="F"
#)
#
## positive particle surface concentration
#comsol = pd.read_csv(
#    "input/data/comsol/{}C/c_p_surf.csv".format(C_rate), sep=",", header=None
#)
#comsol_c_p_surf_vals = np.reshape(
#    comsol[1].values, (comsol_x_p_npts, comsol_time_npts), order="F"
#)
#
## electrolyte concentration
#comsol = pd.read_csv(
#    "input/data/comsol/{}C/c_e.csv".format(C_rate), sep=",", header=None
#)
#comsol_x_npts = int(len(comsol[0].values) / comsol_time_npts)
#comsol_x = comsol[0].values[0:comsol_x_npts]
#comsol_c_e_vals = np.reshape(
#    comsol[1].values, (comsol_x_npts, comsol_time_npts), order="F"
#)
#
## electrolyte potential
#comsol = pd.read_csv(
#    "input/data/comsol/{}C/phi_e.csv".format(C_rate), sep=",", header=None
#)
#comsol_phi_e_vals = np.reshape(
#    comsol[1].values, (comsol_x_npts, comsol_time_npts), order="F"
#)
#
