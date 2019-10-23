import pybamm
import os
import pandas as pd
import pickle

# change working directory the root of pybamm
os.chdir(pybamm.root_dir())

# pick C_rate and load comsol data
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

# volume-averaged temperature
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/{}C/vol_av_T.csv".format(C_rate),
    sep=",",
    header=None,
)
vol_av_T = comsol[1].values

# electrode-averaged irreverisble heating
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/{}C/irrev_heat_neg.csv".format(C_rate),
    sep=",",
    header=None,
)
Q_irrev_n = comsol[1].values

# separator irreverisble heating
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/{}C/irrev_heat_sep.csv".format(C_rate),
    sep=",",
    header=None,
)
Q_irrev_s = comsol[1].values

# electrode-averaged irreverisble heating
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/{}C/irrev_heat_pos.csv".format(C_rate),
    sep=",",
    header=None,
)
Q_irrev_p = comsol[1].values

# electrode-averaged reverisble heating
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/{}C/rev_heat_neg.csv".format(C_rate),
    sep=",",
    header=None,
)
Q_rev_n = comsol[1].values

# electrode-averaged reverisble heating
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/{}C/rev_heat_pos.csv".format(C_rate),
    sep=",",
    header=None,
)
Q_rev_p = comsol[1].values

# negative current collector potential (stored as a (yz_npts, time_npts) size
# array)
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/{}C/phi_s_cn.csv".format(C_rate),
    sep=",",
    header=None,
)
y = comsol[0].values  # first column y
z = comsol[1].values  # second column z
# y_neg_cc = comsol[0].values  # first column y
# z_neg_cc = comsol[1].values  # second column z
phi_s_cn = comsol.values[:, 3:]  # fourth to end columns var data

# positive current collector potential (stored as a (yz_npts, time_npts) size
# array)
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/{}C/phi_s_cp.csv".format(C_rate),
    sep=",",
    header=None,
)
# y_pos_cc = comsol[0].values  # first column y
# z_pos_cc = comsol[1].values  # second column z
phi_s_cp = comsol.values[:, 3:]  # fourth to end columns var data

# temperature (stored as a (yz_npts, time_npts)  size array)
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/{}C/T.csv".format(C_rate), sep=",", header=None
)
# y_sep = comsol[0].values  # first column y
# z_sep = comsol[1].values  # second column z
T = comsol.values[:, 3:]  # fourth to end columns var data

# current (stored as a (yz_npts, time_npts)  size array)
comsol = pd.read_csv(
    "input/comsol_results_csv/2plus1D/{}C/I.csv".format(C_rate), sep=",", header=None
)
# y_sep = comsol[0].values  # first column y
# z_sep = comsol[1].values  # second column z
I = comsol.values[:, 3:]  # fourth to end columns var data

# add comsol variables to dict and pickle
comsol_variables = {
    "time": time,
    "y": y,
    "z": z,
    # "y_neg_cc": y_neg_cc,
    # "z_neg_cc": z_neg_cc,
    # "y_pos_cc": y_pos_cc,
    # "z_pos_cc": z_pos_cc,
    # "y_sep": y_sep,
    # "z_sep": z_sep,
    "voltage": voltage,
    "volume-averaged temperature": vol_av_T,
    "phi_s_cn": phi_s_cn,
    "phi_s_cp": phi_s_cp,
    "temperature": T,
    "current": I,
    "averaged Q_irrev_n": Q_irrev_n,
    "averaged Q_irrev_s": Q_irrev_s,
    "averaged Q_irrev_p": Q_irrev_p,
    "averaged Q_rev_n": Q_rev_n,
    "averaged Q_rev_p": Q_rev_p,
}

savefile = "comsol_{}C.pickle".format(C_rate)
with open(savefile, "wb") as f:
    pickle.dump(comsol_variables, f, pickle.HIGHEST_PROTOCOL)
