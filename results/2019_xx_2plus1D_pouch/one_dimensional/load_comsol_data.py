import pybamm
import os
import pandas as pd
import pickle
import numpy as np

# change working directory the root of pybamm
os.chdir(pybamm.root_dir())

# set filepath for data and name of file to pickle to
path = "input/comsol_results_csv/thermal/1C/"
savefile = "input/comsol_results/comsol_thermal_1C.pickle"

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

# temperature
comsol = pd.read_csv(path + "temperature.csv", sep=",", header=None)
comsol_temp_vals = np.reshape(
    comsol[1].values, (comsol_x_npts, comsol_time_npts), order="F"
)

# average temperature
comsol_temp_av = np.mean(comsol_temp_vals, axis=0)

# irreversible heat source in negative electrode
comsol = pd.read_csv(path + "q_irrev_n.csv", sep=",", header=None)
comsol_q_irrev_n_vals = np.reshape(
    comsol[1].values, (comsol_x_n_npts, comsol_time_npts), order="F"
)

# irreversible heat source in positive electrode
comsol = pd.read_csv(path + "q_irrev_p.csv", sep=",", header=None)
comsol_q_irrev_p_vals = np.reshape(
    comsol[1].values, (comsol_x_p_npts, comsol_time_npts), order="F"
)

# reversible heat source in negative electrode
comsol = pd.read_csv(path + "q_rev_n.csv", sep=",", header=None)
comsol_q_rev_n_vals = np.reshape(
    comsol[1].values, (comsol_x_n_npts, comsol_time_npts), order="F"
)

# reversible heat source in positive electrode
comsol = pd.read_csv(path + "q_rev_p.csv", sep=",", header=None)
comsol_q_rev_p_vals = np.reshape(
    comsol[1].values, (comsol_x_p_npts, comsol_time_npts), order="F"
)

# total heat source in negative electrode
comsol = pd.read_csv(path + "q_total_n.csv", sep=",", header=None)
comsol_q_total_n_vals = np.reshape(
    comsol[1].values, (comsol_x_n_npts, comsol_time_npts), order="F"
)

# total heat source in separator
comsol = pd.read_csv(path + "q_total_s.csv", sep=",", header=None)
comsol_x_s_npts = int(len(comsol[0].values) / comsol_time_npts)
comsol_x_s = comsol[0].values[0:comsol_x_s_npts]
comsol_q_total_s_vals = np.reshape(
    comsol[1].values, (comsol_x_s_npts, comsol_time_npts), order="F"
)

# total heat source in positive electrode
comsol = pd.read_csv(path + "q_total_p.csv", sep=",", header=None)
comsol_q_total_p_vals = np.reshape(
    comsol[1].values, (comsol_x_p_npts, comsol_time_npts), order="F"
)


# add comsol variables to dict and pickle
comsol_variables = {
    "time": comsol_time,
    "x_n": comsol_x_n,
    "x_s": comsol_x_s,
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
    "temperature": comsol_temp_vals,
    "average temperature": comsol_temp_av,
    "Q_irrev_n": comsol_q_irrev_n_vals,
    "Q_irrev_p": comsol_q_irrev_p_vals,
    "Q_rev_n": comsol_q_rev_n_vals,
    "Q_rev_p": comsol_q_rev_p_vals,
    "Q_total_n": comsol_q_total_n_vals,
    "Q_total_s": comsol_q_total_s_vals,
    "Q_total_p": comsol_q_total_p_vals,
}

with open(savefile, "wb") as f:
    pickle.dump(comsol_variables, f, pickle.HIGHEST_PROTOCOL)
