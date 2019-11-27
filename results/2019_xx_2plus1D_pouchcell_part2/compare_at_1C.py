import pybamm
import numpy as np
import matplotlib.pyplot as plt

import models
import plots

pybamm.set_logging_level("INFO")

thermal = True

solve_spm = False
solve_spmecc = False
solve_reduced_2p1 = True
solve_full_2p1 = True

plot_voltage = False
plot_potentials = False
reduced_and_full_potential_errors = False

plot_current = False
plot_av_current = False
plot_reduced_full_current_errors = False

plot_temperature_profile = True
plot_temperature_profile_errors_red_full = False
plot_average_temperature = True

t_eval = np.linspace(0, 0.17, 100)

C_rate = 1

var_pts = {
    pybamm.standard_spatial_vars.x_n: 5,
    pybamm.standard_spatial_vars.x_s: 5,
    pybamm.standard_spatial_vars.x_p: 5,
    pybamm.standard_spatial_vars.y: 5,
    pybamm.standard_spatial_vars.z: 5,
    pybamm.standard_spatial_vars.r_n: 5,
    pybamm.standard_spatial_vars.r_p: 5,
}

params = {
    # "Heat transfer coefficient [W.m-2.K-1]": 0.1,
    # "Negative current collector conductivity [S.m-1]": 5.96e6,
    # "Positive current collector conductivity [S.m-1]": 3.55e6,
    # "Negative current collector conductivity [S.m-1]": 5.96,
    # "Positive current collector conductivity [S.m-1]": 3.55,
}

if solve_spm:
    spm = models.solve_spm(
        C_rate=C_rate, t_eval=t_eval, var_pts=var_pts, thermal=thermal
    )
else:
    spm = None


if solve_spmecc:
    spmecc = models.solve_spmecc(
        t_eval=t_eval, var_pts=var_pts, params=params, C_rate=C_rate, thermal=thermal
    )
else:
    spmecc = None

if solve_reduced_2p1:
    reduced = models.solve_reduced_2p1(
        t_eval=t_eval, var_pts=var_pts, params=params, C_rate=C_rate, thermal=thermal
    )
else:
    reduced = None

if solve_full_2p1:
    full = models.solve_full_2p1(
        t_eval=t_eval, C_rate=C_rate, var_pts=var_pts, thermal=thermal
    )
else:
    full = None

if plot_voltage:
    fig, ax = plt.subplots()
    plots.plot_voltage(ax, spmecc=spmecc, reduced=reduced, full=full)


# times = [0, t_eval[30], t_eval[50], t_eval[70]]
times = [t_eval[50]]
if plot_potentials:
    for t in times:
        plots.plot_yz_potential(t, spmecc=spmecc, reduced=reduced, full=full)

if reduced_and_full_potential_errors:
    for t in times:
        plots.plot_potential_errors(t, reduced=reduced, full=full)

# current
if plot_current:
    for t in times:
        plots.plot_yz_current(t, spmecc=spmecc, reduced=reduced, full=full)

if plot_reduced_full_current_errors:
    for t in times:
        plots.plot_current_errors(t, reduced=reduced, full=full)

if plot_av_current:
    fig, ax = plt.subplots()
    plots.plot_av_cc_current(ax, spm=spm, spmecc=spmecc, reduced=reduced, full=full)

# temperature
if plot_temperature_profile:
    for t in times:
        plots.plot_temperature_profile(t, spmecc=spmecc, reduced=reduced, full=full)

if plot_average_temperature:
    fig, ax = plt.subplots()
    plots.plot_average_temperature(
        ax, spm=spm, spmecc=spmecc, reduced=reduced, full=full
    )

plt.show()

