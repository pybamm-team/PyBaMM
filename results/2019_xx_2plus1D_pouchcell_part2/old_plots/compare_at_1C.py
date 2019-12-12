import pybamm
import numpy as np
import matplotlib.pyplot as plt

import models
import plots

pybamm.set_logging_level("INFO")

# which models?
solve_spm = False
solve_spmecc = True
solve_reduced_2p1 = True
solve_full_2p1 = True

# potential and voltage plots?
plot_voltage = False
plot_potentials = False
spmecc_and_reduced_potential_errors = False
reduced_and_full_potential_errors = False

# current plots?
plot_current = False
plot_av_current = False
plot_reduced_full_current_errors = False

# concentration plots?
x_av_particle_surface_concentration = False
vol_av_particle_surface_concentration = False

plot_yz_average_electrolyte = True


# thermal plots?
thermal = False
plot_temperature_profile = False
plot_temperature_profile_errors_red_full = False
plot_average_temperature = False

t_eval = np.linspace(0, 0.17, 100)

# max 4
plot_times = [0.0001, 0.001, 0.002, 0.017]

C_rate = 1
var_pts = {
    pybamm.standard_spatial_vars.x_n: 15,
    pybamm.standard_spatial_vars.x_s: 5,
    pybamm.standard_spatial_vars.x_p: 15,
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

if thermal is False:
    plot_temperature_profile = False
    plot_temperature_profile_errors_red_full = False
    plot_average_temperature = False


if solve_spm:
    spm = models.solve_spm(
        C_rate=C_rate, t_eval=t_eval, var_pts=var_pts, thermal=thermal, params=params
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


times = [t_eval[50]]
if plot_potentials:
    for t in times:
        plots.plot_yz_potential(t, spmecc=spmecc, reduced=reduced, full=full)

if reduced_and_full_potential_errors:
    for t in times:
        plots.plot_potential_errors(t, reduced=reduced, full=full)

if spmecc_and_reduced_potential_errors:
    for t in times:
        plots.plot_potential_errors(t, spmecc=spmecc, reduced=reduced)

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

if x_av_particle_surface_concentration:
    for t in times:
        plots.plot_x_av_surf_concentration(t, reduced=reduced, full=full)

if vol_av_particle_surface_concentration:
    var_names = [
        "YZ-averaged negative particle surface concentration",
        "YZ-averaged negative particle surface concentration",
    ]
    for var_name in var_names:
        plots.plot_vol_av_particle_concentration(
            ax, var_name, spmecc=spmecc, reduced=reduced, full=full
        )

if plot_yz_average_electrolyte:
    plots.plot_yz_averaged_electrolyte_concentration(
        plot_times, spmecc=spmecc, reduced=reduced, full=full
    )

plt.show()

