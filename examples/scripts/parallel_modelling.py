# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:14:16 2020

@author: tom
"""
import pybamm
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import time as ticker_timer
pybamm.set_logging_level(50)

def battery_sim_parallel(l_n):
    factor = 5.0
    l_p = 1e-4
    # Dimensions
    h = 0.137
    w = 0.207 / factor
    l_s = 2.5e-5
    l1d = l_n + l_p + l_s
    vol = h * w * l1d
    vol_cm3 = vol * 1e6
    tot_cap = 0.0
    tot_time = 0.0

    I_mag = 1.00 / factor
    for enum, I_app in enumerate([-1.0, 1.0]):
        I_app *= I_mag
        # load model
        model = pybamm.lithium_ion.SPMe()
        # create geometry
        geometry = model.default_geometry
        # load parameter values and process model and geometry
        param = model.default_parameter_values
        param.update(
            {
                "Electrode height [m]": h,
                "Electrode width [m]": w,
                "Negative electrode thickness [m]": l_n,
                "Positive electrode thickness [m]": l_p,
                "Separator thickness [m]": l_s,
                "Lower voltage cut-off [V]": 2.8,
                "Upper voltage cut-off [V]": 4.7,
                "Maximum concentration in negative electrode [mol.m-3]": 25000,
                "Maximum concentration in positive electrode [mol.m-3]": 50000,
                "Initial concentration in negative electrode [mol.m-3]": 12500,
                "Initial concentration in positive electrode [mol.m-3]": 25000,
                "Negative electrode surface area to volume ratio [m-1]": 180000.0,
                "Positive electrode surface area to volume ratio [m-1]": 150000.0,
                "Current function [A]": I_app,
            }
        )
        param.process_model(model)
        param.process_geometry(geometry)
        s_var = pybamm.standard_spatial_vars
        var_pts = {
            s_var.x_n: 5,
            s_var.x_s: 5,
            s_var.x_p: 5,
            s_var.r_n: 5,
            s_var.r_p: 10,
        }
        # set mesh
        mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
        # discretise model
        disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
        disc.process_model(model)
        # solve model
        t_eval = np.linspace(0, 3600, 100)
        sol = pybamm.CasadiSolver(mode='safe').solve(model, t_eval)
        time = sol["Time [h]"]
        # Coulomb counting
        time_secs = sol["Time [s]"].entries
        time_hours = time(time_secs)
        dc_time = np.around(time_hours[-1], 3)
        # Capacity mAh
        cap = np.absolute(I_app * 1000 * dc_time)
        tot_cap += cap
        tot_time += dc_time

    specific_cap = np.around(tot_cap, 3) / vol_cm3
    return [tot_cap, specific_cap]

if __name__ == '__main__':
    l_p = 1e-4
    thicknesses = np.linspace(1.0, 2.5, 11) * l_p
    ex = ProcessPoolExecutor()
    st = ticker_timer.time()
    cap_parallel = list(ex.map(battery_sim_parallel, thicknesses.tolist()))
    et = ticker_timer.time()
    print('Simulation Time', np.around(et-st, 3))
    cap_parallel = np.asarray(cap_parallel)
    ex.shutdown(wait=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax1.plot(thicknesses / l_p, cap_parallel[:, 0])
    ax2.plot(thicknesses / l_p, cap_parallel[:, 1])
    ax1.set_ylabel("Capacity [mAh]")
    ax2.set_ylabel("Specific Capacity [mAh.cm-3]")
    ax2.set_xlabel("Anode : Cathode thickness ratio")