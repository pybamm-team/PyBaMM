import pybamm
import numpy as np
import matplotlib.pyplot as plt

import models
import plots

t_eval = np.linspace(0, 0.17, 100)

var_pts = {
    pybamm.standard_spatial_vars.x_n: 5,
    pybamm.standard_spatial_vars.x_s: 5,
    pybamm.standard_spatial_vars.x_p: 5,
    pybamm.standard_spatial_vars.y: 7,
    pybamm.standard_spatial_vars.z: 7,
    pybamm.standard_spatial_vars.r_n: 5,
    pybamm.standard_spatial_vars.r_p: 5,
}

spmecc = models.solve_spmecc(t_eval=t_eval, var_pts=var_pts)

fig, ax = plt.subplots()
plots.plot_voltage(ax, spmecc=spmecc)

plots.plot_yz_potential(spmecc=spmecc)

