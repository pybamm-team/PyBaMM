import pybamm
import numpy as np
import matplotlib.pyplot as plt

import models
import plots

pybamm.set_logging_level("INFO")

t_eval = np.linspace(0, 0.17, 100)

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

C_rate = 0.5

spmecc = models.solve_spmecc(
    t_eval=t_eval, var_pts=var_pts, params=params, C_rate=C_rate
)
reduced = models.solve_reduced_2p1(
    t_eval=t_eval, var_pts=var_pts, params=params, C_rate=C_rate
)
# reduced = None

fig, ax = plt.subplots()
plots.plot_voltage(ax, spmecc=spmecc, reduced=reduced)

times = [0, t_eval[30], t_eval[50], t_eval[70]]
for t in times:
    plots.plot_yz_potential(t, spmecc=spmecc, reduced=reduced)

plt.show()

