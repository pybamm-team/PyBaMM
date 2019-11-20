import pybamm
import numpy as np
import matplotlib.pyplot as plt

import models
import plots

C_rates = [0.1, 0.5, 1]
# \definecolor{c1}{RGB}{228,26,28}
#  \definecolor{c2}{RGB}{55,126,184}
#  13 \definecolor{c3}{RGB}{77,175,74}
#  14 \definecolor{c4}{RGB}{152,78,163}
#  15 \definecolor{c5}{RGB}{255,127,0}
#  16 \definecolor{c6}{RGB}{106,61,154}


colors = [(0.89, 0.1, 0.1), (0.21, 0.49, 0.72), (0.3, 0.68, 0.68), (0.59, 0.3, 0.64)]
t_eval = np.linspace(0, 0.17, 100)

pybamm.set_logging_level("INFO")

var_pts = {
    pybamm.standard_spatial_vars.x_n: 5,
    pybamm.standard_spatial_vars.x_s: 5,
    pybamm.standard_spatial_vars.x_p: 5,
    pybamm.standard_spatial_vars.y: 7,
    pybamm.standard_spatial_vars.z: 7,
    pybamm.standard_spatial_vars.r_n: 5,
    pybamm.standard_spatial_vars.r_p: 5,
}


fig, ax = plt.subplots()

for i, C_rate in enumerate(C_rates):

    spmecc = models.solve_spmecc(t_eval=t_eval, C_rate=C_rate, var_pts=var_pts)
    spm = models.solve_spm(t_eval=t_eval, C_rate=C_rate, var_pts=var_pts)
    # reduced = models.solve_reduced_2p1(t_eval=t_eval, C_rate=C_rate, var_pts=var_pts)
    reduced = None
    # full = models.solve_full_2p1(t_eval=t_eval, C_rate=C_rate, var_pts=var_pts)
    full = None

    plots.plot_voltage(
        ax,
        spm=spm,
        spmecc=spmecc,
        reduced=reduced,
        full=full,
        x_axis="Discharge capacity [A.h]",
        color=colors[i],
    )

plt.show()

