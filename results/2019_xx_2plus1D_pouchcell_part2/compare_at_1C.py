import numpy as np
import matplotlib.pyplot as plt

import models
import plots

t_eval = np.linspace(0, 0.17, 100)
spmecc = models.solve_spmecc(t_eval=t_eval)

fig, ax = plt.subplots()
plots.plot_voltage(ax, spmecc=spmecc)

plots.plot_yz_potential(spmecc=spmecc)

