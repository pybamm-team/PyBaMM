import numpy as np

import models
import plots

t_eval = np.linspace(0, 0.17, 100)
spmecc = models.solve_spmecc(t_eval=t_eval)
plots.plot_voltage(t_eval, spmecc=spmecc)

