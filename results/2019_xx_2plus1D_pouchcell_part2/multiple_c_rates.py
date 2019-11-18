import numpy as np
import matplotlib.pyplot as plt

import models
import plots

C_rates = [
    0.1,
    0.5,
]
colors = ["b", "g", "r"]
t_eval = np.linspace(0, 0.17, 100)

for i, C_rate in enumerate(C_rates):

    spmecc = models.solve_spmecc(t_eval=t_eval, C_rate=C_rate)
    spm = models.solve_spm(t_eval=t_eval, C_rate=C_rate)

    plots.plot_voltage(
        t_eval,
        spm=spm,
        spmecc=spmecc,
        x_axis="Discharge capacity [A.h]",
        color=colors[i],
    )

plt.show()

