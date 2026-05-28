import numpy as np

import pybamm

model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})

eis_sim = pybamm.EISSimulation(model)
frequencies = np.logspace(-4, 4, 30)
eis_sim.solve(frequencies)
eis_sim.nyquist_plot()
