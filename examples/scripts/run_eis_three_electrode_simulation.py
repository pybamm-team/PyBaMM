import numpy as np

import pybamm

model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})

frequencies = np.logspace(-4, 4, 50)

eis_sim = pybamm.EISSimulation(
    model,
    three_electrodes=True,
)
eis_sim.solve(frequencies)
eis_sim.nyquist_plot()
