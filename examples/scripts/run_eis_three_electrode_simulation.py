import numpy as np

import pybamm

model = pybamm.lithium_ion.SPM(options={"surface form": "differential"})

# Per-electrode impedances need a reference electrode; the default position is
# the mid-point of the separator.
model.insert_reference_electrode()

frequencies = np.logspace(-4, 4, 50)

eis_sim = pybamm.EISSimulation(model)
solution = eis_sim.solve(frequencies)

# The electrode impedances sum to the cell impedance.
z_cell = solution["Cell impedance [Ohm]"]
z_positive = solution["Positive electrode impedance [Ohm]"]
z_negative = solution["Negative electrode impedance [Ohm]"]

solution.nyquist_plot()
