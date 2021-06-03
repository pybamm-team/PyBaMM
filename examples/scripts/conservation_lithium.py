#
# Check conservation of lithium
#

import pybamm
import matplotlib.pyplot as plt

pybamm.set_logging_level("INFO")

model = pybamm.lithium_ion.DFN()

experiment = pybamm.Experiment(
    [
        "Discharge at 1C until 3.2 V",
        "Rest for 2 hours",
        "Charge at C/3 until 4 V",
        "Charge at 4 V until 5 mA",
        "Rest for 2 hours",
    ]
    * 3
)

sim = pybamm.Simulation(model, experiment=experiment)
sim.solve()
solution = sim.solution

t = solution["Time [s]"].entries
Np = solution["Total lithium in positive electrode [mol]"].entries
Nn = solution["Total lithium in negative electrode [mol]"].entries
Ntot = solution["Total lithium [mol]"].entries

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(t, Ntot / Ntot[0] - 1)
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Variation of total lithium as fraction of initial value")

ax[1].plot(t, Np + Nn, label="total")
ax[1].plot(t, Np, label="positive")
ax[1].plot(t, Nn, label="negative")
ax[1].set_xlabel("Time (s)")
ax[1].set_ylabel("Total lithium in electrode (mol)")
ax[1].legend()

fig.tight_layout()

plt.show()
