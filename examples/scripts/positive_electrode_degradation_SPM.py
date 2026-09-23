import matplotlib.pyplot as plt
import numpy as np

import pybamm

pybamm.settings.tolerances["reg_power"] = 0

model = pybamm.lithium_ion.SPM({"positive electrode degradation": "true"})
param = pybamm.ParameterValues("Zhuo2023")

experiment = pybamm.Experiment(
    [
        (
            "Charge at 0.5 C until 4.2 V",
            "Hold at 4.2 V until C/50",
            "Rest for 60 minutes",
            "Discharge at 0.5 C until 2.8 V",
            "Hold at 2.8 V until C/50",
            "Rest for 60 minutes",
        )
    ]
    * 20,
    period="0.5 minute",
)

sim = pybamm.Simulation(model, experiment=experiment, parameter_values=param)
solution = sim.solve(calc_esoh=False)

sim.plot(
    [
        "Current [A]",
        "Terminal voltage [V]",
        "X-averaged moving phase boundary location",
        [
            "X-averaged positive core surface stoichiometry",
            "X-averaged negative particle surface concentration",
        ],
        "X-averaged positive particle shell volume fraction",
        "Discharge capacity [A.h]",
        [
            "Total cyclable lithium in positive electrode [mol]",
            "Total cyclable lithium in negative electrode [mol]",
            "Total cyclable lithium in particles [mol]",
        ],
        "Loss of cyclable lithium inventory",
    ]
)

discharge_capacities = []
for cycle in solution.cycles:
    discharge = cycle.steps[3]
    capacity = discharge["Discharge capacity [A.h]"]
    discharge_capacities.append(
        float(capacity(discharge.t[-1]) - capacity(discharge.t[0]))
    )

plt.figure(figsize=(8, 6))
plt.plot(
    np.arange(1, len(discharge_capacities) + 1),
    discharge_capacities,
    "o",
    mfc="none",
    label="LAM and LLI",
)
plt.xlabel("Cycle number")
plt.ylabel("Discharge capacity [A.h]")
plt.legend()
plt.show()
