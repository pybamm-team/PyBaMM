"""
Cycle the core-shell positive electrode degradation model of Zhuo et al. (2023),
doi: 10.1016/j.jpowsour.2022.232461. Figs. 5, 7 and 9 of the paper use a trapped
lithium concentration c_s and shell resistivity rho of (14802, 0), (20000, 0) and
(20000, 1e6).
"""

import pybamm

# %%
model = pybamm.lithium_ion.SPM({"PE degradation": "phase transition"})
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
)
sim = pybamm.Simulation(
    model,
    experiment=experiment,
    parameter_values=param,
)
# %%
solution = sim.solve(calc_esoh=False)
# %%
output_variables = [
    "X-averaged positive particle moving phase boundary location",
    "X-averaged positive electrode shell layer overpotential [V]",
    # "X-averaged loss of active material due to PE phase transition",
    "LLI_cyc [%]",
    "Discharge capacity [A.h]",
    [
        "Total cyclable lithium in primary phase in positive electrode [mol]",
        "Total cyclable lithium in primary phase in negative electrode [mol]",
        "Total cyclable lithium in particles [mol]",
    ],
    [
        "X-averaged positive core surface lithium concentration",
        "X-averaged negative particle surface concentration",
    ],
]
sim.plot(output_variables, time_unit="hours")
