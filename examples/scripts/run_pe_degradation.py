"""
    This tutorial runs the core-shell model for phase transition caused PE 
    (NMC811) degradation, based on the paper
    -- Mingzhao Zhuo, Gregory Offer, Monica Marinescu, "Degradation model of
       high-nickel positive electrodes: Effects of loss of active material and
       cyclable lithium on capacity fade", Journal of Power Sources,
       556 (2023): 232461. doi: 10.1016/j.jpowsour.2022.232461.
    The following scripts reproduce the results presented in Figs. 5, 7, 9 etc.
    c_s: Trapped lithium concentration in the shell [mol.m-3]
    rho: Positive electrode shell resistivity [Ohm.m]
    Fig. 5: c_s = 14802, rho = 0
    Fig. 7: c_s = 20000, rho = 0
    Fig. 9: c_s = 20000, rho = 1e6
"""
import pybamm
import numpy as np
import matplotlib.pyplot as plt

#%%
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
    ] * 20,
)
sim = pybamm.Simulation(
    model, experiment=experiment, 
    parameter_values=param,
    # solver=pybamm.CasadiSolver("fast with events"),
)
#%%
solution = sim.solve(calc_esoh=False)
#%%
output_variables = [
    "X-averaged positive particle moving phase boundary location",
    "X-averaged positive electrode shell layer overpotential [V]",
    # "X-averaged loss of active material due to PE phase transition",
    "LLI_cyc [%]",
    "Cell SoC",
    [
        "Total cyclable lithium in primary phase in positive electrode [mol]", 
        "Total cyclable lithium in primary phase in negative electrode [mol]",
        "Total cyclable lithium in particles [mol]",
    ],
    [
        "X-averaged positive core surface lithium concentration",
        "X-averaged negative particle surface concentration",
    ],
    # "Current [A]",
    # "Terminal voltage [V]",
    # "X-averaged negative particle concentration [mol.m-3]",
    # "X-averaged positive core lithium concentration [mol.m-3]",
    # "X-averaged positive shell oxygen concentration [mol.m-3]",
    # "LLI [%]",
]
sim.plot(output_variables, time_unit="hours")
