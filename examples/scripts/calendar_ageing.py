import pybamm as pb
import numpy as np

pb.set_logging_level("INFO")

models = [
    pb.lithium_ion.SPM({"sei": "reaction limited"}),
    pb.lithium_ion.SPM(
        {"sei": "reaction limited", "surface form": "algebraic"}, name="Algebraic SPM",
    ),
    pb.lithium_ion.DFN({"sei": "reaction limited"}),
]

sims = []
for model in models:
    parameter_values = model.default_parameter_values

    parameter_values["Current function [A]"] = 0

    sim = pb.Simulation(model, parameter_values=parameter_values)

    solver = pb.CasadiSolver(mode="fast")

    years = 30
    days = years * 365
    hours = days * 24
    minutes = hours * 60
    seconds = minutes * 60

    t_eval = np.linspace(0, seconds, 100)

    sim.solve(t_eval=t_eval, solver=solver)
    sims.append(sim)
pb.dynamic_plot(
    sims,
    [
        "Terminal voltage [V]",
        "Negative particle surface concentration",
        "X-averaged negative particle surface concentration",
        "Electrolyte concentration [mol.m-3]",
        "Total negative electrode sei thickness [m]",
        "X-averaged total negative electrode sei thickness [m]",
        "X-averaged total negative electrode sei thickness",
        "X-averaged negative electrode sei concentration [mol.m-3]",
        "Loss of lithium to negative electrode sei [mol]",
        [
            "Negative electrode sei interfacial current density [A.m-2]",
            "Negative electrode interfacial current density [A.m-2]",
        ],
        [
            "X-averaged negative electrode sei interfacial current density [A.m-2]",
            "X-averaged negative electrode interfacial current density [A.m-2]",
        ],
        "Sum of x-averaged negative electrode interfacial current densities",
        "X-averaged electrolyte concentration",
    ],
)
