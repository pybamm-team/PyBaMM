import pybamm as pb
import numpy as np

pb.set_logging_level("INFO")

models = [
    pb.lithium_ion.SPM({"SEI": "reaction limited"}),
    pb.lithium_ion.SPMe({"SEI": "reaction limited"}),
    pb.lithium_ion.SPM(
        {"SEI": "reaction limited", "surface form": "algebraic"},
        name="Algebraic SPM",
    ),
    pb.lithium_ion.SPMe(
        {"SEI": "reaction limited", "surface form": "algebraic"},
        name="Algebraic SPMe",
    ),
    pb.lithium_ion.DFN({"SEI": "reaction limited"}),
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
        "Voltage [V]",
        "Negative particle surface concentration",
        "X-averaged negative particle surface concentration",
        "Electrolyte concentration [mol.m-3]",
        "Negative total SEI thickness [m]",
        "X-averaged negative total SEI thickness [m]",
        "X-averaged negative SEI concentration [mol.m-3]",
        "Sum of x-averaged negative electrode volumetric "
        "interfacial current densities [A.m-3]",
        "Loss of lithium inventory [%]",
        ["Total lithium lost [mol]", "Loss of lithium to negative SEI [mol]"],
    ],
)
