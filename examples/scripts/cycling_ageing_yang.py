import pybamm as pb
options = {"sei": "ec reaction limited",
           "porosity": "variable porosity"}
param = pb.ParameterValues(chemistry=pb.parameter_sets.Ramadass2004)
model = pb.lithium_ion.DFN(options)
experiment = pb.Experiment((
    [
        "Charge at 1 C until 4.2 V",
        "Hold at 4.2 V until C/10",
        "Rest for 5 minutes",
        "Discharge at 2 C until 2.8 V (1 seconds period)",
        "Rest for 5 minutes",
    ]
    * 5 +
    [
        "Charge at 1 C until 4.2 V",
        "Hold at 4.2 V until C/20",
        "Rest for 30 minutes",
        "Discharge at C/3 until 2.8 V(1 seconds period)",
        "Charge at 1 C until 4.2 V",
        "Hold at 4.2 V until C/20",
        "Rest for 30 minutes",
        "Discharge at 1 C until 2.8 V(1 seconds period)",
        "Charge at 1 C until 4.2 V",
        "Hold at 4.2 V until C/20",
        "Rest for 30 minutes",
        "Discharge at 2 C until 2.8 V(1 seconds period)",
        "Charge at 1 C until 4.2 V",
        "Hold at 4.2 V until C/20",
        "Rest for 30 minutes",
        "Discharge at 3 C until 2.8 V(1 seconds period)",
    ]) * 2
)
sim = pb.Simulation(model, experiment=experiment,
                    parameter_values=param)
sim.solve(solver=pb.CasadiSolver(mode="safe"))
sim.plot(
    [
        "Current [A]",
        'Total current density [A.m-2]',
        "Terminal voltage [V]",
        "Discharge capacity [A.h]",
        "Electrolyte potential [V]",
        "Electrolyte concentration [mol.m-3]",
        "Total negative electrode sei thickness",
        "Negative electrode porosity",
        "Negative electrode sei interfacial current density [A.m-2]",
        "X-averaged total negative electrode sei thickness [m]",
    ]
)

