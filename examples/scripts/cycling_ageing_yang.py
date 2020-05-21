import pybamm as pb

options = {"sei": "ec reaction limited",
           "sei film resistance": "distributed",
           "porosity": "variable porosity"}
param = pb.ParameterValues(chemistry=pb.parameter_sets.Marquis2019)
model = pb.lithium_ion.DFN(options)
experiment = pb.Experiment(
    [
        "Discharge at {:.4f} C until 3.5 V".format(2),
        "Rest for 5 minutes",
        "Charge at 1 C until 4.2 V",
        "Hold at 4.2 V until C/5",
        "Rest for 5 minutes",
    ]
    * 1
)

sim = pb.Simulation(model, experiment=experiment,
                    parameter_values=param)
sim.solve(solver=pb.CasadiSolver(mode="safe"))
sim.plot(
    [
        "Current [A]",
        "Terminal voltage [V]",
        "Discharge capacity [A.h]",
        "Electrolyte concentration [mol.m-3]",
        "Negative electrode surface potential difference",
        "Total negative electrode sei thickness",
        "Negative electrode porosity",
        "Negative electrode sei interfacial current density",
        "Negative electrode interfacial current density",
        "Sum of negative electrode interfacial current densities",
        #"Negative electrode EC surface concentration",
    ]
)
