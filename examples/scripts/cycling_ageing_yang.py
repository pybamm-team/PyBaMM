import pybamm as pb

pb.set_logging_level("INFO")
options = {"SEI": "ec reaction limited", "SEI porosity change": "true"}
param = pb.ParameterValues(chemistry=pb.parameter_sets.Ramadass2004)
model = pb.lithium_ion.DFN(options)
experiment = pb.Experiment(
    [
        (
            "Charge at 1 C until 4.2 V",
            "Hold at 4.2 V until C/10",
            "Rest for 5 minutes",
            "Discharge at 2 C until 2.8 V",
            "Rest for 5 minutes",
        )
    ]
    * 2
    + [
        (
            "Charge at 1 C until 4.2 V",
            "Hold at 4.2 V until C/20",
            "Rest for 30 minutes",
            "Discharge at C/3 until 2.8 V",
        ),
        (
            "Charge at 1 C until 4.2 V",
            "Hold at 4.2 V until C/20",
            "Rest for 30 minutes",
            "Discharge at 1 C until 2.8 V",
        ),
        (
            "Charge at 1 C until 4.2 V",
            "Hold at 4.2 V until C/20",
            "Rest for 30 minutes",
            "Discharge at 2 C until 2.8 V",
        ),
        (
            "Charge at 1 C until 4.2 V",
            "Hold at 4.2 V until C/20",
            "Rest for 30 minutes",
            "Discharge at 3 C until 2.8 V",
        ),
    ]
)
sim = pb.Simulation(model, experiment=experiment, parameter_values=param)
sim.solve(solver=pb.CasadiSolver(mode="safe", dt_max=120))
sim.plot(
    [
        "Current [A]",
        "Total current density [A.m-2]",
        "Terminal voltage [V]",
        "Discharge capacity [A.h]",
        "Electrolyte potential [V]",
        "Electrolyte concentration [mol.m-3]",
        "Total negative electrode SEI thickness",
        "Negative electrode porosity",
        "X-averaged negative electrode porosity",
        "Negative electrode SEI interfacial current density [A.m-2]",
        "X-averaged total negative electrode SEI thickness [m]",
        "Loss of lithium to negative electrode SEI [mol]",
    ]
)
