import pybamm as pb

# Note: the Yang model is still in active development and results do not
# match with those reported in the paper

pb.set_logging_level("NOTICE")
model = pb.lithium_ion.Yang2017()

experiment = pb.Experiment(
    [
        (
            "Charge at 1 C until 4.2 V",
            "Hold at 4.2 V until C/10",
            "Rest for 5 minutes",
            "Discharge at 1 C until 2.8 V",
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
sim = pb.Simulation(model, experiment=experiment)
sim.solve(solver=pb.CasadiSolver(mode="fast with events"))
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
        [
            "Total lithium lost [mol]",
            "Loss of lithium to negative electrode SEI [mol]",
        ],
    ]
)
