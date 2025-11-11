import pybamm as pb

pb.set_logging_level("NOTICE")
model = pb.lithium_ion.SPM(
    {
        "SEI": "ec reaction limited",
        "SEI film resistance": "distributed",
        "SEI porosity change": "true",
        "lithium plating": "irreversible",
        "lithium plating porosity change": "true",
    }
)

param = pb.ParameterValues("Mohtat2020")

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
            "Rest for 30 minutes",
        ),
        (
            "Charge at 1 C until 4.2 V",
            "Hold at 4.2 V until C/20",
            "Rest for 30 minutes",
            "Discharge at 1 C until 2.8 V",
            "Rest for 30 minutes",
        ),
        (
            "Charge at 1 C until 4.2 V",
            "Hold at 4.2 V until C/20",
            "Rest for 30 minutes",
            "Discharge at 2 C until 2.8 V",
            "Rest for 30 minutes",
        ),
        (
            "Charge at 1 C until 4.2 V",
            "Hold at 4.2 V until C/20",
            "Rest for 30 minutes",
            "Discharge at 3 C until 2.8 V",
            "Rest for 30 minutes",
        ),
    ]
)

sim = pb.Simulation(model, experiment=experiment, parameter_values=param)
sim.solve()
sim.plot(
    [
        "Current [A]",
        "Total current density [A.m-2]",
        "Voltage [V]",
        "Discharge capacity [A.h]",
        "Electrolyte potential [V]",
        "Electrolyte concentration [mol.m-3]",
        "X-averaged negative electrode porosity",
        "X-averaged negative electrode SEI interfacial current density [A.m-2]",
        "X-averaged negative electrode lithium plating interfacial current density "
        "[A.m-2]",
        "X-averaged negative SEI concentration [mol.m-3]",
        "X-averaged negative dead lithium concentration [mol.m-3]",
        [
            "Total lithium lost [mol]",
            "Loss of lithium to negative SEI [mol]",
            "Loss of lithium to negative lithium plating [mol]",
        ],
    ]
)
