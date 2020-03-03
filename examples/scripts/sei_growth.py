import pybamm as pb

pb.set_logging_level("INFO")
# pb.settings.debug_mode = True

options = {"sei": "reaction limited"}
model = pb.lithium_ion.DFN(options)

experiment = pb.Experiment(
    [
        "Discharge at C/10 for 13 hours or until 3.3 V",
        "Rest for 1 hour",
        "Charge at 1 A until 4.1 V",
        "Hold at 4.1 V until 50 mA",
        "Rest for 1 hour",
    ]
    * 10
)

parameter_values = model.default_parameter_values

parameter_values.update(
    {
        "Inner SEI reaction proportion": 0.5,
        "Inner SEI partial molar volume [m3.mol-1]": 95.86e-18,
        "Outer SEI partial molar volume [m3.mol-1]": 95.86e-18,
        "SEI reaction exchange current density [A.m-2]": 1.5e-6,
        "SEI resistance per unit thickness [Ohm.m-1]": 1,
        "Outer SEI solvent diffusivity [m2.s-1]": 2.5e-22,
        "Bulk solvent concentration [mol.m-3]": 2.636e3,
        "Ratio of inner and outer SEI exchange current densities": 1,
        "Inner SEI open-circuit potential [V]": 0.1,
        "Outer SEI open-circuit potential [V]": 0.8,
        "Inner SEI electron conducticity [S.m-1]": 8.95e-14,
        "Inner SEI lithium interstitial diffusivity [m2.s-1]": 1e-15,
        "Lithium interstitial reference concentration [mol.m-3]": 15,
        "Initial inner SEI thickness [m]": 7.5e-14,
    }
)

parameter_values.update(
    {
        "Inner SEI reaction proportion": 0.5,
        "Inner SEI partial molar volume [m3.mol-1]": 1,
        "Outer SEI partial molar volume [m3.mol-1]": 1,
        "SEI reaction exchange current density [A.m-2]": 0.1,
        "SEI resistance per unit thickness [Ohm.m-1]": 0,
        "Outer SEI solvent diffusivity [m2.s-1]": 1,
        "Bulk solvent concentration [mol.m-3]": 1,
        "Ratio of inner and outer SEI exchange current densities": 1,
        "Inner SEI open-circuit potential [V]": 1,
        "Outer SEI open-circuit potential [V]": 1,
        "Inner SEI electron conducticity [S.m-1]": 1,
        "Inner SEI lithium interstitial diffusivity [m2.s-1]": 1,
        "Lithium interstitial reference concentration [mol.m-3]": 1,
        "Initial inner SEI thickness [m]": 1,
    }
)

sim = pb.Simulation(model, parameter_values=parameter_values, experiment=experiment)

solver = pb.CasadiSolver(mode="fast")

sim.solve(solver=solver)
sim.plot(
    [
        "Terminal voltage [V]",
        "Total SEI thickness",
        "X-averaged total SEI thickness",
        "Loss of lithium to SEI [mols]",
        "SEI reaction interfacial current density [A.m-2]",
        "X-averaged SEI reaction interfacial current density [A.m-2]",
    ]
)
