import pybamm as pb

pb.set_logging_level("INFO")
options = {
    "sei": "ec reaction limited",
    "sei porosity change": True,
    "thermal": "x-lumped",
}
param = pb.ParameterValues(chemistry=pb.parameter_sets.Ramadass2004)
param.update(
    {
        "Separator density [kg.m-3]": 397,
        "Separator specific heat capacity [J.kg-1.K-1]": 700,
        "Separator thermal conductivity [W.m-1.K-1]": 0.16,
    },
    check_already_exists=False,
)
model = pb.lithium_ion.DFN(options)
experiment = pb.Experiment(
    [
        "Charge at 0.3 C until 4.2 V",
        "Rest for 5 minutes",
        "Discharge at 1 C until 2.8 V",
        "Rest for 5 minutes",
    ]
    * 5
)
sim = pb.Simulation(model, experiment=experiment, parameter_values=param)
sim.solve(solver=pb.CasadiSolver(mode="safe", dt_max=120))