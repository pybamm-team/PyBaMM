import pybamm


class TimeSimulation:
    param_names = ["experiment", "parameter", "model_class", "solver_class"]
    params = [
        ["CCCV", "GITT"],
        ["Marquis2019", "Chen2020"],
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [pybamm.CasadiSolver, pybamm.IDAKLUSolver],
    ]
    experiment_descriptions = {
        "CCCV": [
            "Discharge at C/5 for 10 hours or until 3.3 V",
            "Rest for 1 hour",
            "Charge at 1 A until 4.1 V",
            "Hold at 4.1 V until 10 mA",
            "Rest for 1 hour",
        ],
        "GITT": [("Discharge at C/20 for 1 hour", "Rest for 1 hour")] * 10,
    }

    def setup(self, experiment, parameters, model_class, solver_class):
        if (experiment, parameters, model_class, solver_class) == (
            "GITT",
            "Marquis2019",
            pybamm.lithium_ion.DFN,
            pybamm.CasadiSolver,
        ):
            raise NotImplementedError
        self.param = pybamm.ParameterValues(parameters)
        self.model = model_class()
        self.solver = solver_class()
        self.exp = pybamm.Experiment(self.experiment_descriptions[experiment])
        self.sim = pybamm.Simulation(
            self.model,
            parameter_values=self.param,
            experiment=self.exp,
            solver=self.solver,
        )

    def time_setup(self, experiment, parameters, model_class, solver_class):
        param = pybamm.ParameterValues(parameters)
        model = model_class()
        solver = solver_class()
        exp = pybamm.Experiment(self.experiment_descriptions[experiment])
        pybamm.Simulation(model, parameter_values=param, experiment=exp, solver=solver)

    def time_solve(self, experiment, parameters, model_class, solver_class):
        self.sim.solve()
