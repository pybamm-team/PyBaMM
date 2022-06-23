import pybamm

parameters = ["Marquis2019", "Chen2020"]


class TimeSPMSimulationCCCV:
    param_names = ["parameter"]
    params = parameters

    def time_setup_SPM_simulation(self, parameters):
        self.param = pybamm.ParameterValues(parameters)
        self.model = pybamm.lithium_ion.SPM()
        exp = pybamm.Experiment(
            [
                "Discharge at C/5 for 10 hours or until 3.3 V",
                "Rest for 1 hour",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until 10 mA",
                "Rest for 1 hour",
            ]
        )
        pybamm.Simulation(self.model, parameter_values=self.param, experiment=exp)


class TimeDFNSimulationCCCV:
    param_names = ["parameter"]
    params = parameters

    def time_setup_SPM_simulation(self, parameters):
        self.param = pybamm.ParameterValues(parameters)
        self.model = pybamm.lithium_ion.DFN()
        exp = pybamm.Experiment(
            [
                "Discharge at C/5 for 10 hours or until 3.3 V",
                "Rest for 1 hour",
                "Charge at 1 A until 4.1 V",
                "Hold at 4.1 V until 10 mA",
                "Rest for 1 hour",
            ]
        )
        pybamm.Simulation(self.model, parameter_values=self.param, experiment=exp)


class TimeSPMSimulationGITT:
    param_names = ["parameter"]
    params = parameters

    def time_setup_SPM_simulation(self, parameters):
        self.param = pybamm.ParameterValues(parameters)
        self.model = pybamm.lithium_ion.SPM()
        exp = pybamm.Experiment(
            [("Discharge at C/20 for 1 hour", "Rest for 1 hour")] * 20
        )
        pybamm.Simulation(self.model, parameter_values=self.param, experiment=exp)


class TimeDFNSimulationGITT:
    param_names = ["parameter"]
    params = parameters

    def time_setup_SPM_simulation(self, parameters):
        self.param = pybamm.ParameterValues(parameters)
        self.model = pybamm.lithium_ion.DFN()
        exp = pybamm.Experiment(
            [("Discharge at C/20 for 1 hour", "Rest for 1 hour")] * 20
        )
        pybamm.Simulation(self.model, parameter_values=self.param, experiment=exp)
