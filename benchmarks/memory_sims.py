import pybamm
from pybamm.util import set_random_seed

parameters = ["Marquis2019", "Chen2020"]


class MemSPMSimulationCCCV:
    param_names = ["parameter"]
    params = parameters

    def setup(self):
        set_random_seed()

    def mem_setup_SPM_simulationCCCV(self, parameters):
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
        self.sim = pybamm.Simulation(
            self.model, parameter_values=self.param, experiment=exp
        )
        return self.sim


class MemDFNSimulationCCCV:
    param_names = ["parameter"]
    params = parameters

    def mem_setup_DFN_simulationCCCV(self, parameters):
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
        self.sim = pybamm.Simulation(
            self.model, parameter_values=self.param, experiment=exp
        )
        return self.sim


class MemSPMSimulationGITT:
    param_names = ["parameter"]
    params = parameters

    def setup(self):
        set_random_seed()

    def mem_setup_SPM_simulationGITT(self, parameters):
        self.param = pybamm.ParameterValues(parameters)
        self.model = pybamm.lithium_ion.SPM()
        exp = pybamm.Experiment(
            [("Discharge at C/20 for 1 hour", "Rest for 1 hour")] * 20
        )
        self.sim = pybamm.Simulation(
            self.model, parameter_values=self.param, experiment=exp
        )
        return self.sim


class MemDFNSimulationGITT:
    param_names = ["parameter"]
    params = parameters

    def setup(self):
        set_random_seed()

    def mem_setup_DFN_simulationGITT(self, parameters):
        self.param = pybamm.ParameterValues(parameters)
        self.model = pybamm.lithium_ion.SPM()
        exp = pybamm.Experiment(
            [("Discharge at C/20 for 1 hour", "Rest for 1 hour")] * 20
        )
        self.sim = pybamm.Simulation(
            self.model, parameter_values=self.param, experiment=exp
        )
        return self.sim
