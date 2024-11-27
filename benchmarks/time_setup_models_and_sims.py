import pybamm
from benchmarks.benchmark_utils import set_random_seed

parameters = [
    "Marquis2019",
    "ORegan2022",
    "NCA_Kim2011",
    "Prada2013",
    "Ai2020",
    "Ramadass2004",
    "Mohtat2020",
    "Chen2020",
    "OKane2022",
    "Ecker2015",
]


def compute_discretisation(model, param):
    var_pts = {
        "negative electrode": 20,
        "separator": 20,
        "positive electrode": 20,
        "negative particle": 30,
        "positive particle": 30,
        "current collector": (10, 10),
    }
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    return pybamm.Discretisation(mesh, model.default_spatial_methods)


class TimeBuildSPM:
    param_names = ["parameter"]
    params = parameters
    param: pybamm.ParameterValues
    model: pybamm.BaseModel

    def setup(self, _params):
        set_random_seed()

    def time_setup_SPM(self, params):
        self.param = pybamm.ParameterValues(params)
        self.model = pybamm.lithium_ion.SPM()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)


class TimeBuildSPMe:
    param_names = ["parameter"]
    params = parameters

    def setup(self, _params):
        set_random_seed()

    def time_setup_SPMe(self, params):
        self.param = pybamm.ParameterValues(params)
        self.model = pybamm.lithium_ion.SPMe()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)


class TimeBuildDFN:
    param_names = ["parameter"]
    params = parameters
    param: pybamm.ParameterValues
    model: pybamm.BaseModel

    def setup(self, _params):
        set_random_seed()

    def time_setup_DFN(self, params):
        self.param = pybamm.ParameterValues(params)
        self.model = pybamm.lithium_ion.DFN()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)


class TimeBuildSPMSimulation:
    param_names = ["with experiment", "parameter"]
    params = ([False, True], parameters)
    param: pybamm.ParameterValues
    model: pybamm.BaseModel

    def setup(self, _with_experiment, _params):
        set_random_seed()

    def time_setup_SPM_simulation(self, with_experiment, params):
        self.param = pybamm.ParameterValues(params)
        self.model = pybamm.lithium_ion.SPM()
        if with_experiment:
            exp = pybamm.Experiment(
                [
                    "Discharge at 0.1C until 3.105 V",
                ]
            )
            pybamm.Simulation(self.model, parameter_values=self.param, experiment=exp)
        else:
            pybamm.Simulation(self.model, parameter_values=self.param, C_rate=1)


class TimeBuildSPMeSimulation:
    param_names = ["with experiment", "parameter"]
    params = ([False, True], parameters)
    param: pybamm.ParameterValues
    model: pybamm.BaseModel

    def setup(self, _with_experiment, _params):
        set_random_seed()

    def time_setup_SPMe_simulation(self, with_experiment, params):
        self.param = pybamm.ParameterValues(params)
        self.model = pybamm.lithium_ion.SPMe()
        if with_experiment:
            exp = pybamm.Experiment(
                [
                    "Discharge at 0.1C until 3.105 V",
                ]
            )
            pybamm.Simulation(self.model, parameter_values=self.param, experiment=exp)
        else:
            pybamm.Simulation(self.model, parameter_values=self.param, C_rate=1)


class TimeBuildDFNSimulation:
    param_names = ["with experiment", "parameter"]
    params = ([False, True], parameters)
    param: pybamm.ParameterValues
    model: pybamm.BaseModel

    def setup(self, _with_experiment, _params):
        set_random_seed()

    def time_setup_DFN_simulation(self, with_experiment, params):
        self.param = pybamm.ParameterValues(params)
        self.model = pybamm.lithium_ion.DFN()
        if with_experiment:
            exp = pybamm.Experiment(
                [
                    "Discharge at 0.1C until 3.105 V",
                ]
            )
            pybamm.Simulation(self.model, parameter_values=self.param, experiment=exp)
        else:
            pybamm.Simulation(self.model, parameter_values=self.param, C_rate=1)
