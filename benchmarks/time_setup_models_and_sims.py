import pybamm


def compute_discretisation(model, param):
    var_pts = {
        pybamm.standard_spatial_vars.x_n: 20,
        pybamm.standard_spatial_vars.x_s: 20,
        pybamm.standard_spatial_vars.x_p: 20,
        pybamm.standard_spatial_vars.r_n: 30,
        pybamm.standard_spatial_vars.r_p: 30,
        pybamm.standard_spatial_vars.y: 10,
        pybamm.standard_spatial_vars.z: 10,
    }
    geometry = model.default_geometry
    param.process_geometry(geometry)
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    return pybamm.Discretisation(mesh, model.default_spatial_methods)


class TimeBuildSPM:
    def __init__(self):
        self.param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

    def time_setup_SPM(self):
        self.model = pybamm.lithium_ion.SPM()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)


class TimeBuildSPMe:
    def __init__(self):
        self.param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

    def time_setup_SPMe(self):
        self.model = pybamm.lithium_ion.SPMe()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)


class TimeBuildDFN:
    def __init__(self):
        self.param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

    def time_setup_DFN(self):
        self.model = pybamm.lithium_ion.DFN()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)


class TimeBuildSPMSimulation:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

    def time_setup_SPM_simulation(self, with_experiment):
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
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

    def time_setup_SPMe_simulation(self, with_experiment):
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
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Marquis2019)

    def time_setup_DFN_simulation(self, with_experiment):
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
