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


class TimeBuildSPMMarquis2019:
    def __init__(self):
        self.param = pybamm.ParameterValues("Marquis2019")

    def time_setup_SPM(self):
        self.model = pybamm.lithium_ion.SPM()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMORegan2019:
    def __init__(self):
        self.param = pybamm.ParameterValues("ORegan2021")

    def time_setup_SPM(self):
        self.model = pybamm.lithium_ion.SPM()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMNCA_Kim2011:
    def __init__(self):
        self.param = pybamm.ParameterValues("NCA_Kim2011")

    def time_setup_SPM(self):
        self.model = pybamm.lithium_ion.SPM()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMPrada2013:
    def __init__(self):
        self.param = pybamm.ParameterValues("Prada2013")

    def time_setup_SPM(self):
        self.model = pybamm.lithium_ion.SPM()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMAi2020:
    def __init__(self):
        self.param = pybamm.ParameterValues("Ai2020")

    def time_setup_SPM(self):
        self.model = pybamm.lithium_ion.SPM()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMRamadass2004:
    def __init__(self):
        self.param = pybamm.ParameterValues("Ramadass2004")

    def time_setup_SPM(self):
        self.model = pybamm.lithium_ion.SPM()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMMohtat2020:
    def __init__(self):
        self.param = pybamm.ParameterValues("Mohtat2020")

    def time_setup_SPM(self):
        self.model = pybamm.lithium_ion.SPM()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMChen2020:
    def __init__(self):
        self.param = pybamm.ParameterValues("Chen2020")

    def time_setup_SPM(self):
        self.model = pybamm.lithium_ion.SPM()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMChen2020_plating:
    def __init__(self):
        self.param = pybamm.ParameterValues("Chen2020_plating")

    def time_setup_SPM(self):
        self.model = pybamm.lithium_ion.SPM()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMEcker2015:
    def __init__(self):
        self.param = pybamm.ParameterValues("Ecker2015")

    def time_setup_SPM(self):
        self.model = pybamm.lithium_ion.SPM()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)


class TimeBuildSPMeMarquis2019:
    def __init__(self):
        self.param = pybamm.ParameterValues("Marquis2019")

    def time_setup_SPMe(self):
        self.model = pybamm.lithium_ion.SPMe()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMeORegan2021:
    def __init__(self):
        self.param = pybamm.ParameterValues("ORegan2021")

    def time_setup_SPMe(self):
        self.model = pybamm.lithium_ion.SPMe()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMeNCA_Kim2011:
    def __init__(self):
        self.param = pybamm.ParameterValues("NCA_Kim2011")

    def time_setup_SPMe(self):
        self.model = pybamm.lithium_ion.SPMe()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMePrada2013:
    def __init__(self):
        self.param = pybamm.ParameterValues("Prada2013")

    def time_setup_SPMe(self):
        self.model = pybamm.lithium_ion.SPMe()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMeAi2020:
    def __init__(self):
        self.param = pybamm.ParameterValues("Ai2020")

    def time_setup_SPMe(self):
        self.model = pybamm.lithium_ion.SPMe()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMeRamadass2004:
    def __init__(self):
        self.param = pybamm.ParameterValues("Ramadass2004")

    def time_setup_SPMe(self):
        self.model = pybamm.lithium_ion.SPMe()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMeMohtat2020:
    def __init__(self):
        self.param = pybamm.ParameterValues("Mohtat2020")

    def time_setup_SPMe(self):
        self.model = pybamm.lithium_ion.SPMe()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMeChen2020:
    def __init__(self):
        self.param = pybamm.ParameterValues("Chen2020")

    def time_setup_SPMe(self):
        self.model = pybamm.lithium_ion.SPMe()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMeChen2020_plating:
    def __init__(self):
        self.param = pybamm.ParameterValues("Chen2020_plating")

    def time_setup_SPMe(self):
        self.model = pybamm.lithium_ion.SPMe()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMeEcker2015:
    def __init__(self):
        self.param = pybamm.ParameterValues("Ecker2015")

    def time_setup_SPMe(self):
        self.model = pybamm.lithium_ion.SPMe()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildDFNMarquis2019:
    def __init__(self):
        self.param = pybamm.ParameterValues("Marquis2019")

    def time_setup_DFN(self):
        self.model = pybamm.lithium_ion.DFN()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildDFNORegan2021:
    def __init__(self):
        self.param = pybamm.ParameterValues("ORegan2021")

    def time_setup_DFN(self):
        self.model = pybamm.lithium_ion.DFN()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildDFNNCA_Kim2011:
    def __init__(self):
        self.param = pybamm.ParameterValues("NCA_Kim2011")

    def time_setup_DFN(self):
        self.model = pybamm.lithium_ion.DFN()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildDFNPrada2013:
    def __init__(self):
        self.param = pybamm.ParameterValues("Prada2013")

    def time_setup_DFN(self):
        self.model = pybamm.lithium_ion.DFN()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildDFNAi2020:
    def __init__(self):
        self.param = pybamm.ParameterValues("Ai2020")

    def time_setup_DFN(self):
        self.model = pybamm.lithium_ion.DFN()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildDFNRamadass2004:
    def __init__(self):
        self.param = pybamm.ParameterValues("Ramadass2004")

    def time_setup_DFN(self):
        self.model = pybamm.lithium_ion.DFN()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildDFNMohtat2020:
    def __init__(self):
        self.param = pybamm.ParameterValues("Mohtat2020")

    def time_setup_DFN(self):
        self.model = pybamm.lithium_ion.DFN()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildDFNChen2020:
    def __init__(self):
        self.param = pybamm.ParameterValues("Chen2020")

    def time_setup_DFN(self):
        self.model = pybamm.lithium_ion.DFN()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildDFNChen2020_plating:
    def __init__(self):
        self.param = pybamm.ParameterValues("Chen2020_plating")

    def time_setup_DFN(self):
        self.model = pybamm.lithium_ion.DFN()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildDFNEcker2015:
    def __init__(self):
        self.param = pybamm.ParameterValues("Ecker2015")

    def time_setup_DFN(self):
        self.model = pybamm.lithium_ion.DFN()
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)

class TimeBuildSPMSimulationMarquis2019:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Marquis2019")

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

class TimeBuildSPMSimulationORegan2021:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("ORegan2021")

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

class TimeBuildSPMSimulationNCA_Kim2011:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("NCA_Kim2011")

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

class TimeBuildSPMSimulationPrada2013:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Prada2013")

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

class TimeBuildSPMSimulationAi2020:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Ai2020")

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

class TimeBuildSPMSimulationRamadass2004:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Ramadass2004")

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

class TimeBuildSPMSimulationMohtat2020:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Mohtat2020")

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

class TimeBuildSPMSimulationChen2020:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Chen2020")

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

class TimeBuildSPMSimulationChen2020_plating:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Chen2020_plating")

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

class TimeBuildSPMSimulationEcker2015:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Ecker2015")

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

class TimeBuildSPMeSimulationMarquis2019:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Marquis2019")

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

class TimeBuildSPMeSimulationORegan2021:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("ORegan2021")

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

class TimeBuildSPMeSimulationNCA_Kim2011:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("NCA_Kim2011")

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

class TimeBuildSPMeSimulationPrada2013:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Prada2013")

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

class TimeBuildSPMeSimulationAi2020:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Ai2020")

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

class TimeBuildSPMeSimulationRamadass2004:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Ramadass2004")

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

class TimeBuildSPMeSimulationMohtat2020:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Mohtat2020")

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

class TimeBuildSPMeSimulationChen2020:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Chen2020")

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

class TimeBuildSPMeSimulationChen2020_plating:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Chen2020_plating")

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

class TimeBuildSPMeSimulationEcker2015:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Ecker2015")

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

class TimeBuildDFNSimulationMarquis2019:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Marquis2019")

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

class TimeBuildDFNSimulationORegan2021:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("ORegan2021")

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

class TimeBuildDFNSimulationNCA_Kim2011:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("NCA_Kim2011")

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

class TimeBuildDFNSimulationPrada2013:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Prada2013")

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

class TimeBuildDFNSimulationAi2020:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Ai2020")

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

class TimeBuildDFNSimulationRamadass2004:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Ramadass2004")

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

class TimeBuildDFNSimulationMohtat2020:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Mohtat2020")

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

class TimeBuildDFNSimulationChen2020:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Chen2020")

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

class TimeBuildDFNSimulationChen2020_plating:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Chen2020_plating")

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

class TimeBuildDFNSimulationEcker2015:
    # with_experiment
    params = [False, True]

    def __init__(self):
        self.param = pybamm.ParameterValues("Ecker2015")

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

