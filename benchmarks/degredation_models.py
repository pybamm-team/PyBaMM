import pybamm
import numpy as np


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


def solve_model_once(model, solver, t_eval):
    solver.solve(model, t_eval=t_eval)


class TimeBuildModelLossActiveMaterial:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["none", "stress-driven", "reaction-driven", "stress and reaction-driven"],
    )

    def time_setup_model(self, model, params):
        self.param = pybamm.ParameterValues("Ai2020")
        self.model = model({"loss of active material": params})
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)


class TimeBuildSimulationLossActiveMaterial:
    param_names = ["model", "with experiment", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [False, True],
        ["none", "stress-driven", "reaction-driven", "stress and reaction-driven"],
    )

    def time_setup_simulation(self, model, with_experiment, params):
        self.param = pybamm.ParameterValues("Ai2020")
        self.model = model({"loss of active material": params})
        if with_experiment:
            exp = pybamm.Experiment(
                [
                    "Discharge at 0.1C until 3.105 V",
                ]
            )
            pybamm.Simulation(self.model, parameter_values=self.param, experiment=exp)
        else:
            pybamm.Simulation(self.model, parameter_values=self.param, C_rate=1)


class TimeSolveLossActiveMaterial:
    param_names = ["model", "solve first", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [False, True],
        ["none", "stress-driven", "reaction-driven", "stress and reaction-driven"],
    )

    solver = pybamm.CasadiSolver()

    def setup(self, model, solve_first, params):
        self.model = model({"loss of active material": params})
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        geometry = self.model.default_geometry

        # load parameter values and process model and geometry
        param = pybamm.ParameterValues("Ai2020")
        param.process_model(self.model)
        param.process_geometry(geometry)

        # set mesh
        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 30,
            "r_p": 30,
            "y": 10,
            "z": 10,
        }
        mesh = pybamm.Mesh(geometry, self.model.default_submesh_types, var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, self.model.default_spatial_methods)
        disc.process_model(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveLossActiveMaterial.solver, self.t_eval)

    def time_solve_model(self, model, solve_first, params):
        TimeSolveLossActiveMaterial.solver.solve(self.model, t_eval=self.t_eval)


class TimeBuildModelLithiumPlating:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["none", "irreversible", "reversible", "partially reversible"],
    )

    def time_setup_model(self, model, params):
        self.param = pybamm.ParameterValues("OKane2022")
        self.model = model({"lithium plating": params})
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)


class TimeBuildSimulationLithiumPlating:
    param_names = ["model", "with experiment", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [False, True],
        ["none", "irreversible", "reversible", "partially reversible"],
    )

    def time_setup_simulation(self, model, with_experiment, params):
        self.param = pybamm.ParameterValues("OKane2022")
        self.model = model({"lithium plating": params})
        if with_experiment:
            exp = pybamm.Experiment(
                [
                    "Discharge at 0.1C until 3.105 V",
                ]
            )
            pybamm.Simulation(self.model, parameter_values=self.param, experiment=exp)
        else:
            pybamm.Simulation(self.model, parameter_values=self.param, C_rate=1)


class TimeSolveLithiumPlating:
    param_names = ["model", "solve first", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [False, True],
        ["none", "irreversible", "reversible", "partially reversible"],
    )

    solver = pybamm.CasadiSolver()

    def setup(self, model, solve_first, params):
        self.model = model({"lithium plating": params})
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        geometry = self.model.default_geometry

        # load parameter values and process model and geometry
        param = pybamm.ParameterValues("OKane2022")
        param.process_model(self.model)
        param.process_geometry(geometry)

        # set mesh
        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 30,
            "r_p": 30,
            "y": 10,
            "z": 10,
        }
        mesh = pybamm.Mesh(geometry, self.model.default_submesh_types, var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, self.model.default_spatial_methods)
        disc.process_model(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveLithiumPlating.solver, self.t_eval)

    def time_solve_model(self, model, solve_first, params):
        TimeSolveLithiumPlating.solver.solve(self.model, t_eval=self.t_eval)


class TimeBuildModelSEI:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [
            "none",
            "constant",
            "reaction limited",
            "solvent-diffusion limited",
            "electron-migration limited",
            "interstitial-diffusion limited",
            "ec reaction limited",
        ],
    )

    def time_setup_model(self, model, params):
        self.param = pybamm.ParameterValues("Marquis2019")
        self.model = model({"SEI": params})
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)


class TimeBuildSimulationSEI:
    param_names = ["model", "with experiment", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [False, True],
        [
            "none",
            "constant",
            "reaction limited",
            "solvent-diffusion limited",
            "electron-migration limited",
            "interstitial-diffusion limited",
            "ec reaction limited",
        ],
    )

    def time_setup_simulation(self, model, with_experiment, params):
        self.param = pybamm.ParameterValues("Marquis2019")
        self.model = model({"SEI": params})
        if with_experiment:
            exp = pybamm.Experiment(
                [
                    "Discharge at 0.1C until 3.105 V",
                ]
            )
            pybamm.Simulation(self.model, parameter_values=self.param, experiment=exp)
        else:
            pybamm.Simulation(self.model, parameter_values=self.param, C_rate=1)


class TimeSolveSEI:
    param_names = ["model", "solve first", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [False, True],
        [
            "none",
            "constant",
            "reaction limited",
            "solvent-diffusion limited",
            "electron-migration limited",
            "interstitial-diffusion limited",
            "ec reaction limited",
        ],
    )

    solver = pybamm.CasadiSolver()

    def setup(self, model, solve_first, params):
        self.model = model({"SEI": params})
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        geometry = self.model.default_geometry

        # load parameter values and process model and geometry
        param = pybamm.ParameterValues("Marquis2019")
        param.process_model(self.model)
        param.process_geometry(geometry)

        # set mesh
        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 30,
            "r_p": 30,
            "y": 10,
            "z": 10,
        }
        mesh = pybamm.Mesh(geometry, self.model.default_submesh_types, var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, self.model.default_spatial_methods)
        disc.process_model(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSEI.solver, self.t_eval)

    def time_solve_model(self, model, solve_first, params):
        TimeSolveSEI.solver.solve(self.model, t_eval=self.t_eval)


class TimeBuildModelParticle:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [
            "Fickian diffusion",
            "uniform profile",
            "quadratic profile",
            "quartic profile",
        ],
    )

    def time_setup_model(self, model, params):
        self.param = pybamm.ParameterValues("Marquis2019")
        self.model = model({"particle": params})
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)


class TimeBuildSimulationParticle:
    param_names = ["model", "with experiment", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [False, True],
        [
            "Fickian diffusion",
            "uniform profile",
            "quadratic profile",
            "quartic profile",
        ],
    )

    def time_setup_simulation(self, model, with_experiment, params):
        self.param = pybamm.ParameterValues("Marquis2019")
        self.model = model({"particle": params})
        if with_experiment:
            exp = pybamm.Experiment(
                [
                    "Discharge at 0.1C until 3.105 V",
                ]
            )
            pybamm.Simulation(self.model, parameter_values=self.param, experiment=exp)
        else:
            pybamm.Simulation(self.model, parameter_values=self.param, C_rate=1)


class TimeSolveParticle:
    param_names = ["model", "solve first", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [False, True],
        [
            "Fickian diffusion",
            "uniform profile",
            "quadratic profile",
            "quartic profile",
        ],
    )

    solver = pybamm.CasadiSolver()

    def setup(self, model, solve_first, params):
        self.model = model({"particle": params})
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        geometry = self.model.default_geometry

        # load parameter values and process model and geometry
        param = pybamm.ParameterValues("Marquis2019")
        param.process_model(self.model)
        param.process_geometry(geometry)

        # set mesh
        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 30,
            "r_p": 30,
            "y": 10,
            "z": 10,
        }
        mesh = pybamm.Mesh(geometry, self.model.default_submesh_types, var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, self.model.default_spatial_methods)
        disc.process_model(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveParticle.solver, self.t_eval)

    def time_solve_model(self, model, solve_first, params):
        TimeSolveParticle.solver.solve(self.model, t_eval=self.t_eval)


class TimeBuildModelThermal:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["isothermal", "lumped", "x-lumped", "x-full"],
    )

    def time_setup_model(self, model, params):
        self.param = pybamm.ParameterValues("Marquis2019")
        self.model = model({"thermal": params})
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)


class TimeBuildSimulationThermal:
    param_names = ["model", "with experiment", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [False, True],
        ["isothermal", "lumped", "x-lumped", "x-full"],
    )

    def time_setup_simulation(self, model, with_experiment, params):
        self.param = pybamm.ParameterValues("Marquis2019")
        self.model = model({"thermal": params})
        if with_experiment:
            exp = pybamm.Experiment(
                [
                    "Discharge at 0.1C until 3.105 V",
                ]
            )
            pybamm.Simulation(self.model, parameter_values=self.param, experiment=exp)
        else:
            pybamm.Simulation(self.model, parameter_values=self.param, C_rate=1)


class TimeSolveThermal:
    param_names = ["model", "solve first", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [False, True],
        ["isothermal", "lumped", "x-lumped", "x-full"],
    )

    solver = pybamm.CasadiSolver()

    def setup(self, model, solve_first, params):
        self.model = model({"thermal": params})
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        geometry = self.model.default_geometry

        # load parameter values and process model and geometry
        param = pybamm.ParameterValues("Marquis2019")
        param.process_model(self.model)
        param.process_geometry(geometry)

        # set mesh
        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 30,
            "r_p": 30,
            "y": 10,
            "z": 10,
        }
        mesh = pybamm.Mesh(geometry, self.model.default_submesh_types, var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, self.model.default_spatial_methods)
        disc.process_model(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveThermal.solver, self.t_eval)

    def time_solve_model(self, model, solve_first, params):
        TimeSolveThermal.solver.solve(self.model, t_eval=self.t_eval)


class TimeBuildModelSurfaceForm:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["false", "differential", "algebraic"],
    )

    def time_setup_model(self, model, params):
        self.param = pybamm.ParameterValues("Marquis2019")
        self.model = model({"surface form": params})
        self.param.process_model(self.model)
        compute_discretisation(self.model, self.param).process_model(self.model)


class TimeBuildSimulationSurfaceForm:
    param_names = ["model", "with experiment", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [False, True],
        ["false", "differential", "algebraic"],
    )

    def time_setup_simulation(self, model, with_experiment, params):
        self.param = pybamm.ParameterValues("Marquis2019")
        self.model = model({"surface form": params})
        if with_experiment:
            exp = pybamm.Experiment(
                [
                    "Discharge at 0.1C until 3.105 V",
                ]
            )
            pybamm.Simulation(self.model, parameter_values=self.param, experiment=exp)
        else:
            pybamm.Simulation(self.model, parameter_values=self.param, C_rate=1)


class TimeSolveSurfaceForm:
    param_names = ["model", "solve first", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [False, True],
        ["false", "differential", "algebraic"],
    )

    solver = pybamm.CasadiSolver()

    def setup(self, model, solve_first, params):
        self.model = model({"surface form": params})
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        geometry = self.model.default_geometry

        # load parameter values and process model and geometry
        param = pybamm.ParameterValues("Marquis2019")
        param.process_model(self.model)
        param.process_geometry(geometry)

        # set mesh
        var_pts = {
            "x_n": 20,
            "x_s": 20,
            "x_p": 20,
            "r_n": 30,
            "r_p": 30,
            "y": 10,
            "z": 10,
        }
        mesh = pybamm.Mesh(geometry, self.model.default_submesh_types, var_pts)

        # discretise model
        disc = pybamm.Discretisation(mesh, self.model.default_spatial_methods)
        disc.process_model(self.model)
        if solve_first:
            solve_model_once(self.model, TimeSolveSurfaceForm.solver, self.t_eval)

    def time_solve_model(self, model, solve_first, params):
        TimeSolveSurfaceForm.solver.solve(self.model, t_eval=self.t_eval)
