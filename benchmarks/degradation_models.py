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


def build_model(parameter, model_, option, value):
    param = pybamm.ParameterValues(parameter)
    model = model_({option: value})
    param.process_model(model)
    compute_discretisation(model, param).process_model(model)


class SolveModel:
    solver = pybamm.CasadiSolver()

    def solve_setup(self, parameter, model_, option, value):
        self.model = model_({option: value})
        c_rate = 1
        tmax = 4000 / c_rate
        nb_points = 500
        self.t_eval = np.linspace(0, tmax, nb_points)
        geometry = self.model.default_geometry

        # load parameter values and process model and geometry
        param = pybamm.ParameterValues(parameter)
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

    def solve_model(self, model, params):
        SolveModel.solver.solve(self.model, t_eval=self.t_eval)


class TimeBuildModelLossActiveMaterial:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["none", "stress-driven", "reaction-driven", "stress and reaction-driven"],
    )

    def time_setup_model(self, model, params):

        build_model("Ai2020", model, "loss of active material", params)


class TimeSolveLossActiveMaterial:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["none", "stress-driven", "reaction-driven", "stress and reaction-driven"],
    )

    def setup(self, model, params):
        SolveModel.solve_setup(self, "Ai2020", model, "loss of active material", params)

    def time_solve_model(self, model, params):
        SolveModel.solver.solve(self.model, t_eval=self.t_eval)


class TimeBuildModelLithiumPlating:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["none", "irreversible", "reversible", "partially reversible"],
    )

    def time_setup_model(self, model, params):
        build_model("OKane2022", model, "lithium plating", params)


class TimeSolveLithiumPlating:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["none", "irreversible", "reversible", "partially reversible"],
    )

    def setup(self, model, params):
        SolveModel.solve_setup(self, "OKane2020", model, "lithium plating", params)

    def time_solve_model(self, model, params):
        SolveModel.solver.solve(self.model, t_eval=self.t_eval)


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
        build_model("Marquis2019", model, "SEI", params)


class TimeSolveSEI:
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

    def setup(self, model, params):
        SolveModel.solve_setup(self, "Marquis2019", model, "SEI", params)

    def time_solve_model(self, model, params):
        SolveModel.solver.solve(self.model, t_eval=self.t_eval)


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
        build_model("Marquis2019", model, "particle", params)


class TimeSolveParticle:
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

    def setup(self, model, params):
        SolveModel.solve_setup(self, "Marquis2019", model, "particle", params)

    def time_solve_model(self, model, params):
        SolveModel.solver.solve(self.model, t_eval=self.t_eval)


class TimeBuildModelThermal:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["isothermal", "lumped", "x-lumped", "x-full"],
    )

    def time_setup_model(self, model, params):
        build_model("Marquis2019", model, "thermal", params)


class TimeSolveThermal:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["isothermal", "lumped", "x-lumped", "x-full"],
    )

    def setup(self, model, params):
        SolveModel.solve_setup(self, "Marquis2019", model, "thermal", params)

    def time_solve_model(self, model, params):
        SolveModel.solver.solve(self.model, t_eval=self.t_eval)


class TimeBuildModelSurfaceForm:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["false", "differential", "algebraic"],
    )

    def time_setup_model(self, model, params):
        build_model("Marquis2019", model, "surface form", params)


class TimeSolveSurfaceForm:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["false", "differential", "algebraic"],
    )

    def setup(self, model, params):
        SolveModel.solve_setup(self, "Marquis2019", model, "surface form", params)

    def time_solve_model(self, model, params):
        SolveModel.solver.solve(self.model, t_eval=self.t_eval)
