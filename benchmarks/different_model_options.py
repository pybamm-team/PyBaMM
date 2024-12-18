import pybamm
from benchmarks.benchmark_utils import set_random_seed
import numpy as np


def compute_discretisation(model, param):
    var_pts = {
        "x_n": 20,
        "x_s": 20,
        "x_p": 20,
        "r_n": 30,
        "r_p": 30,
        "y": 10,
        "z": 10,
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
    solver: pybamm.BaseSolver
    model: pybamm.BaseModel
    t_eval: np.ndarray
    t_interp: np.ndarray | None

    def solve_setup(self, parameter, model_, option, value, solver_class):
        import importlib

        idaklu_spec = importlib.util.find_spec("pybamm.solvers.idaklu")
        if idaklu_spec is not None:
            try:
                idaklu = importlib.util.module_from_spec(idaklu_spec)
                idaklu_spec.loader.exec_module(idaklu)
            except ImportError as e:  # pragma: no cover
                print("XXXXX cannot find klu", e)
                idaklu_spec = None

        self.solver = solver_class()
        self.model = model_({option: value})
        c_rate = 1
        tmax = 4000 / c_rate
        if self.solver.supports_interp:
            self.t_eval = np.array([0, tmax])
            self.t_interp = None
        else:
            nb_points = 500
            self.t_eval = np.linspace(0, tmax, nb_points)
            self.t_interp = None
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

    def solve_model(self, _model, _params):
        self.solver.solve(self.model, t_eval=self.t_eval, t_interp=self.t_interp)


class TimeBuildModelLossActiveMaterial:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["none", "stress-driven", "reaction-driven", "stress and reaction-driven"],
    )

    def setup(self, _model, _params):
        set_random_seed()

    def time_setup_model(self, model, params):
        build_model("Ai2020", model, "loss of active material", params)


class TimeSolveLossActiveMaterial(SolveModel):
    param_names = ["model", "model option", "solver class"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["none", "stress-driven", "reaction-driven", "stress and reaction-driven"],
        [pybamm.CasadiSolver, pybamm.IDAKLUSolver],
    )

    def setup(self, model, params, solver_class):
        set_random_seed()
        SolveModel.solve_setup(
            self, "Ai2020", model, "loss of active material", params, solver_class
        )

    def time_solve_model(self, _model, _params, _solver_class):
        self.solver.solve(self.model, t_eval=self.t_eval, t_interp=self.t_interp)


class TimeBuildModelLithiumPlating:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["none", "irreversible", "reversible", "partially reversible"],
    )

    def setup(self, _model, _params):
        set_random_seed()

    def time_setup_model(self, model, params):
        build_model("OKane2022", model, "lithium plating", params)


class TimeSolveLithiumPlating(SolveModel):
    param_names = ["model", "model option", "solver class"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["none", "irreversible", "reversible", "partially reversible"],
        [pybamm.CasadiSolver, pybamm.IDAKLUSolver],
    )

    def setup(self, model, params, solver_class):
        set_random_seed()
        SolveModel.solve_setup(
            self, "OKane2022", model, "lithium plating", params, solver_class
        )

    def time_solve_model(self, _model, _params, _solver_class):
        self.solver.solve(self.model, t_eval=self.t_eval, t_interp=self.t_interp)


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

    def setup(self, _model, _params):
        set_random_seed()

    def time_setup_model(self, model, params):
        build_model("Marquis2019", model, "SEI", params)


class TimeSolveSEI(SolveModel):
    param_names = ["model", "model option", "solver class"]
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
            "tunnelling limited",
            "VonKolzenberg2020",
        ],
        [pybamm.CasadiSolver, pybamm.IDAKLUSolver],
    )

    def setup(self, model, params, solver_class):
        set_random_seed()
        SolveModel.solve_setup(self, "Marquis2019", model, "SEI", params, solver_class)

    def time_solve_model(self, _model, _params, _solver_class):
        self.solver.solve(self.model, t_eval=self.t_eval, t_interp=self.t_interp)


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

    def setup(self, _model, _params):
        set_random_seed()

    def time_setup_model(self, model, params):
        build_model("Marquis2019", model, "particle", params)


class TimeSolveParticle(SolveModel):
    param_names = ["model", "model option", "solver class"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        [
            "Fickian diffusion",
            "uniform profile",
            "quadratic profile",
            "quartic profile",
        ],
        [pybamm.CasadiSolver, pybamm.IDAKLUSolver],
    )

    def setup(self, model, params, solver_class):
        set_random_seed()
        SolveModel.solve_setup(
            self, "Marquis2019", model, "particle", params, solver_class
        )

    def time_solve_model(self, _model, _params, _solver_class):
        self.solver.solve(self.model, t_eval=self.t_eval, t_interp=self.t_interp)


class TimeBuildModelThermal:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["isothermal", "lumped", "x-full"],
    )

    def setup(self, _model, _params):
        set_random_seed()

    def time_setup_model(self, model, params):
        build_model("Marquis2019", model, "thermal", params)


class TimeSolveThermal(SolveModel):
    param_names = ["model", "model option", "solver class"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["isothermal", "lumped", "x-full"],
        [pybamm.CasadiSolver, pybamm.IDAKLUSolver],
    )

    def setup(self, model, params, solver_class):
        set_random_seed()
        SolveModel.solve_setup(
            self, "Marquis2019", model, "thermal", params, solver_class
        )

    def time_solve_model(self, _model, _params, _solver_class):
        self.solver.solve(self.model, t_eval=self.t_eval, t_interp=self.t_interp)


class TimeBuildModelSurfaceForm:
    param_names = ["model", "model option"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["false", "differential", "algebraic"],
    )

    def setup(self, _model, _params):
        set_random_seed()

    def time_setup_model(self, model, params):
        build_model("Marquis2019", model, "surface form", params)


class TimeSolveSurfaceForm(SolveModel):
    param_names = ["model", "model option", "solver class"]
    params = (
        [pybamm.lithium_ion.SPM, pybamm.lithium_ion.DFN],
        ["false", "differential", "algebraic"],
        [pybamm.CasadiSolver, pybamm.IDAKLUSolver],
    )

    def setup(self, model, params, solver_class):
        set_random_seed()
        if (model, params, solver_class) == (
            pybamm.lithium_ion.SPM,
            "differential",
            pybamm.IDAKLUSolver,
        ):
            raise NotImplementedError
        SolveModel.solve_setup(
            self, "Marquis2019", model, "surface form", params, solver_class
        )

    def time_solve_model(self, _model, _params, _solver_class):
        self.solver.solve(self.model, t_eval=self.t_eval, t_interp=self.t_interp)
