import pybamm
import numpy as np
import copy


class Simulation:
    def __init__(self, model):
        self.model = model
        self.set_defaults()
        self.reset()

    def set_defaults(self):
        self.geometry = self._model.default_geometry

        self._parameter_values = self._model.default_parameter_values
        self._submesh_types = self._model.default_submesh_types
        self._var_pts = self._model.default_var_pts
        self._spatial_methods = self._model.default_spatial_methods
        self._solver = self._model.default_solver
        self._quick_plot_vars = None

    def reset(self):
        self.model = self._model_class(self._model_options)
        self.geometry = copy.deepcopy(self._unprocessed_geometry)
        self._mesh = None
        self._discretization = None
        self._solution = None
        self._status = "Unprocessed"

    def parameterize(self):
        if self._status == "Unprocessed":
            self._parameter_values.process_model(self._model)
            self._parameter_values.process_geometry(self._geometry)
            self._status = "Parameterized"
        elif self._status == "Built":
            # There is a function to update parameters in the model
            # but this misses some geometric parameters. This class
            # is for convenience and not speed so just re-build.
            self.reset()
            self.build()

    def build(self):
        if self._status != "Built":
            self.parameterize()
            self._mesh = pybamm.Mesh(self._geometry, self._submesh_types, self._var_pts)
            self._disc = pybamm.Discretisation(self._mesh, self._spatial_methods)
            self._disc.process_model(self._model)
            self._model_status = "Built"

    def solve(self, t_eval=None):
        self.build()

        if t_eval is None:
            t_eval = np.linspace(0, 1, 100)

        self._solution = self.solver.solve(self._model, t_eval)

    def plot(self):
        plot = pybamm.QuickPlot(
            self._model, self._mesh, self._solution, self._quick_plot_vars
        )
        plot.dynamic_plot()

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        self._model_class = model.__class__
        self._model_options = model.options

    @property
    def model_options(self):
        return self._model_options

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        self._geometry = geometry
        self._unprocessed_geometry = copy.deepcopy(geometry)

    @property
    def parameter_values(self):
        return self.parameter_values

    @parameter_values.setter
    def parameter_values(self, parameter_values):
        self._parameter_values = parameter_values

        if self._status == "Parameterized":
            self.reset()
            self.parameterize()

        elif self._status == "Built":
            self._parameter_values.update_model(self._model, self._disc)

    @property
    def submesh_types(self):
        return self._submesh_types

    @property
    def var_pts(self):
        return self._var_pts

    @property
    def spatial_methods(self):
        return self._spatial_methods

    @property
    def solver(self):
        return self._solver

    @property
    def quick_plot_vars(self):
        return self._quick_plot_vars

    @property
    def solution(self):
        return self._solution

    def set_specs(
        self,
        model_options=None,
        geometry=None,
        parameter_values=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        quick_plot_vars=None,
    ):

        if model_options:
            self._model_options = model_options

        if geometry:
            self.geometry = geometry

        if parameter_values:
            self._parameter_values = parameter_values
        if submesh_types:
            self._submesh_types = submesh_types
        if var_pts:
            self._var_pts = var_pts
        if spatial_methods:
            self._spatial_methods = spatial_methods
        if solver:
            self._solver = solver
        if quick_plot_vars:
            self._quick_plot_vars = quick_plot_vars

        if self._status == "Parameterized":
            self.reset()
            self.parameterize
        elif self._status == "Built":
            self.reset()
            self.build()

