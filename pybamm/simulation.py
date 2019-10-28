import pybamm
import numpy as np


class Simulation:
    def __init__(self, model):

        self._model = model
        self._geometry = model.default_geometry
        self._parameter_values = model.default_parameter_values
        self._submesh_types = model.default_submesh_types
        self._var_pts = model.default_var_pts
        self._spatial_methods = model.default_spatial_methods
        self._solver = model.default_solver
        self._quick_plot_vars = None

    @property
    def model(self):
        return self._model

    @property
    def geometry(self):
        return self._geometry

    @geometry.setter
    def geometry(self, geometry):
        self._geometry = geometry

    @property
    def parameter_values(self):
        return self.parameter_values

    @parameter_values.setter
    def parameter_values(self, parameter_values):
        self._parameter_values = parameter_values

    @property
    def submesh_types(self):
        return self._submesh_types

    @submesh_types.setter
    def submesh_types(self, submesh_types):
        # check of correct form
        self._submesh_types = submesh_types

    @property
    def var_pts(self):
        return self._var_pts

    @var_pts.setter
    def var_pts(self, var_pts):
        self._var_pts = var_pts

    @property
    def spatial_methods(self):
        return self._spatial_methods

    @spatial_methods.setter
    def spatial_methods(self, spatial_methods):
        self._spatial_methods = spatial_methods

    @property
    def solver(self):
        return self._solver

    @solver.setter
    def solver(self, solver):
        self._solver = solver

    @property
    def quick_plot_vars(self):
        return self._quick_plot_vars

    @quick_plot_vars.setter
    def quick_plot_vars(self, quick_plot_vars):
        self._quick_plot_vars = quick_plot_vars

    @property
    def solution(self):
        return self._solution

    def solve(self, t_eval=None):
        self._parameter_values.process_model(self._model)
        self._parameter_values.process_geometry(self._geometry)
        self._mesh = pybamm.Mesh(self._geometry, self._submesh_types, self._var_pts)
        self._disc = pybamm.Discretisation(self._mesh, self._spatial_methods)
        self._disc.process_model(self._model)

        if t_eval is None:
            t_eval = np.linspace(0, 1, 100)

        self._solution = self.solver.solve(self._model, t_eval)

    def plot(self):
        plot = pybamm.QuickPlot(
            self._model, self._mesh, self._solution, self._quick_plot_vars
        )
        plot.dynamic_plot()

