import pybamm
import numpy as np
import copy


class Simulation:
    """A Simulation class for easy building and running of PyBaMM simulations.

    Parameters
    ----------
    model : :class:`pybamm.BaseModel`
        The model to be simulated
    geometry: :class:`pybamm.Geometry` (optional)
            The geometry upon which to solve the model
    parameter_values: dict (optional)
        A dictionary of all the parameters and their corresponding numerical
        values. This will totally overwrite all default parameter values.
    update_parameter_values: dict (optional)
        A dictionary of a subset of the parameters and their corresponding
        numerical values. This will only overwrite the parameter values in
        this dictionary and leave the
    submesh_types: dict (optional)
        A dictionary of the types of submesh to use on each subdomain
    var_pts: dict (optional)
        A dictionary of the number of points used by each spatial
        variable
    spatial_methods: dict (optional)
        A dictionary of the types of spatial method to use on each
        domain (e.g. pybamm.FiniteVolume)
    solver: :class:`pybamm.BaseSolver` (optional)
        The solver to use to solve the model.
    quick_plot_vars: list (optional)
        A list of variables to plot automatically
    """

    def __init__(
        self,
        model,
        geometry=None,
        parameter_values=None,
        update_parameter_values=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        quick_plot_vars=None,
    ):
        self.model = model

        self.geometry = geometry or model.default_geometry
        self._parameter_values = parameter_values or model.default_parameter_values
        self._submesh_types = submesh_types or model.default_submesh_types
        self._var_pts = var_pts or model.default_var_pts
        self._spatial_methods = spatial_methods or model.default_spatial_methods
        self._solver = solver or self._model.default_solver
        self._quick_plot_vars = quick_plot_vars

        if update_parameter_values:
            self._parameter_values.update(update_parameter_values)

        self.reset()

    def set_defaults(self):
        """
        A method to set all the simulation specs to default values for the
        supplied model.
        """
        self.geometry = self._model.default_geometry
        self._parameter_values = self._model.default_parameter_values
        self._submesh_types = self._model.default_submesh_types
        self._var_pts = self._model.default_var_pts
        self._spatial_methods = self._model.default_spatial_methods
        self._solver = self._model.default_solver
        self._quick_plot_vars = None

    def reset(self):
        """
        A method to reset a simulation back to its unprocessed state.
        """
        self.model = self._model_class(self._model_options)
        self.geometry = copy.deepcopy(self._unprocessed_geometry)
        self._model_with_set_params = None
        self._built_model = None
        self._mesh = None
        self._disc = None
        self._solution = None

    def set_parameters(self):
        """
        A method to set the parameters in the model and the associated geometry. If
        the model has already been built or solved then this will first reset to the
        unprocessed state and then set the parameter values.
        """

        if self.model_with_set_params:
            return None

        self._model_with_set_params = self._parameter_values.process_model(
            self._model, inplace=True
        )
        self._parameter_values.process_geometry(self._geometry)

    def build(self):
        """
        A method to build the model into a system of matrices and vectors suitable for
        performing numerical computations. If the model has already been built or
        solved then this function will have no effect. If you want to rebuild,
        first use "reset()". This method will automatically set the parameters
        if they have not already been set.
        """

        if self.built_model:
            return None

        self.set_parameters()
        self._mesh = pybamm.Mesh(self._geometry, self._submesh_types, self._var_pts)
        self._disc = pybamm.Discretisation(self._mesh, self._spatial_methods)
        self._built_model = self._disc.process_model(self._model, inplace=False)

    def solve(self, t_eval=None, solver=None):
        """
        A method to solve the model. This method will automatically build
        and set the model parameters if not already done so.

        Parameters
        ----------
        t_eval : numeric type (optional)
            The times at which to compute the solution
        solver : :class:`pybamm.BaseSolver`
            The solver to use to solve the model.
        """
        self.build()

        if t_eval is None:
            t_eval = np.linspace(0, 1, 100)

        if solver is None:
            solver = self.solver

        self._solution = solver.solve(self.built_model, t_eval)

    def plot(self, quick_plot_vars=None):
        """
        A method to quickly plot the outputs of the simulation.

        Parameters
        ----------
        quick_plot_vars: list
            A list of the variables to plot.
        """

        if self._solution is None:
            raise ValueError(
                "Model has not been solved, please solve the model before plotting."
            )

        if quick_plot_vars is None:
            quick_plot_vars = self.quick_plot_vars

        plot = pybamm.QuickPlot(
            self.built_model,
            self._mesh,
            self._solution,
            output_variables=quick_plot_vars,
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
    def model_with_set_params(self):
        return self._model_with_set_params

    @property
    def built_model(self):
        return self._built_model

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
    def unprocessed_geometry(self):
        return self._unprocessed_geometry

    @property
    def parameter_values(self):
        return self._parameter_values

    @property
    def submesh_types(self):
        return self._submesh_types

    @property
    def mesh(self):
        return self._mesh

    @property
    def var_pts(self):
        return self._var_pts

    @property
    def spatial_methods(self):
        return self._spatial_methods

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

    def specs(
        self,
        model_options=None,
        geometry=None,
        parameter_values=None,
        update_parameter_values=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        quick_plot_vars=None,
    ):
        """
        A method to set the various specs of the simulation. This method
        automatically resets the model after the new specs have been set.

        Parameters
        ----------
        model_options: dict (optional)
            A dictionary of options to tweak the model you are using
        geometry: :class:`pybamm.Geometry` (optional)
            The geometry upon which to solve the model
        parameter_values: dict (optional)
            A dictionary of parameters and their corresponding numerical
            values
        update_parameter_values: dict (optional)
            A dictionary of a subset of the parameters and their corresponding
            numerical values. This will only overwrite the parameter values in
            this dictionary and leave the
        submesh_types: dict (optional)
            A dictionary of the types of submesh to use on each subdomain
        var_pts: dict (optional)
            A dictionary of the number of points used by each spatial
            variable
        spatial_methods: dict (optional)
            A dictionary of the types of spatial method to use on each
            domain (e.g. pybamm.FiniteVolume)
        solver: :class:`pybamm.BaseSolver`
            The solver to use to solve the model.
        quick_plot_vars: list
            A list of variables to plot automatically
        """

        if model_options:
            self._model_options = model_options

        if geometry:
            self.geometry = geometry

        if parameter_values:
            self._parameter_values = parameter_values

        if update_parameter_values:
            self._parameter_values.update(update_parameter_values)

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

        if (
            model_options
            or geometry
            or parameter_values
            or submesh_types
            or var_pts
            or spatial_methods
        ):
            self.reset()
