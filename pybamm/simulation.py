#
# Simulation class
#
import pickle
import pybamm
import numpy as np
import copy


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class Simulation:
    """A Simulation class for easy building and running of PyBaMM simulations.

    Parameters
    ----------
    model : :class:`pybamm.BaseModel`
        The model to be simulated
    geometry: :class:`pybamm.Geometry` (optional)
        The geometry upon which to solve the model
    parameter_values: dict (optional)
        A dictionary of parameters and their corresponding numerical
        values
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
    C_rate: float (optional)
        The C_rate at which you would like to run a constant current
        experiment at.
    """

    def __init__(
        self,
        model,
        geometry=None,
        parameter_values=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        quick_plot_vars=None,
        C_rate=None,
    ):
        self.model = model

        self.geometry = geometry or model.default_geometry
        self._parameter_values = parameter_values or model.default_parameter_values
        self._submesh_types = submesh_types or model.default_submesh_types
        self._var_pts = var_pts or model.default_var_pts
        self._spatial_methods = spatial_methods or model.default_spatial_methods
        self._solver = solver or self._model.default_solver
        self._quick_plot_vars = quick_plot_vars

        self.C_rate = C_rate
        if self.C_rate:
            self._parameter_values.update({"C-rate": self.C_rate})

        self._made_first_step = False

        self.reset(update_model=False)

        # ignore runtime warnings in notebooks
        if isnotebook():
            import warnings

            warnings.filterwarnings("ignore")

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

    def reset(self, update_model=True):
        """
        A method to reset a simulation back to its unprocessed state.
        """
        if update_model:
            self.model = self.model.new_copy(self._model_options)
        self.geometry = copy.deepcopy(self._unprocessed_geometry)
        self._model_with_set_params = None
        self._built_model = None
        self._mesh = None
        self._disc = None
        self._solution = None
        self._made_first_step = False

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

    def build(self, check_model=True):
        """
        A method to build the model into a system of matrices and vectors suitable for
        performing numerical computations. If the model has already been built or
        solved then this function will have no effect. If you want to rebuild,
        first use "reset()". This method will automatically set the parameters
        if they have not already been set.

        Parameters
        ----------
        check_model : bool, optional
            If True, model checks are performed after discretisation (see
            :meth:`pybamm.Discretisation.process_model`). Default is True.
        """

        if self.built_model:
            return None

        self.set_parameters()
        self._mesh = pybamm.Mesh(self._geometry, self._submesh_types, self._var_pts)
        self._disc = pybamm.Discretisation(self._mesh, self._spatial_methods)
        self._built_model = self._disc.process_model(
            self._model, inplace=False, check_model=check_model
        )

    def solve(self, t_eval=None, solver=None, inputs=None, check_model=True):
        """
        A method to solve the model. This method will automatically build
        and set the model parameters if not already done so.

        Parameters
        ----------
        t_eval : numeric type, optional
            The times at which to compute the solution. If None the model will
            be solved for a full discharge (1 hour / C_rate) if the discharge
            timescale is provided. Otherwise the model will be solved up to a
            non-dimensional time of 1.
        solver : :class:`pybamm.BaseSolver`
            The solver to use to solve the model.
        inputs : dict, optional
            Any input parameters to pass to the model when solving
        check_model : bool, optional
            If True, model checks are performed after discretisation (see
            :meth:`pybamm.Discretisation.process_model`). Default is True.
        """
        self.build(check_model=check_model)

        if t_eval is None:
            try:
                # Try to compute discharge time
                tau = self._parameter_values.evaluate(self.model.param.tau_discharge)
                C_rate = self._parameter_values["C-rate"]
                t_end = 3600 / tau / C_rate
                t_eval = np.linspace(0, t_end, 100)
            except AttributeError:
                t_eval = np.linspace(0, 1, 100)

        if solver is None:
            solver = self.solver

        self.t_eval = t_eval
        self._solution = solver.solve(self.built_model, t_eval, inputs=inputs)

    def step(self, dt, solver=None, external_variables=None, inputs=None, save=True):
        """
        A method to step the model forward one timestep. This method will
        automatically build and set the model parameters if not already done so.

        Parameters
        ----------
        dt : numeric type
            The timestep over which to step the solution
        solver : :class:`pybamm.BaseSolver`
            The solver to use to solve the model.
        external_variables : dict
            A dictionary of external variables and their corresponding
            values at the current time. The variables must correspond to
            the variables that would normally be found by solving the
            submodels that have been made external.
        inputs : dict, optional
            Any input parameters to pass to the model when solving
        save : bool
            Turn on to store the solution of all previous timesteps
        """
        self.build()

        if solver is None:
            solver = self.solver

        solution = solver.step(
            self.built_model, dt, external_variables=external_variables, inputs=inputs
        )

        if save is False or self._made_first_step is False:
            self._solution = solution
        elif self._solution.t[-1] == solution.t[-1]:
            pass
        else:
            self._update_solution(solution)

        self._made_first_step = True

    def _update_solution(self, solution):

        self._solution.set_up_time += solution.set_up_time
        self._solution.solve_time += solution.solve_time
        self._solution.t = np.append(self._solution.t, solution.t[-1])
        self._solution.t_event = solution.t_event
        self._solution.termination = solution.termination
        self._solution.y = np.concatenate(
            [self._solution.y, solution.y[:, -1][:, np.newaxis]], axis=1
        )
        self._solution.y_event = solution.y_event

    def get_variable_array(self, *variables):
        """
        A helper function to easily obtain a dictionary of arrays of values
        for a list of variables at the latest timestep.

        Parameters
        ----------
        variable: str
            The name of the variable/variables you wish to obtain the arrays for.

        Returns
        -------
        variable_arrays: dict
            A dictionary of the variable names and their corresponding
            arrays.
        """

        variable_arrays = [
            self.built_model.variables[var].evaluate(
                self.solution.t[-1], self.solution.y[:, -1]
            )
            for var in variables
        ]

        if len(variable_arrays) == 1:
            return variable_arrays[0]
        else:
            return tuple(variable_arrays)

    def plot(self, quick_plot_vars=None, testing=False):
        """
        A method to quickly plot the outputs of the simulation.

        Parameters
        ----------
        quick_plot_vars: list, optional
            A list of the variables to plot.
        testing, bool, optional
            If False the plot will not be displayed
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

        if isnotebook():
            import ipywidgets as widgets

            widgets.interact(
                plot.plot,
                t=widgets.FloatSlider(min=0, max=plot.max_t, step=0.05, value=0),
            )
        else:
            plot.dynamic_plot(testing=testing)

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
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        quick_plot_vars=None,
        C_rate=None,
    ):
        """
        A method to set the various specs of the simulation. This method
        automatically resets the model after the new specs have been set.

        Parameters
        ----------
        model_options: dict, optional
            A dictionary of options to tweak the model you are using
        geometry: :class:`pybamm.Geometry`, optional
            The geometry upon which to solve the model
        parameter_values: dict, optional
            A dictionary of parameters and their corresponding numerical
            values
        submesh_types: dict, optional
            A dictionary of the types of submesh to use on each subdomain
        var_pts: dict, optional
            A dictionary of the number of points used by each spatial
            variable
        spatial_methods: dict, optional
            A dictionary of the types of spatial method to use on each
            domain (e.g. pybamm.FiniteVolume)
        solver: :class:`pybamm.BaseSolver` (optional)
            The solver to use to solve the model.
        quick_plot_vars: list (optional)
            A list of variables to plot automatically
        C_rate: float (optional)
            The C_rate at which you would like to run a constant current
            experiment at.
        """

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

        if C_rate:
            self.C_rate = C_rate
            self._parameter_values.update({"C-rate": self.C_rate})

        if (
            model_options
            or geometry
            or parameter_values
            or submesh_types
            or var_pts
            or spatial_methods
        ):
            self.reset()

    def save(self, filename):
        """Save simulation using pickle"""
        if self.model.convert_to_format == "python":
            # We currently cannot save models in the 'python'
            raise NotImplementedError(
                """
                Cannot save simulation if model format is python.
                Set model.convert_to_format = 'casadi' instead.
                """
            )
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def load_sim(filename):
    """Load a saved simulation"""
    with open(filename, "rb") as f:
        sim = pickle.load(f)
    return sim
