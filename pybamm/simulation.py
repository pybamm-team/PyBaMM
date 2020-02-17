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


def constant_current_constant_voltage_constant_power(variables):
    I = variables["Current [A]"]
    V = variables["Terminal voltage [V]"]
    s_I = pybamm.InputParameter("Current switch")
    s_V = pybamm.InputParameter("Voltage switch")
    s_P = pybamm.InputParameter("Power switch")
    n_electrodes_parallel = pybamm.electrical_parameters.n_electrodes_parallel
    n_cells = pybamm.electrical_parameters.n_cells
    return (
        s_I * (I - pybamm.InputParameter("Current input [A]") / n_electrodes_parallel)
        + s_V * (V - pybamm.InputParameter("Voltage input [V]") / n_cells)
        + s_P
        * (
            V * I
            - pybamm.InputParameter("Power input [W]")
            / (n_cells * n_electrodes_parallel)
        )
    )


class Simulation:
    """A Simulation class for easy building and running of PyBaMM simulations.

    Parameters
    ----------
    model : :class:`pybamm.BaseModel`
        The model to be simulated
    experiment : : class:`pybamm.Experiment` (optional)
        The experimental conditions under which to solve the model
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
        experiment=None,
        geometry=None,
        parameter_values=None,
        submesh_types=None,
        var_pts=None,
        spatial_methods=None,
        solver=None,
        quick_plot_vars=None,
        C_rate=None,
    ):
        self._parameter_values = parameter_values or model.default_parameter_values

        if experiment is None:
            self.operating_mode = "without experiment"
            self.C_rate = C_rate
            if self.C_rate:
                self._parameter_values.update({"C-rate": self.C_rate})
            self.model = model
        else:
            self.set_up_experiment(model, experiment)

        self.geometry = geometry or self.model.default_geometry
        self._submesh_types = submesh_types or self.model.default_submesh_types
        self._var_pts = var_pts or self.model.default_var_pts
        self._spatial_methods = spatial_methods or self.model.default_spatial_methods
        self._solver = solver or self.model.default_solver
        self._quick_plot_vars = quick_plot_vars

        self.reset(update_model=False)

        # ignore runtime warnings in notebooks
        if isnotebook():
            import warnings

            warnings.filterwarnings("ignore")

    def set_up_experiment(self, model, experiment):
        """
        Set up a simulation to run with an experiment. This creates a dictionary of
        inputs (current/voltage/power, running time, stopping condition) for each
        operating condition in the experiment. The model will then be solved by
        integrating the model successively with each group of inputs, one group at a
        time.
        """
        self.operating_mode = "with experiment"
        self.model = model.new_copy(
            options={
                **model.options,
                "operating mode": constant_current_constant_voltage_constant_power,
            }
        )
        if not isinstance(experiment, pybamm.Experiment):
            raise TypeError("experiment must be a pybamm `Experiment` instance")
        # Save the experiment
        self.experiment = experiment
        # Update parameter values with experiment parameters
        self._parameter_values.update(experiment.parameters)
        # Create a new submodel for each set of operating conditions and update
        # parameters and events accordingly
        self._experiment_inputs = []
        self._experiment_times = []
        for op, events in zip(experiment.operating_conditions, experiment.events):
            if op[1] in ["A", "C"]:
                # Update inputs for constant current
                if op[1] == "A":
                    I = op[0]
                else:
                    # Scale C-rate with capacity to obtain current
                    capacity = self._parameter_values["Cell capacity [A.h]"]
                    I = op[0] * capacity
                operating_inputs = {
                    "Current switch": 1,
                    "Voltage switch": 0,
                    "Power switch": 0,
                    "Current input [A]": I,
                    "Voltage input [V]": 0,  # doesn't matter
                    "Power input [W]": 0,  # doesn't matter
                }
            elif op[1] == "V":
                # Update inputs for constant voltage
                V = op[0]
                operating_inputs = {
                    "Current switch": 0,
                    "Voltage switch": 1,
                    "Power switch": 0,
                    "Current input [A]": 0,  # doesn't matter
                    "Voltage input [V]": V,
                    "Power input [W]": 0,  # doesn't matter
                }
            elif op[1] == "W":
                # Update inputs for constant power
                P = op[0]
                operating_inputs = {
                    "Current switch": 0,
                    "Voltage switch": 0,
                    "Power switch": 1,
                    "Current input [A]": 0,  # doesn't matter
                    "Voltage input [V]": 0,  # doesn't matter
                    "Power input [W]": P,
                }
            # Update period
            operating_inputs["period"] = op[3]
            # Update events
            if events is None:
                # make current and voltage values that won't be hit
                operating_inputs.update(
                    {"Current cut-off [A]": -1e10, "Voltage cut-off [V]": -1e10}
                )
            elif events[1] in ["A", "C"]:
                # update current cut-off, make voltage a value that won't be hit
                if events[1] == "A":
                    I = events[0]
                else:
                    # Scale C-rate with capacity to obtain current
                    capacity = self._parameter_values["Cell capacity [A.h]"]
                    I = events[0] * capacity
                operating_inputs.update(
                    {"Current cut-off [A]": I, "Voltage cut-off [V]": -1e10}
                )
            elif events[1] == "V":
                # update voltage cut-off, make current a value that won't be hit
                V = events[0]
                operating_inputs.update(
                    {"Current cut-off [A]": -1e10, "Voltage cut-off [V]": V}
                )

            self._experiment_inputs.append(operating_inputs)
            # Convert time to dimensionless
            dt_dimensional = op[2]
            if dt_dimensional is None:
                # max simulation time: 1 week
                dt_dimensional = 7 * 24 * 3600
            tau = self._parameter_values.evaluate(self.model.timescale)
            dt_dimensionless = dt_dimensional / tau
            self._experiment_times.append(dt_dimensionless)

        # add current and voltage events to the model
        # current events both negative and positive to catch specification
        n_electrodes_parallel = pybamm.electrical_parameters.n_electrodes_parallel
        n_cells = pybamm.electrical_parameters.n_cells
        self.model.events.extend(
            [
                pybamm.Event(
                    "Current cut-off (positive) [A] [experiment]",
                    self.model.variables["Current [A]"]
                    - abs(pybamm.InputParameter("Current cut-off [A]"))
                    / n_electrodes_parallel,
                ),
                pybamm.Event(
                    "Current cut-off (negative) [A] [experiment]",
                    self.model.variables["Current [A]"]
                    + abs(pybamm.InputParameter("Current cut-off [A]"))
                    / n_electrodes_parallel,
                ),
                pybamm.Event(
                    "Voltage cut-off [V] [experiment]",
                    self.model.variables["Terminal voltage [V]"]
                    - pybamm.InputParameter("Voltage cut-off [V]") / n_cells,
                ),
            ]
        )

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

    def solve(
        self,
        t_eval=None,
        solver=None,
        external_variables=None,
        inputs=None,
        check_model=True,
    ):
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
        external_variables : dict
            A dictionary of external variables and their corresponding
            values at the current time. The variables must correspond to
            the variables that would normally be found by solving the
            submodels that have been made external.
        inputs : dict, optional
            Any input parameters to pass to the model when solving
        check_model : bool, optional
            If True, model checks are performed after discretisation (see
            :meth:`pybamm.Discretisation.process_model`). Default is True.
        """
        # Setup
        self.build(check_model=check_model)
        if solver is None:
            solver = self.solver

        if self.operating_mode == "without experiment":
            # Solve the normal way, with a single solve
            if t_eval is None:
                try:
                    # Try to compute discharge time
                    tau = self._parameter_values.evaluate(self.model.param.timescale)
                    C_rate = self._parameter_values["C-rate"]
                    t_end = 3600 / tau / C_rate
                    t_eval = np.linspace(0, t_end, 100)
                except AttributeError:
                    t_eval = np.linspace(0, 1, 100)

            self.t_eval = t_eval
            self._solution = solver.solve(self.built_model, t_eval, inputs=inputs)
        elif self.operating_mode == "with experiment":
            if t_eval is not None:
                pybamm.logger.warning(
                    "Ignoring t_eval as solution times are specified by the experiment"
                )
            # Step through all experimental conditions
            inputs = inputs or {}
            pybamm.logger.info("Start running experiment")
            timer = pybamm.Timer()
            for idx, (exp_inputs, dt) in enumerate(
                zip(self._experiment_inputs, self._experiment_times)
            ):
                pybamm.logger.info(self.experiment.operating_conditions_strings[idx])
                inputs.update(exp_inputs)
                # Non-dimensionalise period
                tau = self._parameter_values.evaluate(self.model.timescale)
                freq = exp_inputs["period"] / tau
                # Make sure we take at least 2 timesteps
                npts = max(int(round(dt / freq)) + 1, 2)
                self.step(
                    dt, npts=npts, external_variables=external_variables, inputs=inputs
                )
                # Only allow events specified by experiment
                if not (
                    self._solution.termination == "final time"
                    or "[experiment]" in self._solution.termination
                ):
                    pybamm.logger.warning(
                        """
                        Experiment is infeasible: '{}' was triggered during '{}'. Try
                        reducing current, shortening the time interval, or reducing
                        the period.
                        """.format(
                            self._solution.termination,
                            self.experiment.operating_conditions_strings[idx],
                        )
                    )
                    break
            pybamm.logger.info(
                "Finish experiment simulation, took {}".format(
                    timer.format(timer.time())
                )
            )

    def step(
        self, dt, solver=None, npts=2, external_variables=None, inputs=None, save=True
    ):
        """
        A method to step the model forward one timestep. This method will
        automatically build and set the model parameters if not already done so.

        Parameters
        ----------
        dt : numeric type
            The timestep over which to step the solution
        solver : :class:`pybamm.BaseSolver`
            The solver to use to solve the model.
        npts : int, optional
            The number of points at which the solution will be returned during
            the step dt. default is 2 (returns the solution at t0 and t0 + dt).
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

        if save is False:
            # Don't pass previous solution
            self._solution = solver.step(
                None,
                self.built_model,
                dt,
                npts=npts,
                external_variables=external_variables,
                inputs=inputs,
            )
        else:
            self._solution = solver.step(
                self._solution,
                self.built_model,
                dt,
                npts=npts,
                external_variables=external_variables,
                inputs=inputs,
            )

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

        plot = pybamm.QuickPlot(self._solution, output_variables=quick_plot_vars)

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
            # We currently cannot save models in the 'python' format
            raise NotImplementedError(
                """
                Cannot save simulation if model format is python.
                Set model.convert_to_format = 'casadi' instead.
                """
            )
        # Clear solver problem (not pickle-able, will automatically be recomputed)
        if (
            isinstance(self._solver, pybamm.CasadiSolver)
            and self._solver.problems != {}
        ):
            self._solver.problems = {}
        with open(filename, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def load_sim(filename):
    """Load a saved simulation"""
    return pybamm.load(filename)
